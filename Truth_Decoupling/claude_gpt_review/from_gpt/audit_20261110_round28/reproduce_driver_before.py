"""Execute pinned pre-review driver/export defects on synthetic data only."""
import copy
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent.parent
CODE = REVIEW / 'from_claude/code'
sys.path[:0] = [str(CODE), str(CODE / 'scripts')]
import e43_qa_prompt_das as e43


def original(name, filename):
    loader = importlib.machinery.SourceFileLoader(name, str(HERE / filename))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def main():
    driver = original('r28_original_driver', 'e44_driver_original.py.txt')
    # The original exporter imports the original driver, not the corrected one.
    with patch.dict(sys.modules, {'e44_factorial_das': driver}):
        export = original('r28_original_export', 'e44_export_original.py.txt')
    result = {'scope': 'Synthetic offline defects in pinned 268708c driver/export', 'calls': 0}
    spec = driver.family_spec('C', 'E1')
    hidden = {'conf': torch.tensor([[0., 0.]]), 'caut': torch.tensor([[2., 0.]]),
              'conf_E1': torch.tensor([[10., 0.]]), 'caut_E1': torch.tensor([[8., 0.]])}
    rendered = {name: [{'prefix_text': name}] for name in hidden}
    with patch.object(driver.pd, 'capture_prompt_end', side_effect=lambda b, texts, layer, bs: hidden[texts[0]]):
        got = driver.question_differences(None, spec, rendered, 0, 1)
    result['conditional_C_difference'] = {'actual_conf_to_caut': [2., 0.],
        'actual_confE_to_cautE': [-2., 0.], 'old_driver_used_for_both': got.tolist()}
    assert torch.equal(got, torch.zeros(1, 2))
    train, val = driver.split_train_val([{'pair_id': str(i)} for i in range(3)], 16, 42)
    result['unfulfillable_holdout'] = {'requested': 16, 'available': 3, 'old_train': len(train), 'old_val': len(val)}
    assert len(train) == 3 and not val
    empty = driver.mc_summary([{'pair_id': 'q', 'scored': False, 'abstain_logp': -2.}])
    result['empty_mc_cohort'] = empty
    assert empty['n_scored'] == 0 and empty['mc1'] == 0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); source = root / 'source'; run = root / 'run'
        source.mkdir(); (run / 'gen').mkdir(parents=True); (run / 'dirs').mkdir()
        q = {'pair_id': 'q', 'question': 'Capital of France?', 'gold': ['Paris'],
             'incorrect': ['Lyon'], 'best_answer': 'Paris'}
        (source / 'bank.json').write_text('{}')
        (source / 'questions.json').write_text(json.dumps({'splits': {'train': [], 'tune': [q]}}))
        ident = {'source_files_sha256': {name: e43.file_sha(source / name) for name in ('bank.json', 'questions.json')}}
        signature = driver.pd.fingerprint(ident)
        answer = {'text': 'Paris', 'token_ids': [1], 'n_tokens': 1, 'finished': True}
        frozen = e43.seal({'phase': 'freeze', 'split': 'tune', 'model': 'synthetic', 'protocol': 'synthetic',
            'identity': ident, 'identity_sha256': signature, 'prompt_systems': {'conf': 'Answer'},
            'pilot': False, 'completed': True, 'questions': [{'pair_id': 'q', 'question': q['question'], 'answers': {'conf': answer}}],
            'question_ids': ['q'], 'cohort_sha256': driver.pd.fingerprint([q])})
        driver.save(run / 'frozen_tune.json', frozen)
        (run / 'dirs/weights.pt').write_bytes(b'not even a checkpoint')
        gen = {'identity': ident, 'identity_sha256': signature, 'completed': True, 'pilot': False,
            'phase': 'generate', 'split': 'tune', 'family': 'E', 'layer': 1, 'arm': 'conf.w.p1',
            'direction': 'w', 'beta': 1., 'base_prompt': 'conf', 'frozen_record_sha256': frozen['artifact_sha256'],
            'direction_file': 'weights.pt', 'direction_file_sha256': e43.file_sha(run / 'dirs/weights.pt'),
            'question_ids': ['q'], 'rows': [{'pair_id': 'q', 'question': q['question'], 'text': 'Paris',
                                          'baseline_text': 'Paris', 'finished': True}]}
        changed = copy.deepcopy(gen); duplicate = dict(changed['rows'][0], text='Lyon')
        changed['rows'].append(duplicate)
        driver.save(run / 'gen/E_L1_conf-w-p1_tune.json', e43.seal(changed))
        exported = export.export(source, run, 'tune')
        payload = driver.read(exported['path'])
        result['duplicate_row_export'] = {'exported_answer': payload['evaluations'][0]['outputs']['E_L1_conf-w-p1'],
                                         'duplicate_was_rejected': False, 'direction_bytes_were_valid_checkpoint': False}
        assert result['duplicate_row_export']['exported_answer'] == 'Lyon'
        changed = copy.deepcopy(gen); changed['rows'].append(dict(gen['rows'][0], pair_id='foreign'))
        driver.save(run / 'gen/E_L1_conf-w-p1_tune.json', e43.seal(changed))
        exported = export.export(source, run, 'tune', name='extra')
        result['foreign_row_export'] = {'source_rows': 2, 'exported_rows': exported['n_questions'], 'extra_rejected': False}
        assert exported['n_questions'] == 1
    (HERE / 'driver_before.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

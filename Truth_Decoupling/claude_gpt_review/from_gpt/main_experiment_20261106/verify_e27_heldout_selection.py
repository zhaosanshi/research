#!/usr/bin/env python3
"""Verify the saved pre-test choice against archived bytes, not filename claims.

Reads tar members directly, without extracting arbitrary paths. Also checks
exported scored records against the actual archived source bytes and checks
all model/data/extra-optimization-budget identities of the selected three arms.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile

import torch


def digest(b):
    return hashlib.sha256(b).hexdigest()


def verify(archive, questions_path, test_export, tune_export):
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'pass': bool(ok)})
        if not ok:
            raise ValueError('selection manifest audit failed: ' + name)
    questions_bytes = Path(questions_path).read_bytes()
    questions = json.loads(questions_bytes)
    test, tune = json.loads(Path(test_export).read_text()), json.loads(Path(tune_export).read_text())
    with tarfile.open(archive) as tar:
        def member(name):
            m = tar.getmember(name)
            check('regular_member/' + name, m.isfile())
            return tar.extractfile(m).read()
        prefix = 'qwen3-8b/'
        selection_bytes = member(prefix + 'heldout_selection.json')
        choice = json.loads(selection_bytes)
        check('choice_is_tune_only', choice['selection_split'] == 'tune' and choice['status'] == 'frozen_before_test_generation')
        check('choice_constants', choice['model'] == 'qwen3-8b' and choice['selected_alpha'] == 3
              and choice['layers'] == [34] and choice['epochs'] == 3 and choice['training_pairs'] == 92
              and choice['tune_questions'] == 192 and choice['ranking_weight'] == 0)
        checksums = {}
        for rel, expected in choice['source_sha256'].items():
            actual = digest(member(prefix + rel))
            check('frozen_source_hash/' + rel, actual == expected)
            checksums[rel] = actual
        check('frozen_manifest_complete', len(checksums) == 24)
        tune_counts = {}
        for alpha in (1, 3):
            tune_counts[str(alpha)] = {}
            for init in ('das', 'md', 'random'):
                p = prefix + 'ce_alpha%d_ep3/%s/tune_scored_items.json' % (alpha, init)
                rows = json.loads(member(p))['evaluations']
                name = init + '_generation_adapter'
                check('tune_ids/%s/%s' % (alpha, init), [r['pair_id'] for r in rows] ==
                      [r['pair_id'] for r in questions['splits']['tune']])
                check('tune_resolved/%s/%s' % (alpha, init), all(r['outcome'][name] in ('correct', 'wrong', 'abstain') for r in rows))
                tune_counts[str(alpha)][init] = sum(r['outcome'][name] == 'correct' for r in rows if not r['gold_is_abstention'])
        check('recomputed_alpha_choice', tune_counts['3']['das'] > tune_counts['1']['das'])
        adapters = {}
        for init in ('das', 'md', 'random'):
            folder = prefix + 'ce_alpha3_ep3/' + init + '/'
            binary = member(folder + 'adapter.pt')
            metadata = json.loads(member(folder + 'adapter.json'))
            adapter = torch.load(io.BytesIO(binary), map_location='cpu', weights_only=True)
            check(init + '/manifest_checkpoint_hash', metadata['checkpoint_sha256'] == digest(binary))
            for k, value in adapter.items():
                if k not in ('W', 'initial_W', 'centers'):
                    check(init + '/checkpoint_metadata/' + k, metadata[k] == value)
            check(init + '/method', adapter['method'] == init + '_generation_adapter' and adapter['init'] == init)
            check(init + '/fixed_layers_alpha', adapter['layers'] == [34] and adapter['alpha'] == 3.)
            check(init + '/fixed_train_ids', adapter['training_pair_ids'] == [r['pair_id'] for r in questions['e_train']])
            check(init + '/no_test_or_tune_in_train', not (set(adapter['training_pair_ids']) &
                  {r['pair_id'] for split in ('tune', 'test') for r in questions['splits'][split]}))
            check(init + '/source_questions', adapter['source']['questions_sha256'] == digest(questions_bytes))
            check(init + '/fixed_training_budget', adapter['train_config'] ==
                  {'epochs': 3, 'lr': .001, 'clip': 1., 'seed': 42, 'ranking_weight': 0., 'optimizer': 'Adam', 'alpha': 3.})
            check(init + '/updates', adapter['gradient_audit']['34']['updates'] == 276)
            check(init + '/full_run', adapter['pilot'] is False)
            check(init + '/unit_finite_W', bool(torch.isfinite(adapter['W']['34']).all())
                  and abs(float(adapter['W']['34'].float().norm()) - 1) < 1e-4)
            adapters[init] = metadata
            for split, exported in [('tune', tune), ('test', test)]:
                scored_bytes = member(folder + split + '_scored_items.json')
                data = json.loads(scored_bytes)
                check(init + '/' + split + '/export_file_hash', exported[init]['sha256'] == digest(scored_bytes))
                check(init + '/' + split + '/export_contents', exported[init]['data'] == data)
                check(init + '/' + split + '/export_summary', exported[init]['summary'] == json.loads(member(folder + split + '_scored.json')))
                raw_bytes = member(folder + split + '.json')
                raw = json.loads(raw_bytes)
                check(init + '/' + split + '/generation_hash', data['grading_source_generation_sha256'] == digest(raw_bytes))
                check(init + '/' + split + '/checkpoint_hash', data['training_sha256'] == digest(binary)
                      and raw['training_sha256'] == digest(binary))
                check(init + '/' + split + '/exact_split', [r['pair_id'] for r in data['evaluations']] ==
                      [r['pair_id'] for r in questions['splits'][split]])
                for a, b in zip(data['evaluations'], raw['evaluations']):
                    for field in ('pair_id', 'question', 'gold', 'incorrect', 'outputs', 'finished'):
                        check(init + '/' + split + '/' + a['pair_id'] + '/raw_' + field, a[field] == b[field])
        for init in ('md', 'random'):
            for field in ('model_config', 'config', 'source', 'training_pair_ids', 'training_tokens_sha256',
                          'train_config', 'positions', 'operator', 'precision'):
                check(init + '/common_' + field, adapters[init][field] == adapters['das'][field])
        frozen_mtime = tar.getmember(prefix + 'heldout_selection.json').mtime
        generation_mtimes = {init: tar.getmember(prefix + 'ce_alpha3_ep3/' + init + '/test.json').mtime
                             for init in ('das', 'md', 'random')}
        check('archive_timestamp_order', all(t > frozen_mtime for t in generation_mtimes.values()))
        return {'archive_sha256': digest(Path(archive).read_bytes()),
                'selection_sha256': digest(selection_bytes), 'selection': choice, 'frozen_source_hashes': checksums,
                'tune_correct_counts': tune_counts,
                'training_budget': 'Each initializer receives 92 pairs x 3 extra optimization epochs = 276 updates at L34, alpha3. This does not assert equal total computation including original DAS estimation/training.',
                'source_questions_sha256': digest(questions_bytes),
                'file_mtime_utc': {'heldout_selection': datetime.datetime.fromtimestamp(frozen_mtime, datetime.timezone.utc).isoformat(),
                                   'test_generation_files': {i: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat() for i, t in generation_mtimes.items()}},
                'timestamp_limit': 'Archive modification timestamps agree with the stated order; they are not an independent immutable audit of generation start time.',
                'checks': {'n': len(checks), 'failures': [c for c in checks if not c['pass']]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--source-questions', required=True, type=Path)
    p.add_argument('--test-export', required=True, type=Path)
    p.add_argument('--tune-export', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = verify(args.archive, args.source_questions, args.test_export, args.tune_export)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result['checks']))


if __name__ == '__main__':
    main()

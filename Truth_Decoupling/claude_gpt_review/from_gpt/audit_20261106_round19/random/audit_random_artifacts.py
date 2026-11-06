#!/usr/bin/env python3
"""Read saved E27 tensors and execute local dependency tests. No model/API run.

Default inputs are the previously published artifacts. A second invocation on
the original server can check CPU-RNG equality in its original torch/runtime.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import tarfile

import numpy as np
import torch
import torch.nn.functional as F


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tsha(value):
    x = value.detach().float().cpu().contiguous()
    return sha(json.dumps({'shape': list(x.shape), 'dtype': 'float32'}, sort_keys=True).encode() + x.numpy().tobytes())


def cosine(a, b):
    return float(F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))


def load_pt(tar, path):
    raw = tar.extractfile(path).read()
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), sha(raw)


def artifact_audit(root, archive, pilots):
    records, groups = [], {}
    with tarfile.open(archive) as tar, tarfile.open(pilots) as pilot:
        for model in ('llama3.1-8b', 'qwen3-8b'):
            source = root / 'from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u' / model
            questions = json.loads((source / 'questions.json').read_text())
            source_train = [str(x['pair_id']) for x in questions['e_train']]
            tune = {str(x['pair_id']) for x in questions['splits']['tune']}
            test = {str(x['pair_id']) for x in questions['splits']['test']}
            pilot_bank = np.load(io.BytesIO(pilot.extractfile('e26/' + model + '/pilot3/bank.npz').read()), allow_pickle=False)
            for branch in ('ce_alpha1_ep3', 'ce_alpha3_ep3', 'ce_eos_alpha3_ep3'):
                prefix = model + '/' + branch + '/'
                if prefix + 'random/adapter.pt' not in tar.getnames():
                    continue
                adapters = {}
                for init in ('das', 'md', 'random'):
                    p = prefix + init + '/adapter.pt'
                    d, digest = load_pt(tar, p)
                    adapters[init] = d
                    assert d['init'] == init and d['method'] == init + '_generation_adapter'
                    assert d['training_pair_ids'] == source_train and len(source_train) == 92
                    assert not set(source_train) & (tune | test)
                    assert d['train_config']['epochs'] == 3 and d['train_config']['ranking_weight'] == 0
                    assert d['base_parameter_versions_unchanged']
                    assert d['source']['questions_sha256'] == sha((source / 'questions.json').read_bytes())
                    layer_rows = []
                    for l in d['layers']:
                        initial, final = d['initial_W'][str(l)].float(), d['W'][str(l)].float()
                        grad = d['gradient_audit'][str(l)]
                        assert grad['updates'] == 276 and grad['zero_updates'] == 0
                        assert grad['minimum'] > 0
                        for kind in ('prompt', 'decode'):
                            from_bank = torch.from_numpy(pilot_bank['qa_%s_center_L%d' % (kind, l)]).float()
                            assert torch.equal(d['centers'][str(l)][kind].float(), from_bank)
                        if init in ('das', 'md'):
                            want = torch.from_numpy(pilot_bank['qa_%s_E_L%d' % (init, l)]).float()
                            assert torch.equal(initial, want)
                            initialization = {'exact_match_to_source_bank': True}
                        else:
                            g = torch.Generator(device='cpu').manual_seed(d['train_config']['seed'] + l)
                            want = F.normalize(torch.randn(d['hidden_size'], generator=g), dim=0)
                            initialization = {'seed': d['train_config']['seed'] + l,
                                'local_rng_equal_bitwise': torch.equal(initial, want),
                                'local_rng_max_absolute_error': float((initial - want).abs().max()),
                                'local_rng_cosine': cosine(initial, want),
                                'saved_initial_sha256': tsha(initial), 'recomputed_initial_sha256': tsha(want)}
                            assert initialization['local_rng_max_absolute_error'] < 1e-6
                        layer_rows.append({'layer': l, 'initialization': initialization,
                            'initial_final_cosine': cosine(initial, final),
                            'initial_final_L2': float((initial - final).norm()),
                            'initial_sha256': tsha(initial), 'final_sha256': tsha(final),
                            'gradient_audit': grad})
                    records.append({'artifact': p, 'artifact_sha256': digest, 'model': model, 'init': init,
                        'training_pairs': len(d['training_pair_ids']), 'train_config': d['train_config'],
                        'objective': d['objective'], 'losses': d['losses'], 'layers': layer_rows,
                        'sources': d['source'], 'source_centres_equal_bitwise': True,
                        'base_parameter_versions_unchanged_recorded': True})
                anchor = adapters['das']
                shared_fields = ('config', 'model_config', 'source', 'training_pair_ids', 'training_tokens_sha256',
                                 'train_config', 'positions', 'operator', 'precision', 'layers', 'alpha')
                common = {k: all(adapters[x][k] == anchor[k] for x in ('md', 'random')) for k in shared_fields}
                assert all(common.values())
                groups[prefix] = {'common_fields': common, 'pairwise_directions': {}}
                for l in anchor['layers']:
                    pairs = {}
                    for other in ('das', 'md'):
                        a, b = adapters['random'], adapters[other]
                        pairs[other] = {'initial_cosine': cosine(a['initial_W'][str(l)], b['initial_W'][str(l)]),
                            'final_cosine': cosine(a['W'][str(l)], b['W'][str(l)]),
                            'initial_equal': torch.equal(a['initial_W'][str(l)], b['initial_W'][str(l)]),
                            'final_equal': torch.equal(a['W'][str(l)], b['W'][str(l)])}
                        assert not pairs[other]['initial_equal'] and not pairs[other]['final_equal']
                    groups[prefix]['pairwise_directions'][str(l)] = pairs
        for model in ('llama3.1-8b', 'qwen3-8b'):
            first, _ = load_pt(tar, model + '/ce_alpha1_ep3/random/adapter.pt')
            third, _ = load_pt(tar, model + '/ce_alpha3_ep3/random/adapter.pt')
            assert all(torch.equal(first['initial_W'][str(l)], third['initial_W'][str(l)]) for l in first['layers'])
    return {'artifacts': records, 'shared_control_groups': groups,
            'same_saved_random_initialization_across_alpha_1_3': True}


def dependency_tests(root):
    code, tests = root / 'from_claude/code', root / 'from_gpt'
    sys.path[:0] = [str(tests), str(code / 'scripts'), str(code)]
    import test_qa_das_pipeline_20261105 as fixtures
    from core.model_io import ModelBundle
    from core.qa_generation_adapter import train_generation_adapter
    tok = fixtures.ToyTokenizer()
    model = fixtures.PipelineModel(tok)
    bundle = ModelBundle(model, tok, {'name': 'random-dependency-cpu', 'layers_attr': 'layers', 'chat': {}})
    cfg = {'qa_system': 'Answer briefly.', 'qa_user': 'Question: {question}\nAnswer:', 'max_new_tokens': 2, 'batch_size': 2}
    data = [{'pair_id': 'p1', 'question': 'Where?', 'positive_answer': 'Paris', 'negative_answer': 'Rome'},
            {'pair_id': 'p2', 'question': 'Who?', 'positive_answer': 'A', 'negative_answer': 'Someone else'}]
    a = {0: F.normalize(torch.arange(1., 9.), dim=0)}
    b = {0: F.normalize(torch.arange(8., 0., -1), dim=0)}
    centres = {0: {'prompt': torch.zeros(8), 'decode': torch.arange(8.) * .03}}
    original = [p.detach().clone() for p in model.parameters()]
    def run(initial, examples=data, center=centres):
        return train_generation_adapter(bundle, examples, cfg, initial, center,
            init='random', epochs=2, alpha=3., seed=42, ranking_weight=0., log_fn=None)
    x, y = run(a), run(b)
    assert torch.equal(x['initial_W']['0'], y['initial_W']['0'])
    assert torch.equal(x['W']['0'], y['W']['0']) and x['losses'] == y['losses']
    swapped = copy.deepcopy(data)
    for row in swapped:
        row['positive_answer'], row['negative_answer'] = row['negative_answer'], row['positive_answer']
    z = run(a, swapped)
    assert torch.equal(x['initial_W']['0'], z['initial_W']['0'])
    assert not torch.equal(x['W']['0'], z['W']['0']) and x['losses'] != z['losses']
    changed_centres = {0: {'prompt': centres[0]['prompt'] + .5, 'decode': centres[0]['decode'] - .5}}
    c = run(a, center=changed_centres)
    assert not torch.equal(x['W']['0'], c['W']['0'])
    assert all(torch.equal(old, current) for old, current in zip(original, model.parameters()))
    return {'production_train_function_executed': True,
        'changing_supplied_DAS_vector_leaves_random_initial_final_and_losses_bitwise_equal': True,
        'changing_correct_answer_targets_changes_random_final_and_losses': True,
        'changed_targets_final_vector_L2': float((x['W']['0'] - z['W']['0']).norm()),
        'changing_data_centres_changes_learned_random_vector': True,
        'changed_centres_final_vector_L2': float((x['W']['0'] - c['W']['0']).norm()),
        'base_weights_bitwise_unchanged': True,
        'scope': 'Synthetic CPU causal model exercises actual trainer dependencies; no large-model efficacy claim.'}


def algebra_check(archive):
    with tarfile.open(archive) as tar:
        d, _ = load_pt(tar, 'qwen3-8b/ce_alpha3_ep3/das/adapter.pt')
    l = d['layers'][0]; w = F.normalize(d['W'][str(l)].double(), dim=0); alpha = d['alpha']
    g = torch.Generator().manual_seed(0)
    out = {}
    for phase in ('prompt', 'decode'):
        c = d['centers'][str(l)][phase].double()
        h = c + torch.randn((3, len(w)), generator=g, dtype=torch.float64)
        direct = h + alpha * ((h-c) @ w)[:, None] * w[None, :]
        b = -alpha * (w @ c)
        loreft = h + (h @ ((1+alpha) * w) + b - h @ w)[:, None] * w[None, :]
        error = float((direct-loreft).abs().max())
        assert error < 1e-9
        out[phase] = {'max_absolute_error_float64': error, 'mapped_bias': float(b)}
    return {'definition': 'Phi_LoReFT(h)=h+R^T(W h+b-R h)',
        'mapping': 'R=w^T; W=(1+alpha)w^T; b=-alpha*w^T*c',
        'restriction': 'Unit w only is trained; alpha and centres are fixed. Prompt/decode centres give position-conditioned biases. This is a constrained tied-rank-1 member of the LoReFT operator family, not unrestricted LoReFT.',
        'per_centre_checks': out,
        'not_DAS_source_interchange': 'DAS uses h+w*w^T(s-h) for an external source s; E27 has no external source in its content-training operation.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[3]
    ap.add_argument('--review-root', type=Path, default=default)
    ap.add_argument('--archive', type=Path)
    ap.add_argument('--pilots', type=Path)
    ap.add_argument('--out', type=Path, default=Path(__file__).with_name('random_artifact_audit.json'))
    ap.add_argument('--skip-toy', action='store_true')
    args = ap.parse_args()
    archive = args.archive or args.review_root/'from_gpt/main_experiment_20261106/artifacts/generation_results.tar.gz'
    pilots = args.pilots or args.review_root/'from_gpt/main_experiment_20261106/artifacts/real_pilots.tar.gz'
    result = {'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform()},
              'archive_sha256': sha(archive.read_bytes()), 'pilot_archive_sha256': sha(pilots.read_bytes()),
              **artifact_audit(args.review_root, archive, pilots), 'operator_identity': algebra_check(archive)}
    if not args.skip_toy:
        result['dependency_execution'] = dependency_tests(args.review_root)
    result['production_source_sha256'] = {p: sha((args.review_root/'from_claude/code'/p).read_bytes())
        for p in ('core/qa_generation_adapter.py', 'scripts/e27_qa_generation.py', 'core/qa_das.py')}
    result['interpretation'] = {'random_control_is_untrained': False,
        'random_control_uses_correct_answer_supervision': True,
        'observed_DAS_initial_vector_alias_in_random_control': False,
        'frozen_untrained_random_was_in_published_three_arm_generation': False,
        'scope': 'Only the archived runs and inspected implementation; no claim that every random direction must have zero effect.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'saved': str(args.out), 'artifacts_audited': len(result['artifacts']),
                      'dependency_execution': result.get('dependency_execution'),
                      'operator_identity': result['operator_identity']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

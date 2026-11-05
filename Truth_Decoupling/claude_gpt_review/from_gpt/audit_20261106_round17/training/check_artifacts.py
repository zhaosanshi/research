"""Audit saved training artifacts without loading a model or executing pickle code."""
from pathlib import Path
import hashlib
import json
import os
import torch

ROOT = Path(os.environ.get('TD_AUDIT_CLAUDE_ROOT', str(Path(__file__).resolve().parents[3] / 'from_claude')))
OUT = Path(os.environ.get('TD_AUDIT_OUT', str(Path(__file__).resolve().parent)))
OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / 'results/02_问答DAS_v1'
SEARCH = R / '第16轮_DAS正结果搜索'
reports = []


def read(p):
    return json.loads(p.read_text())


def weight_check(d, hidden):
    w = d['W'].float()
    assert list(w.shape) == [hidden, 1]
    assert torch.isfinite(w).all()
    error = abs(float(w.square().sum()) - 1.0)
    assert error < 2e-5
    assert len(d['losses']) == 10
    assert all(torch.isfinite(torch.tensor(d['losses'])))
    return error


for branch in ['第16轮_全量', '第16轮_DAS正结果搜索/v1u']:
    for model in ['llama3.1-8b', 'qwen3-8b', 'gemma3-4b']:
        run = R / branch / model
        bank, train = read(run / 'bank.json'), read(run / 'training.json')
        expected_fmt = 'user_turn' if branch.endswith('/v1u') else 'same_turn'
        assert train['completed'] is True
        assert set(map(int, train['layers'])) == set(range(bank['num_layers'] - 1))
        assert train['structural_null_layers'] == [bank['num_layers'] - 1]
        maximum = 0.0
        for layer, row in train['layers'].items():
            p = run / row['file']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
            d = torch.load(p, map_location='cpu', weights_only=True)
            maximum = max(maximum, weight_check(d, bank['hidden_size']))
            assert d['model_name'] == model
            assert d['layer'] == int(layer) and d['n_pairs'] == 92
            assert d['training_pair_ids'] == bank['training_pair_ids']
            assert d['bank_fingerprint'] == bank['fingerprint']
            assert d['source_tokens_sha256'] == bank['source_tokens_sha256']
            assert d['losses'] == row['losses']
            assert d['bidirectional'] is True and d['T'] == 1
            assert d['loss_kind'] == 'full_vocabulary_label_cross_entropy'
            assert d['selfcheck_template'].get('format', 'same_turn') == expected_fmt
            assert d['positions'] == d['anchor'] == 'answer_end'
            assert {t['label'] for t in d['position_trace']} == {'true', 'false'}
            for t in d['position_trace']:
                assert t['base_answer_end'] < t['first_label_predictor']
                assert t['source_answer_end'] < t['source_input_length'] - 1
        reports.append({'branch': branch, 'model': model, 'checkpoints': len(train['layers']),
                        'max_unit_norm_error': maximum, 'format': expected_fmt})

for branch in ['tfce', 'tfce1']:
    for run in sorted((SEARCH / branch).iterdir()):
        if not run.is_dir() or not (run / 'E1/summary.json').exists():
            continue
        s = read(run / 'E1/summary.json')
        cfg = read(ROOT / ('code/configs/models/' + run.name + '.json'))
        hidden = {'llama3.1-8b': 4096, 'qwen3-8b': 4096, 'gemma3-4b': 2560}[run.name]
        layers = set()
        maximum = 0.0
        for p in sorted((run / 'E1').glob('das_E_*.pt')):
            d = torch.load(p, map_location='cpu', weights_only=True)
            maximum = max(maximum, weight_check(d, hidden))
            layer = d['layer']; layers.add(layer)
            assert d['n_pairs'] == 96 and d['T'] == 1 and d['rank'] == 1
            assert d['bidirectional'] is (branch == 'tfce')
            assert d['loss_kind'] == 'ce' and d['anchor'] == 'prompt' and d['positions'] == 'last'
            assert d['label_ids'] == s['das']['ce_label_ids']
            assert d['losses'][-1] == s['das']['E_L%d_s42' % layer]['final_loss']
            for t in d['position_trace']:
                assert t['read_src_pos'] == t['len_src'] - 1
                assert t['write_base_pos'] == t['len_base'] - 1
        assert layers == set(s['das']['E_trained_layers']) == set(range(s['num_layers']))
        reports.append({'branch': branch, 'model': run.name, 'checkpoints': len(layers),
                        'max_unit_norm_error': maximum, 'bidirectional': branch == 'tfce'})

result = {'passed': True, 'checkpoint_count': sum(r['checkpoints'] for r in reports),
          'runs': reports, 'limits': 'Metadata, hashes and saved numerical properties only; no large-model training replay.'}
(OUT / 'checkpoint_checks.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

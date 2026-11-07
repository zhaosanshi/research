"""Cross-file checks: dose-3 step-5 EOS mass vs trace replay; server random-init source hashes vs archive adapters."""
import hashlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

tr = load(R19 / 'unicode/evidence/qwen_trace.json')
cd = load(R19 / 'unicode/evidence/qwen_current_dose.json')
for b, bd in zip(tr['batches'], cd['batches']):
    assert b['start'] == bd['start']
    for rp in b['common_prefix_replays']:
        s = rp['step']
        for var in ['adapted', 'adapted_history_only']:
            v = list(rp['variants'][var].values())[0]
            eos = sum(math.exp(t['log_probability']) for t in v['tokens'] if t['id'] in (151643, 151645))
            listed = sorted(t['id'] for t in v['tokens'] if t['id'] in (151643, 151645))
            alpha = 3.0 if var == 'adapted' else 0.0
            st = [x for x in bd['steps'] if x['step'] == s][0]
            dose = [x for x in st['doses'] if x['alpha'] == alpha][0]
            (pid, row), = dose['rows'].items()
            print(f"{pid} step{s} {var}: trace-replay EOS mass (ids {listed})={100 * eos:.4f}% | current-dose alpha={alpha} eos_total={100 * row['eos_total_probability']:.4f}% "
                  f"| trace argmax={v['argmax_id']} dose argmax={row['argmax_id']}")
srv = load(R19 / 'random/server_random_initial_exact.json')
for c in srv['checks']:
    p = X_R18 / c['model'] / f"ce_alpha{c['alpha']}_ep3" / 'random' / 'adapter.pt'
    print(c['model'], 'alpha', c['alpha'], 'seed', c['seed'], 'source sha matches archive random adapter.pt:',
          hashlib.sha256(p.read_bytes()).hexdigest() == c['source_sha256'])

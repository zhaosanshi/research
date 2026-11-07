"""Small side claims in models/review.md and random/report.md."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from safe_pt import load_pt  # noqa

for m in ['llama3.1-8b', 'qwen3-8b']:
    t = load(V1U / m / 'training.json')
    L = t['layers']
    k0 = next(iter(L))
    los = L[k0]['losses']
    print(m, 'loss record example:', str(los)[:200])
    def final(v):
        x = v['losses'][-1]
        return x['loss'] if isinstance(x, dict) else x
    order = sorted(L, key=lambda k: final(L[k]))
    print('   lowest final-epoch training loss layers:', [(k, round(final(L[k]), 4)) for k in order[:3]])
    tune = by_pid(load(X_R18 / m / 'ce_alpha3_ep3/das/tune.json'))
    for p in ['tqa:136', 'tqa:647', 'tqa:685']:
        i = tune[p]['idx']
        print(f'   {p}: tune idx={i} batch start={8 * (i // 8)}')
    lay = '29' if m.startswith('llama') else '34'
    a1 = load_pt(X_R18 / m / 'ce_alpha1_ep3/random/adapter.pt')['initial_W'][lay]['raw']
    a3 = load_pt(X_R18 / m / 'ce_alpha3_ep3/random/adapter.pt')['initial_W'][lay]['raw']
    print('   random initial_W alpha1 == alpha3 bytes:', a1 == a3)

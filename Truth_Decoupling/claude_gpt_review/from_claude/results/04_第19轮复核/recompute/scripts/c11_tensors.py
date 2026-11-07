"""Tensor-level checks without torch: early vs release weights, random/DAS start-end geometry, centre facts."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from safe_pt import load_pt, dot, norm, cos  # noqa

LAYER = {'llama3.1-8b': '29', 'qwen3-8b': '34'}
REL_DIR = {'das': 'das_s42', 'md': 'md_s42', 'random': 'random_s42'}
print('--- early (round-18 archive) vs Claude release adapter tensors')
for m in ['llama3.1-8b', 'qwen3-8b']:
    L = LAYER[m]
    for init in ['das', 'md', 'random']:
        e = load_pt(X_R18 / m / 'ce_alpha3_ep3' / init / 'adapter.pt')
        r = load_pt(RELEASE / m / REL_DIR[init] / 'adapter.pt')
        we, wr = e['W'][L], r['W'][L]
        mad = max(abs(x - y) for x, y in zip(we['values'], wr['values']))
        print(f"  {m} {init}: W bytes equal={we['raw'] == wr['raw']} max_abs_diff={mad:.6g} cos={cos(we['values'], wr['values']):.10f} | "
              f"initial_W equal={e['initial_W'][L]['raw'] == r['initial_W'][L]['raw']} | centers equal="
              f"{all(e['centers'][L][k]['raw'] == r['centers'][L][k]['raw'] for k in ('prompt', 'decode'))} | "
              f"early keys={sorted(e)}")

print('--- start/end geometry of alpha-3 adapters (early archive)')
for m in ['llama3.1-8b', 'qwen3-8b']:
    L = LAYER[m]
    A = {i: load_pt(X_R18 / m / 'ce_alpha3_ep3' / i / 'adapter.pt') for i in ['das', 'md', 'random']}
    for i, a in A.items():
        w0, w1 = a['initial_W'][L]['values'], a['W'][L]['values']
        c = cos(w0, w1)
        dist = norm([x - y for x, y in zip(w0, w1)])
        ang = math.degrees(math.acos(min(1.0, abs(c))))
        print(f"  {m} {i}: |w0|={norm(w0):.6f} |w1|={norm(w1):.6f} cos(w0,w1)={c:.4f} dist={dist:.4f} undirected_angle={ang:.2f} deg")
    # random initial vs DAS-init initial (original DAS direction)
    print(f"  {m}: cos(random initial_W, DAS-init initial_W) = {cos(A['random']['initial_W'][L]['values'], A['das']['initial_W'][L]['values']):+.5f}")
    # centres (DAS-init file) facts from models/review.md section 5
    cp, cd = A['das']['centers'][L]['prompt']['values'], A['das']['centers'][L]['decode']['values']
    j = max(range(len(cd)), key=lambda k: abs(cd[k]))
    share = cd[j] ** 2 / dot(cd, cd)
    w0, w1 = A['das']['initial_W'][L]['values'], A['das']['W'][L]['values']
    diff = [y - x for x, y in zip(cp, cd)]
    alg0 = 3.0 * abs(dot(w0, diff)) * norm(w0)
    alg1 = 3.0 * abs(dot(w1, diff)) * norm(w1)
    print(f"  {m}: |prompt centre|={norm(cp):.2f} |decode centre|={norm(cd):.2f} max |coord| of decode centre at dim {j} "
          f"(value {cd[j]:.2f}) energy share={100 * share:.2f}% | DAS w0[{j}]={w0[j]:+.5f} w1[{j}]={w1[j]:+.5f} | "
          f"alpha3 operator difference (prompt vs decode centre) initial={alg0:.3f} trained={alg1:.3f}")
    # same centres in all three inits?
    print(f"  {m}: centres byte-identical across inits:",
          all(A[i]['centers'][L][k]['raw'] == A['das']['centers'][L][k]['raw'] for i in A for k in ('prompt', 'decode')))
# decode-centre projection along the trained Qwen DAS-init direction (unicode/review.md: 273.2195)
a = load_pt(RELEASE / 'qwen3-8b/das_s42/adapter.pt')
print('--- Qwen release das_s42: w1 . decode_centre =', f"{dot(a['W']['34']['values'], a['centers']['34']['decode']['values']):.4f}",
      '| w1 . prompt_centre =', f"{dot(a['W']['34']['values'], a['centers']['34']['prompt']['values']):.4f}")
a = load_pt(RELEASE / 'llama3.1-8b/das_s42/adapter.pt')
print('--- Llama release das_s42: w1 . decode_centre =', f"{dot(a['W']['29']['values'], a['centers']['29']['decode']['values']):.4f}")

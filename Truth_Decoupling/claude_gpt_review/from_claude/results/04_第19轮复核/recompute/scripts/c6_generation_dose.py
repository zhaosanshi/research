"""Claim group 6: full-generation deployment dose on 192 tune questions (generation_dose_full192)."""
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from np_pcg64 import PCG64, quantile_linear  # noqa

raw = load(X_DOSE / 'tune.json')
sc = load(X_DOSE / 'tune_scored_items.json')
conds = sc['conditions']
print('conditions:', conds, '| raw conditions equal:', raw['conditions'] == conds)
for cs in raw.get('condition_specs', []):
    print('   spec:', {k: cs.get(k) for k in ('name', 'kind', 'layers', 'alpha', 'training_alpha', 'deployment_alpha', 'mode') if k in cs})
ev = sc['evaluations']
n = len(ev)
P = by_pid(sc)
R = by_pid(raw)
assert set(P) == set(R) and n == 192
print('raw texts/finished == scored:', all(P[p]['outputs'] == R[p]['outputs'] and P[p]['finished'] == R[p]['finished'] for p in P),
      '| gold_is_abstention:', sum(e['gold_is_abstention'] for e in ev),
      '| outcome==judge label:', all(e['outcome'][c] == e['judge'][c]['label'] for e in ev for c in conds))
for c in conds:
    k = counts(ev, c)
    bad = [FFFD in e['outputs'][c] for e in ev]
    unf = [not e['finished'][c] for e in ev]
    sym = sum(any(unicodedata.category(ch) == 'So' for ch in e['outputs'][c]) for e in ev)
    mean_chars = sum(len(e['outputs'][c]) for e in ev) / n
    corr = [e['outcome'][c] == 'correct' for e in ev]
    q1 = sum(c1 and not b1 for c1, b1 in zip(corr, bad))
    q2 = sum(c1 and not b1 and not u1 for c1, b1, u1 in zip(corr, bad, unf))
    print(f"  {c:24s} correct={k['correct']} wrong={k['wrong']} abstain={k['abstain']} answered={k['correct'] + k['wrong']} "
          f"U+FFFD={sum(bad)} unfinished={sum(unf)} mean_chars={mean_chars:.2f} So-symbol outputs={sym} | "
          f"correct&noFFFD={q1} correct&noFFFD&finished={q2}")

# bootstrap index matrix exactly as numpy default_rng(0).integers(0, n, size=(2000, n)) (C order)
rng = PCG64(0)
flat = rng.integers(n, 2000 * n)
boot = [flat[i * n:(i + 1) * n] for i in range(2000)]
order = [e['pair_id'] for e in ev]  # scored-file order (GPT's script uses this order)


def compare(a, b):
    """a minus b, per-item correct indicator differences."""
    da = [(P[p]['outcome'][a] == 'correct') - (P[p]['outcome'][b] == 'correct') for p in order]
    plus, minus = sum(x > 0 for x in da), sum(x < 0 for x in da)
    means = [sum(da[j] for j in row) / n for row in boot]
    lo, hi = quantile_linear(means, 0.025), quantile_linear(means, 0.975)
    # simple Wald CI for paired difference as an independent sanity check
    dbar = sum(da) / n
    var = (sum(x * x for x in da) / n - dbar ** 2) / n
    w = 1.959963984540054 * var ** 0.5
    return plus, minus, sum(da), lo, hi, dbar - w, dbar + w


names = {c: c for c in conds}
base = conds[0]
a3 = [c for c in conds if c.endswith('alpha3') and 'adapted' in c][0]
for a in conds[1:]:
    for b in [base, a3]:
        if a == b:
            continue
        plus, minus, net, lo, hi, wl, wh = compare(a, b)
        print(f"  {a} minus {b}: fixed={plus} broken={minus} net={net:+d} rate_delta={100 * net / n:+.2f}pp "
              f"bootstrap95(np seed0, 2000)=[{100 * lo:+.2f}, {100 * hi:+.2f}]pp  Wald=[{100 * wl:+.2f}, {100 * wh:+.2f}]pp "
              f"exact_p={exact_sign_p(plus, minus):.4f}")
for a in conds[1:]:
    print(f"  transitions {base} -> {a}:", transitions(ev, base, a))

# independent data: Claude's release rerun Qwen das_s42 tune outputs and labels
rel = by_pid(load(RELEASE / 'qwen3-8b/das_s42/tune_scored_items.json'))
mapping = {'baseline': 'baseline', 'original_das_alpha3': 'original_das', 'adapted_deploy_alpha3': 'das_generation_adapter'}
for mine, theirs in mapping.items():
    if mine not in conds:
        print('  missing arm', mine)
        continue
    td = [p for p in P if P[p]['outputs'][mine] != rel[p]['outputs'][theirs]]
    fd = [p for p in P if P[p]['finished'][mine] != rel[p]['finished'][theirs]]
    ld = [p for p in P if P[p]['outcome'][mine] != rel[p]['outcome'][theirs]]
    print(f"  vs Claude release das_s42 {theirs}: text_diff={len(td)} finished_diff={len(fd)} label_diff={len(ld)} {ld[:5]}")
# and vs the early round-18 archive labels (early cache)
early = by_pid(load(X_R18 / 'qwen3-8b/ce_alpha3_ep3/das/tune_scored_items.json'))
for mine, theirs in mapping.items():
    td = [p for p in P if P[p]['outputs'][mine] != early[p]['outputs'][theirs]]
    ld = [p for p in P if P[p]['outcome'][mine] != early[p]['outcome'][theirs]]
    print(f"  vs early r18 das {theirs}: text_diff={len(td)} label_diff={len(ld)} {ld}")

# pilot (3 questions)
pil = load(X_DOSE_PILOT / 'pilot_scored_items.json')
pp = by_pid(pil)
print('\npilot conditions:', pil['conditions'], 'items:', sorted(pp))
for p in sorted(pp):
    print('  ', p, ' | '.join(f"{c}: {pp[p]['outputs'][c]!r} [{pp[p]['outcome'][c]}]" for c in pil['conditions']))
    for c in pil['conditions']:
        if c in P[p]['outputs']:
            assert pp[p]['outputs'][c] == P[p]['outputs'][c], (p, c)
print('pilot texts equal full192 texts for same arms: True')

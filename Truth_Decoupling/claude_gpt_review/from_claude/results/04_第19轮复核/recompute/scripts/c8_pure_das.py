"""Claim group 8: Llama pure (frozen) DAS layer / timing controls."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

D = X_PURE / 'llama_pure_das_tune192'
sc = load(D / 'tune_scored_items.json')
raw = load(D / 'tune.json')
conds = sc['conditions']
P, R = by_pid(sc), by_pid(raw)
assert set(P) == set(R) and len(P) == 192
print('raw==scored texts/finished:', all(P[p]['outputs'] == R[p]['outputs'] and P[p]['finished'] == R[p]['finished'] for p in P),
      '| outcome==judge label:', all(P[p]['outcome'][c] == P[p]['judge'][c]['label'] for p in P for c in conds),
      '| gold_is_abstention:', sum(P[p]['gold_is_abstention'] for p in P))
B = 'baseline'
for c in conds:
    k = counts(P.values(), c)
    fixed = sum(P[p]['outcome'][B] != 'correct' and P[p]['outcome'][c] == 'correct' for p in P)
    broken = sum(P[p]['outcome'][B] == 'correct' and P[p]['outcome'][c] != 'correct' for p in P)
    txt = sum(P[p]['outputs'][c] != P[p]['outputs'][B] for p in P)
    bad = sum(FFFD in P[p]['outputs'][c] for p in P)
    unf = sum(not P[p]['finished'][c] for p in P)
    print(f"  {c:28s} correct={k['correct']} wrong={k['wrong']} abstain={k['abstain']} fixed/broken={fixed}/{broken} "
          f"net={fixed - broken:+d} text_changed={txt} U+FFFD={bad} unfinished={unf} exact_p={exact_sign_p(fixed, broken):.4f}")

# old source run (E25 user-turn v1u, sha 94eec7e5...)
old = by_pid(load(V1U / 'llama3.1-8b/tune.json'))
olds = by_pid(load(V1U / 'llama3.1-8b/tune_scored_items.json'))
for mine, theirs in [('baseline', 'baseline'), ('E_qa_das_continuous_L29_a3', 'E_qa_das_continuous_L29_a3')]:
    td = sum(P[p]['outputs'][mine] != old[p]['outputs'][theirs] for p in P)
    fd = sum(P[p]['finished'][mine] != old[p]['finished'][theirs] for p in P)
    ld = sum(P[p]['outcome'][mine] != olds[p]['outcome'][theirs] for p in P)
    jd = sum(P[p]['judge'][mine]['raw'] != olds[p]['judge'][theirs]['raw'] for p in P)
    print(f"  vs old source v1u {theirs}: text_diff={td} finished_diff={fd} label_diff={ld} judge_raw_diff={jd}")
# L10 at alpha 3 does not exist in the old source run (old candidate was L10 alpha 2)
print('  old source conditions:', load(V1U / 'llama3.1-8b/tune.json')['conditions'])
# early round-18 archive arms
r18 = by_pid(load(X_R18 / 'llama3.1-8b/ce_alpha3_ep3/das/tune_scored_items.json'))
for mine, theirs in [('baseline', 'baseline'), ('E_qa_das_continuous_L29_a3', 'original_das')]:
    td = sum(P[p]['outputs'][mine] != r18[p]['outputs'][theirs] for p in P)
    ld = [p for p in P if P[p]['outcome'][mine] != r18[p]['outcome'][theirs]]
    print(f"  vs round-18 early archive {theirs}: text_diff={td} label_diff={len(ld)} {ld}")
# Claude release rerun
rel = by_pid(load(RELEASE / 'llama3.1-8b/das_s42/tune_scored_items.json'))
for mine, theirs in [('baseline', 'baseline'), ('E_qa_das_continuous_L29_a3', 'original_das')]:
    td = sum(P[p]['outputs'][mine] != rel[p]['outputs'][theirs] for p in P)
    fd = sum(P[p]['finished'][mine] != rel[p]['finished'][theirs] for p in P)
    ld = [p for p in P if P[p]['outcome'][mine] != rel[p]['outcome'][theirs]]
    print(f"  vs Claude release das_s42 {theirs}: text_diff={td} finished_diff={fd} label_diff={len(ld)} "
          + ', '.join(f"{p}: pure={P[p]['outcome'][mine]} release={rel[p]['outcome'][theirs]}" for p in ld)
          + f" | release correct={counts(rel.values(), theirs)['correct']}")

# questions whose label differs between any arm and baseline
changed = [p for p in sorted(P, key=lambda q: P[q]['idx']) if any(P[p]['outcome'][c] != P[p]['outcome'][B] for c in conds)]
print('\nlabel-changed questions (any arm vs baseline):', len(changed))
for p in changed:
    print(f" {p} Q={P[p]['question']!r}")
    print(f"    gold={P[p]['gold']} | incorrect={P[p]['incorrect']}")
    for c in conds:
        print(f"    {c:28s} [{P[p]['outcome'][c]:7s} raw={P[p]['judge'][c]['raw']} fin={P[p]['finished'][c]}] {P[p]['outputs'][c]!r}")

# pilot3
pil = by_pid(load(X_PURE / 'llama_pure_das_pilot3/tune_scored_items.json'))
print('\npilot3 items:', sorted(pil))
for p in sorted(pil):
    print('  ', p, ' | '.join(f"{c}: {pil[p]['outputs'][c]!r}[{pil[p]['outcome'][c]}]" for c in conds))

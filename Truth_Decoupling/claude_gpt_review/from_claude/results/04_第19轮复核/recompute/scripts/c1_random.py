"""Claim group 1: frozen random control (192 tune questions, early judge cache)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

ARMS = ['baseline', 'original_das', 'frozen_random_initial', 'trained_random', 'trained_das']
# frozen-control arm -> (round-18 run dir, condition name in that run)
R18_MAP = {
    'baseline': [('das', 'baseline'), ('random', 'baseline'), ('md', 'baseline')],
    'original_das': [('das', 'original_das'), ('random', 'original_das'), ('md', 'original_das')],
    'trained_random': [('random', 'random_generation_adapter')],
    'trained_das': [('das', 'das_generation_adapter')],
}

for model in ['llama3.1-8b', 'qwen3-8b']:
    print('=' * 20, model)
    sc = load(X_RANDOM / model / 'full192/tune_scored_items.json')
    raw = load(X_RANDOM / model / 'full192/tune.json')
    assert sc['conditions'] == ARMS == raw['conditions'], sc['conditions']
    ev = sc['evaluations']
    assert len(ev) == 192 and len({e['pair_id'] for e in ev}) == 192
    # scored file carries the same texts / finished flags as the raw generation file
    rawp = by_pid(raw)
    assert all(rawp[e['pair_id']]['outputs'] == e['outputs'] and rawp[e['pair_id']]['finished'] == e['finished'] for e in ev)
    print('gold_is_abstention items:', sum(e['gold_is_abstention'] for e in ev))
    print('outcome == judge label for all arms:', all(e['outcome'][c] == e['judge'][c]['label'] for e in ev for c in ARMS))
    for c in ARMS:
        k = counts(ev, c)
        fffd = sum(FFFD in e['outputs'][c] for e in ev)
        unf = sum(not e['finished'][c] for e in ev)
        both = sum((FFFD in e['outputs'][c]) and (not e['finished'][c]) for e in ev)
        print(f"  {c:22s} correct={k['correct']:3d} wrong={k['wrong']:3d} abstain={k['abstain']:3d} "
              f"answered={k['correct'] + k['wrong']:3d} U+FFFD={fffd:2d} unfinished={unf:2d} fffd&unfinished={both}")
    P = by_pid(sc)
    for a, b in [('baseline', 'frozen_random_initial'), ('frozen_random_initial', 'trained_random'),
                 ('frozen_random_initial', 'trained_das'), ('baseline', 'trained_random'), ('baseline', 'trained_das'),
                 ('baseline', 'original_das')]:
        fixed = [p for p in P if P[p]['outcome'][a] != 'correct' and P[p]['outcome'][b] == 'correct']
        broken = [p for p in P if P[p]['outcome'][a] == 'correct' and P[p]['outcome'][b] != 'correct']
        txt = sum(P[p]['outputs'][a] != P[p]['outputs'][b] for p in P)
        print(f"  {b} vs {a}: fixed={len(fixed)} broken={len(broken)} net={len(fixed) - len(broken):+d} "
              f"exact_p={exact_sign_p(len(fixed), len(broken)):.5f} text_changed={txt}")
        if a == 'baseline' and b == 'frozen_random_initial':
            for p in sorted(P, key=lambda q: P[q]['idx']):
                if P[p]['outputs'][a] != P[p]['outputs'][b] or P[p]['outcome'][a] != P[p]['outcome'][b]:
                    print('     changed:', p, repr(P[p]['outputs'][a]), P[p]['outcome'][a], '->', repr(P[p]['outputs'][b]), P[p]['outcome'][b])
    # item-by-item reproduction of round-18 archive arms
    for arm, sources in R18_MAP.items():
        for run, cond in sources:
            old = by_pid(load(X_R18 / model / 'ce_alpha3_ep3' / run / 'tune_scored_items.json'))
            assert set(old) == set(P)
            diff = {k: 0 for k in ('text', 'finished', 'outcome', 'judge_label', 'judge_raw', 'rule_abstain', 'outcome_judge_only', 'outcome_legacy')}
            for p in P:
                n, o = P[p], old[p]
                diff['text'] += n['outputs'][arm] != o['outputs'][cond]
                diff['finished'] += n['finished'][arm] != o['finished'][cond]
                diff['outcome'] += n['outcome'][arm] != o['outcome'][cond]
                diff['judge_label'] += n['judge'][arm]['label'] != o['judge'][cond]['label']
                diff['judge_raw'] += n['judge'][arm]['raw'] != o['judge'][cond]['raw']
                diff['rule_abstain'] += n['rule_abstain'][arm] != o['rule_abstain'][cond]
                diff['outcome_judge_only'] += n['outcome_judge_only'][arm] != o['outcome_judge_only'][cond]
                diff['outcome_legacy'] += n['outcome_legacy'][arm] != o['outcome_legacy'][cond]
            print(f"  reproduce r18 {run}/{cond} -> {arm}: differing items {diff}")
    # pilot3 vs full192 on the 3 pilot items
    pil = load(X_RANDOM / model / 'pilot3/tune_scored_items.json')
    pp = by_pid(pil)
    print('  pilot3 items:', sorted(pp), 'conditions', pil['conditions'])
    print('  pilot3 texts == full192 texts:', all(pp[p]['outputs'][c] == P[p]['outputs'][c] for p in pp for c in ARMS),
          ' labels equal:', all(pp[p]['outcome'][c] == P[p]['outcome'][c] for p in pp for c in ARMS))
    print('  reproduction_mismatches field:', sc.get('reproduction_mismatches'),
          ' base_parameter_versions_and_flags_unchanged:', sc.get('base_parameter_versions_and_flags_unchanged'))

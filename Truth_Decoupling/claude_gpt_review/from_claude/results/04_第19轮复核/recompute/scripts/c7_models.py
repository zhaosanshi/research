"""Claim group 7: Llama vs Qwen decomposition from the round-18 archive (alpha3, DAS-init content adapter)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

MODELS = ['llama3.1-8b', 'qwen3-8b']
ADP = 'das_generation_adapter'
S = {}
for m in MODELS:
    S[m] = by_pid(load(X_R18 / m / 'ce_alpha3_ep3/das/tune_scored_items.json'))
assert set(S[MODELS[0]]) == set(S[MODELS[1]])
ids = sorted(S[MODELS[0]])
print('same 192 tune ids in both models:', len(ids) == 192)
for m in MODELS:
    ev = S[m]
    k = {c: counts(ev.values(), c)['correct'] for c in ['baseline', 'original_das', ADP]}
    t = transitions(ev, 'baseline', ADP)
    fixed = sum(v for kk, v in t.items() if kk.endswith('->correct') and not kk.startswith('correct'))
    broken = sum(v for kk, v in t.items() if kk.startswith('correct->') and not kk.endswith('correct'))
    print(f"{m}: baseline={k['baseline']} original_das={k['original_das']} content_trained={k[ADP]} "
          f"(adapter - original_das = {k[ADP] - k['original_das']:+d}); baseline abstentions={counts(ev.values(), 'baseline')['abstain']}")
    print(f"   transitions baseline->adapter: {t}  net={fixed - broken:+d}")
    t2 = transitions(ev, 'baseline', 'original_das')
    print(f"   transitions baseline->original_das: {t2}")
# both models originally answered (baseline not abstain in both)
both_ans = [p for p in ids if all(S[m][p]['outcome']['baseline'] != 'abstain' for m in MODELS)]
both_wrong = [p for p in ids if all(S[m][p]['outcome']['baseline'] == 'wrong' for m in MODELS)]
print('both originally answered:', len(both_ans), '| both originally wrong:', len(both_wrong))
for m in MODELS:
    ev = S[m]
    f = sum(ev[p]['outcome']['baseline'] != 'correct' and ev[p]['outcome'][ADP] == 'correct' for p in both_ans)
    b = sum(ev[p]['outcome']['baseline'] == 'correct' and ev[p]['outcome'][ADP] != 'correct' for p in both_ans)
    fw = sum(ev[p]['outcome'][ADP] == 'correct' for p in both_wrong)
    print(f"  {m}: on both-answered fixed={f} broken={b} net={f - b:+d}; on both-wrong fixed={fw}")

# models/review.md section 1 side claims
for m in MODELS:
    for alpha in ['ce_alpha1_ep3', 'ce_alpha3_ep3']:
        runs = {i: by_pid(load(X_R18 / m / alpha / i / 'tune_scored_items.json')) for i in ['das', 'md', 'random']}
        same = all(runs['das'][p][f][c] == runs[i][p][f][c] for i in ['md', 'random'] for p in ids
                   for c in ['baseline', 'original_das'] for f in ['outputs', 'finished', 'outcome'])
        print(f"  {m} {alpha}: baseline/original_das texts+finished+labels identical across das/md/random runs: {same}")
    a3 = by_pid(load(X_R18 / m / 'ce_alpha3_ep3/das/tune.json'))
    v1u = by_pid(load(V1U / m / 'tune.json'))
    v1us = by_pid(load(V1U / m / 'tune_scored_items.json'))
    cond = 'E_qa_das_continuous_L29_a3' if m.startswith('llama') else 'E_qa_das_continuous_L34_a3'
    td = sum(a3[p]['outputs']['original_das'] != v1u[p]['outputs'][cond] for p in ids)
    fd = sum(a3[p]['finished']['original_das'] != v1u[p]['finished'][cond] for p in ids)
    bd = sum(a3[p]['outputs']['baseline'] != v1u[p]['outputs']['baseline'] for p in ids)
    ld = sum(S[m][p]['outcome']['original_das'] != v1us[p]['outcome'][cond] for p in ids)
    print(f"  {m}: r18 original_das vs E25 v1u {cond}: text_diff={td} finished_diff={fd} label_diff={ld}; baseline text_diff={bd}")
    # old candidates' net correct vs baseline in the v1u tune run
    for c in [x for x in load(V1U / m / 'tune_scored_items.json')['conditions'] if x.startswith('E_qa_das')]:
        f = sum(v1us[p]['outcome']['baseline'] != 'correct' and v1us[p]['outcome'][c] == 'correct' for p in ids)
        b = sum(v1us[p]['outcome']['baseline'] == 'correct' and v1us[p]['outcome'][c] != 'correct' for p in ids)
        print(f"     v1u {c}: net correct vs baseline {f - b:+d} ({f}/{b})")
    tr = [load(X_R18 / m / 'ce_alpha3_ep3' / i / 'adapter.json')['training_pair_ids'] for i in ['das', 'md', 'random']]
    S.setdefault('_train', {})[m] = tr[0]
    print(f"  {m}: training ids n={len(tr[0])}, identical across inits={tr[0] == tr[1] == tr[2]}, overlap with tune={len(set(tr[0]) & set(ids))}")
print('training ids identical across models:', S['_train'][MODELS[0]] == S['_train'][MODELS[1]])

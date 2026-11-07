"""Claim group 2: U+FFFD facts in the Qwen DAS-init alpha3 adapter arm (early archive and Claude release rerun)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

COND = 'das_generation_adapter'
SOURCES = {
    'early_r18': X_R18 / 'qwen3-8b/ce_alpha3_ep3/das',
    'release_claude': RELEASE / 'qwen3-8b/das_s42',
}
abst_sets = {}
for tag, d in SOURCES.items():
    for split in ['tune', 'test']:
        raw = load(d / f'{split}.json')
        sc = load(d / f'{split}_scored_items.json')
        assert COND in raw['conditions']
        ev = raw['evaluations']
        n = len(ev)
        bad = [e for e in ev if FFFD in e['outputs'][COND]]
        fin = [e for e in bad if e['finished'][COND]]
        unf_all = [e for e in ev if not e['finished'][COND]]
        inter = [e for e in bad if not e['finished'][COND]]
        scp = by_pid(sc)
        assert set(scp) == {e['pair_id'] for e in ev}
        # raw generation texts == scored texts
        same = all(scp[e['pair_id']]['outputs'] == e['outputs'] and scp[e['pair_id']]['finished'] == e['finished'] for e in ev)
        excl = sorted((p for p in scp if scp[p]['gold_is_abstention']), key=lambda p: int(p.split(':')[1]))
        abst_sets[(tag, split)] = excl
        bad_kept = [e for e in bad if not scp[e['pair_id']]['gold_is_abstention']]
        print(f"{tag:15s} {split}: n={n} U+FFFD={len(bad)} finished={len(fin)} unfinished_total={len(unf_all)} "
              f"U+FFFD&unfinished={len(inter)} | raw==scored texts: {same} | gold_is_abstention={len(excl)} "
              f"U+FFFD over non-abstention-gold items={len(bad_kept)} (n={n - len(excl)})")
        if split == 'test':
            for p in excl:
                e = scp[p]
                print(f"    excluded {p}: gold={e['gold']} | best={e['best_answer']!r} | adapter_out={e['outputs'][COND]!r} "
                      f"U+FFFD={FFFD in e['outputs'][COND]} outcome={e['outcome'][COND]}")
        # literal check: all U+FFFD rows in this arm also present in the other source?
        if split == 'test':
            rows = sorted(e['pair_id'] for e in bad)
            abst_sets[(tag, 'bad_rows')] = rows

print('excluded sets identical early vs release:', abst_sets[('early_r18', 'test')] == abst_sets[('release_claude', 'test')])
print('U+FFFD row sets identical early vs release (test):', abst_sets[('early_r18', 'bad_rows')] == abst_sets[('release_claude', 'bad_rows')])

# independent re-derivation of the excluded set: gold list consists only of 'no comment' style refusals
raw = load(SOURCES['early_r18'] / 'test.json')
mine = []
for e in raw['evaluations']:
    golds = [g.strip().lower().rstrip('.') for g in e['gold']]
    if golds and all(g in ('i have no comment', 'no comment') for g in golds):
        mine.append(e['pair_id'])
print('my rule (every gold answer == "I have no comment"/"no comment"):', sorted(mine, key=lambda p: int(p.split(':')[1])))
# gold_is_abstention also in the other two Qwen inits
for init in ['md', 'random']:
    sc = load(X_R18 / f'qwen3-8b/ce_alpha3_ep3/{init}/test_scored_items.json')
    s = sorted((e['pair_id'] for e in sc['evaluations'] if e['gold_is_abstention']), key=lambda p: int(p.split(':')[1]))
    print(f'early {init} test gold_is_abstention:', s)

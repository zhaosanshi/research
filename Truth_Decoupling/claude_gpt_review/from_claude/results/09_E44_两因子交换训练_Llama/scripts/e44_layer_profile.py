# -*- coding: utf-8 -*-
"""Read-only: for one judged export, right-count change (answered-count change) of each arm against its own base
prompt, one row per layer. Standard library only; uses e44_round1_tables.py.
usage: e44_layer_profile.py WORK FAMILY ARM [ARM ...] [--judge allenai|pro] [--title TEXT]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t

argv = sys.argv[1:]
judge, title = "allenai", None
for flag in ("--judge", "--title"):
    if flag in argv:
        i = argv.index(flag); val = argv[i + 1]; del argv[i:i + 2]
        if flag == "--judge": judge = val
        else: title = val
work, family, arms = argv[0], argv[1], argv[2:]
gen, specs, ids, state, rule, words = t.load(work, judge)
layers = sorted({s["layer"] for s in specs.values() if s.get("family") == family})
print("# %s — family %s, judge %s, %d questions: right-count change (answered-count change), each arm against its own base prompt\n" % (
    title or Path(work).name, family, judge, len(ids)))
bases = sorted({specs[c]["base_condition"] for c in specs if specs[c].get("family") == family})
print("No write: " + "; ".join("%s %s" % (b.replace("baseline_", ""), t.fmt_counts(state, ids, b)) for b in bases) + "  (right / wrong / hedge / abstained)\n")
print("| layer | " + " | ".join(arms) + " |")
print("|---|" + "---|" * len(arms))
tot = {a: [0, 0, 0] for a in arms}
for l in layers:
    cells = []
    for a in arms:
        c = t.cond(family, l, a)
        if c not in specs:
            cells.append("—"); continue
        v, n = t.per_question(state, words, ids, specs[c]["base_condition"], c)
        dR, dA = sum(v["dR"]), sum(v["dAns"])
        cells.append("%+d (%+d)" % (dR, dA))
        tot[a][0] += dR; tot[a][1] += dA; tot[a][2] += 1
    print("| %d | %s |" % (l, " | ".join(cells)))
print("\nMean over the layers present: " + "; ".join("%s %+.1f (%+.1f)" % (a, tot[a][0] / tot[a][2], tot[a][1] / tot[a][2]) for a in arms if tot[a][2]))

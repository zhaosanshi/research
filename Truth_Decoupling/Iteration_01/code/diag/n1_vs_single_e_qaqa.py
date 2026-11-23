# scratch reading (main session, 2026-11-24): (QA)^2 held-out 470, per model, N1 minus plain single E and N1 minus S0,
# by premise type, at natural answer counts and at a common count (the side with more answers drops its lowest
# batch-1 margins), reusing fill_only / selfjudge_stack_pick loaders unchanged. CPU only, prints numbers only.
import sys
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import fill_only as F

def keep(per, ids, name, n):
    cells = [per[q][name] for q in ids]
    avail = sorted((i for i, x in enumerate(cells) if x[1]), key=lambda i: (cells[i][2], i))
    dropped = set(avail[:max(0, len(avail) - n)])
    return [0 if i in dropped else x[0] for i, x in enumerate(cells)]

margins = F.margins()
tot = {}
for m in F.ORDER:
    ids, items, info = F.SKP.qaqa(m)["qaqa"]
    held = sorted(q for q in ids if not info[q][1])
    per = {q: F.SKP.systems(items[q], "baseline", margins[m]) for q in held}
    types = sorted(set(info[q][0] for q in held))
    groups = [("all", held)] + [(t, [q for q in held if info[q][0] == t]) for t in types]
    for a, b in (("N1", "E"), ("N1", "S0"), ("E", "S0")):
        line = []
        for gname, qs in groups:
            na = sum(per[q][a][1] for q in qs); nb = sum(per[q][b][1] for q in qs)
            nat = [per[q][a][0] - per[q][b][0] for q in qs]
            k = min(na, nb)
            va, vb = keep(per, qs, a, k), keep(per, qs, b, k)
            eq = [x - y for x, y in zip(va, vb)]
            line.append("%s n=%d: natural %s (%d vs %d) | equal %s (at %d)" % (gname, len(qs), F.ci_j(nat), na, nb, F.ci_j(eq), k))
            tot.setdefault((a, b, gname, "nat"), []).append(nat); tot.setdefault((a, b, gname, "eq"), []).append(eq)
        print("%-12s %s - %s || %s" % (m, a, b, " || ".join(line)))
for (a, b, gname, kind), lists in sorted(tot.items()):
    if gname == "all":
        d = [sum(x) for x in zip(*lists)]
        print("five models %s - %s %s %s: %s" % (a, b, gname, kind, F.ci_j(d)))

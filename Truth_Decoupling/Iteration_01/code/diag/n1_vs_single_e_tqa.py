# scratch reading (main session, 2026-11-24): TruthfulQA, per model, N1 minus plain single E and minus S0 at natural
# answer counts and at a common count (lowest batch-1 margins dropped), loaders reused unchanged. CPU only.
import sys
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import fill_only as F

def keep(per, ids, name, n):
    cells = [per[q][name] for q in ids]
    avail = sorted((i for i, x in enumerate(cells) if x[1]), key=lambda i: (cells[i][2], i))
    dropped = set(avail[:max(0, len(avail) - n)])
    return [0 if i in dropped else x[0] for i, x in enumerate(cells)]

margins = F.margins()
first = True
for m in F.ORDER:
    D = F.SKP.tqa(m)
    if first:
        print("keys:", list(D)); first = False
    for key in D:
        t = D[key]
        ids, items = t[0], t[1]
        try:
            per = {q: F.SKP.systems(items[q], "baseline", margins[m]) for q in ids}
        except Exception as e:
            print(m, key, "systems failed:", repr(e)[:200]); continue
        names = sorted(next(iter(per.values())).keys())
        out = []
        for a, b in (("N1", "E"), ("N1", "S0"), ("E", "S0")):
            if a not in names or b not in names:
                out.append("%s-%s: missing (%s)" % (a, b, ",".join(names))); continue
            na = sum(per[q][a][1] for q in ids); nb = sum(per[q][b][1] for q in ids)
            nat = [per[q][a][0] - per[q][b][0] for q in ids]
            k = min(na, nb)
            eq = [x - y for x, y in zip(keep(per, ids, a, k), keep(per, ids, b, k))]
            out.append("%s - %s: natural %s (%d vs %d) | equal %s (at %d)" % (a, b, F.ci_j(nat), na, nb, F.ci_j(eq), k))
        print("%-12s %-10s n=%d || %s" % (m, key, len(ids), " || ".join(out)))

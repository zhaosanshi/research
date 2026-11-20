# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call. EXPLORATION, not a result.
# TriviaQA / PopQA: the two writes used as a STABILITY check of the model's own answer (the mirror image of N1's rule).
# Background: on these fact questions N1's release step adds nothing at any E strength (diag/ns_dose_table.py): what C
# releases under E are mostly new, wrong answers ("None", "No one"), and the rule of N1 keeps exactly the new answers.
# The rule here keeps the opposite set. Per question, with the no-write answer A0:
#   A0 blank                          -> blank (nothing to check);
#   E at strength s answers           -> kept only if its answer is A0 (compared as release_origin.norm, the comparison
#                                        of N1's own rule); an answer that E changed is left blank;
#   E at s is blank                   -> asked again with E + C up the ladder to the cap; the first answer is kept only
#                                        if it is A0; otherwise blank. Cap None = never asked again (no C at all).
# So the system only ever returns the model's own no-write answer or a blank: it is a filter on the no-write answers,
# the filter being "the answer survives the E write, or comes back when C is added". Settings: the E strength and the
# cap, chosen on the task's tune split by net score (ties: lower strength, then lower cap). Verdicts by alias match.
# Printed per model and task: the tune grid; the tune choice and its test numbers against no write, single C 0.25,
# single E at its own best tune strength, N1 at its own best tune setting, and the same filter WITHOUT C at its own best
# tune strength (cap None: what E alone achieves as a filter); with paired intervals (questions resampled).
# A strength enters only if its five arms exist on both splits.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/ns_stable_table.py [--split2 test2] [TASK ...]
import sys

sys.path.insert(0, "Iteration_01/diag")
import ns_dose_table as T

S, fr = T.S, T.fr
ro, tc = fr.ro, fr.tc
CAPS = T.CAPS
net = T.net


def stable(sp, tx, s, cap):
    """States under the stability rule (blank = "X") for E strength s and cap."""
    out = []
    e = sp.cE(s)
    for q in sp.qs:
        if sp.blank[(q, "baseline_x")]:
            out.append("X")
            continue
        a0 = ro.norm(tx.ec[(q, "baseline_x")])
        if not sp.blank[(q, e)]:
            out.append(sp.st[(q, e)] if ro.norm(tx.ec[(q, e)]) == a0 else "X")
            continue
        k = fr.first_answer(sp, q, s, tc.lad(cap)) if cap is not None else None
        c = sp.cEC(s, k) if k is not None else None
        out.append(sp.st[(q, c)] if c is not None and ro.norm(tx.ec[(q, c)]) == a0 else "X")
    return out


def cnt(v):
    return "%+4d (ans %3d, R %3d W %3d)" % (net(v), sum(x != "X" for x in v), v.count("R"), v.count("W"))


def diff(a, b):
    return S.ps.ci([x - y for x, y in zip(S.vals(a), S.vals(b))])


def name_cap(c):
    return "none" if c is None else c.replace("-", ".")


def main():
    args = sys.argv[1:]
    second = "test"
    if args[:1] == ["--split2"]:
        second, args = args[1], args[2:]
    tasks = args or list(S.TASKS)
    M = S.models()
    for task in tasks:
        tot = {}
        print("\n# %s (alias match), settings chosen on tune, numbers on %s" % (task, second))
        for name in S.ORDER:
            m = M[name]
            try:
                tsp, ttx, th, tcc, tst, tbl = T.load(task, name, m, "tune")
                xsp, xtx, xh, xc, xst, xbl = T.load(task, name, m, second)
            except SystemExit as e:
                print("\n## %s: %s" % (name, e))
                continue
            have = [s for s in th if s in xh]
            only_tune = [s for s in th if s not in xh]
            if not th:
                print("\n## %s: no complete strength on tune" % name)
                continue
            base_t = [tst[(q, "baseline_x")] if not tbl[(q, "baseline_x")] else "X" for q in tsp.qs]
            print("\n## %s: tune strengths %s; on %s also: %s | no write on tune %s" % (
                name, " ".join(s.replace("-", ".") for s in th), second, " ".join(s.replace("-", ".") for s in have) or "none", cnt(base_t)))
            grid = {}
            for s in th:
                e = S.system(tsp, ttx, ("E", s))[0]
                cells = []
                for cap in CAPS:
                    grid[(s, cap)] = stable(tsp, ttx, s, cap)
                    cells.append("%s: %+d (%d)" % (name_cap(cap), net(grid[(s, cap)]), sum(x != "X" for x in grid[(s, cap)])))
                print("    tune E %-4s single E %+4d (ans %3d) | stable filter by cap, net (answers): %s" % (
                    s.replace("-", "."), net(e), sum(x != "X" for x in e), " | ".join(cells)))
            keys = [(s, cap) for s in th for cap in CAPS]
            best_all = max(keys, key=lambda k: (net(grid[k]), -keys.index(k)))
            print("    best on tune over every tune strength: E %s, cap %s, net %+d%s" % (
                best_all[0].replace("-", "."), name_cap(best_all[1]), net(grid[best_all]),
                "" if best_all[0] in have else "  <- its arms do not exist on %s yet" % second))
            if not have:
                continue
            keys = [(s, cap) for s in have for cap in CAPS]
            pick = max(keys, key=lambda k: (net(grid[k]), -keys.index(k)))
            keys0 = [(s, None) for s in have]
            pick0 = max(keys0, key=lambda k: (net(grid[k]), -keys0.index(k)))
            sE = max(have, key=lambda s: (net(S.system(tsp, ttx, ("E", s))[0]), -have.index(s)))
            n1c = [("N1", s, k1, k2) for s in have for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
            n1p = max(n1c, key=lambda c: (net(S.system(tsp, ttx, c)[0]), -n1c.index(c)))
            base = [xst[(q, "baseline_x")] if not xbl[(q, "baseline_x")] else "X" for q in xsp.qs]
            st_x, st0_x = stable(xsp, xtx, *pick), stable(xsp, xtx, *pick0)
            e_x = S.system(xsp, xtx, ("E", sE))[0]
            n1_x = S.system(xsp, xtx, n1p)[0]
            print("    chosen on tune among strengths present on %s: stable filter E %s cap %s | filter without C: E %s | single E: %s | N1: E %s caps %s / %s" % (
                second, pick[0].replace("-", "."), name_cap(pick[1]), pick0[0].replace("-", "."), sE.replace("-", "."),
                n1p[1].replace("-", "."), name_cap(n1p[2]), name_cap(n1p[3])))
            print("    %s %d questions: no write %s | single E %s | N1 %s" % (second, len(xsp.qs), cnt(base), cnt(e_x), cnt(n1_x)))
            print("        stable filter %s | filter without C %s" % (cnt(st_x), cnt(st0_x)))
            pairs = [("no write", base), ("single E at its own best", e_x), ("N1 at its own best", n1_x), ("the filter without C", st0_x)]
            if xc:
                cx = [xst[(q, xc)] if not xbl[(q, xc)] else "X" for q in xsp.qs]
                print("        single C 0.25 %s" % cnt(cx))
                pairs.insert(1, ("single C", cx))
            print("        stable filter minus: " + " | ".join("%s %s" % (lab, diff(st_x, v)) for lab, v in pairs))
            print("        filter without C minus: no write %s | single E at its own best %s" % (diff(st0_x, base), diff(st0_x, e_x)))
            for lab, v in pairs:
                tot.setdefault("stable filter minus " + lab, []).append(dict(zip(xsp.qs, [a - b for a, b in zip(S.vals(st_x), S.vals(v))])))
            tot.setdefault("filter without C minus no write", []).append(dict(zip(xsp.qs, [a - b for a, b in zip(S.vals(st0_x), S.vals(base))])))
            tot.setdefault("filter without C minus single E at its own best", []).append(dict(zip(xsp.qs, [a - b for a, b in zip(S.vals(st0_x), S.vals(e_x))])))
        if tot and all(len(v) == len(S.ORDER) for v in tot.values()):
            print("\n## %s, five models together (per question the differences are summed; questions resampled)" % task)
            for key, per in tot.items():
                common = sorted(set.intersection(*[set(p) for p in per]))
                print("    %-52s %s (%d questions)" % (key, S.ps.ci([sum(p[q] for p in per) for q in common]), len(common)))


if __name__ == "__main__":
    main()

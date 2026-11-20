# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call. EXPLORATION.
# TriviaQA / PopQA, the two writes on every question with C pushed toward caution: confirmation on the SECOND batch of
# test questions (test2, 500 per task, not used by any choice made so far).
# Why a second step: on the 192 tune questions settings three or four points apart cannot be told apart, and the first
# test batch has been looked at while the variants were explored. So every setting is chosen on tune + the first test
# batch together (692 questions, net score, ties to the earlier = weaker setting) and read on test2 only:
#   single E               x:E<l>.w.p<s>,  s in 0.5 1 1.5 2 3 4 (as far as the passes exist on tune and test)
#   single C toward caution x:C<l>.w.p<k>, k in 0.25 0.5 1 2
#   combination            x:E<l>.w.p<s>+C<l>.w.p<k>, s in 0.5 1 2
# Reported on test2: the combination against no write, against single E and against single C toward caution, each at
# its own chosen setting; paired intervals with questions resampled; models together when every listed model has it.
# Verdicts by alias match.
# usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/ns_caut_confirm.py --choose TASK MODEL    the three arms to generate on test2 (last line)
#   python3 -B Iteration_01/diag/ns_caut_confirm.py [MODEL ...]            the table (models whose test2 passes exist)
import sys

sys.path.insert(0, "Iteration_01/diag")
import ns_probe_table as P

S = P.S
ES = ("0.5", "1", "1.5", "2", "3", "4")
ES2 = ("0.5", "1", "2")
KS = ("0.25", "0.5", "1", "2")


def val(x):
    return (x == "R") - (x == "W")


def choose(task, name, m):
    """{family: (arm, selection net)} over tune + test, or None for a family without a complete member."""
    T, X = P.Passes(task, name, "tune"), P.Passes(task, name, "test")
    e, c = "x:E%d.w.p" % m["el"], "C%d.w.p" % m["cl"]
    fams = {"single E": [e + s for s in ES], "single C toward caution": ["x:" + c + k for k in KS],
            "combination": ["%s%s+%s%s" % (e, s, c, k) for s in ES2 for k in KS]}
    out, n = {}, {}
    for fam, arms in fams.items():
        rows = [(a, T.get(a), X.get(a)) for a in arms]
        rows = [(a, P.net(t[0]) + P.net(x[0])) for a, t, x in rows if t and x]
        n[fam] = (len(rows), len(arms))
        out[fam] = max(rows, key=lambda r: (r[1], -rows.index(r))) if len(rows) == len(arms) or (fam == "single E" and rows) else None
    base = P.net(T.get("x")[0]) + P.net(X.get("x")[0])
    return out, n, base


def main():
    M = S.models()
    if sys.argv[1:2] == ["--choose"]:
        task, name = sys.argv[2], sys.argv[3]
        out, n, base = choose(task, name, M[name])
        for fam, v in out.items():
            print("%s %s: %s: %s (%d of %d members on tune and test); no write %+d" % (
                name, task, fam, "%s, selection net %+d" % v if v else "not complete", n[fam][0], n[fam][1], base), file=sys.stderr)
        if not all(out.values()):
            raise SystemExit("not every family is complete")
        print(" ".join(v[0] for v in out.values()))
        return
    names = sys.argv[1:] or list(S.ORDER)
    for task in S.TASKS:
        print("\n# %s (alias match): settings chosen on tune + test (692 questions), numbers on test2 (500)" % task)
        tot = {}
        for name in names:
            try:
                out, n, base = choose(task, name, M[name])
            except SystemExit as e:
                print("\n## %s: %s" % (name, e))
                continue
            if not all(out.values()):
                print("\n## %s: families not complete on tune and test: %s" % (name, ", ".join("%s %d/%d" % (f, a, b) for f, (a, b) in n.items())))
                continue
            Z = P.Passes(task, name, "test2")
            z = {fam: Z.get(v[0]) for fam, v in out.items()}
            bz = Z.get("x")
            print("\n## %s: chosen on tune + test (no write there %+d): %s" % (
                name, base, " | ".join("%s: %s (%+d)" % (fam, v[0].split(":", 1)[1], v[1]) for fam, v in out.items())))
            if bz is None or not all(z.values()):
                print("    test2 passes missing: %s" % ", ".join(out[f][0] for f in out if not z[f]))
                continue
            print("    test2: no write %s | single E %s | single C toward caution %s | combination %s" % (
                P.cnt(bz[0]), P.cnt(z["single E"][0]), P.cnt(z["single C toward caution"][0]), P.cnt(z["combination"][0])))
            comb = z["combination"][0]
            pairs = [("no write", bz[0]), ("single E", z["single E"][0]), ("single C toward caution", z["single C toward caution"][0])]
            print("    combination minus: " + " | ".join("%s %s" % (lab, P.diff(comb, v)) for lab, v in pairs))
            best_single = max(("single E", "single C toward caution"), key=lambda f: out[f][1])
            print("    combination minus the better single direction on the selection set (%s): %s" % (best_single, P.diff(comb, z[best_single][0])))
            for lab, v in pairs + [("the better single direction", z[best_single][0])]:
                tot.setdefault("combination minus " + lab, []).append(dict(zip(Z.ids, [val(a) - val(b) for a, b in zip(comb, v)])))
        done = [k for k, per in tot.items() if len(per) == len(names)]
        if done:
            print("\n## %s, the %d models together (questions resampled)" % (task, len(names)))
            for key in done:
                per = tot[key]
                common = sorted(set.intersection(*[set(p) for p in per]))
                print("    %-48s %s (%d questions)" % (key, S.ps.ci([sum(p[q] for p in per) for q in common]), len(common)))


if __name__ == "__main__":
    main()

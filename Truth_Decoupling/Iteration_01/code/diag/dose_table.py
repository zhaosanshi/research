# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, test split, DeepSeek Pro, score = right - wrong: the main systems at E strength 1 next to E
# strength 2 (the frozen N1 uses 2, chosen on tune among 1 and 2). Question: how much of the gain is left at the lower
# dose, where writing E costs less on MMLU (novelty/ours/2026-11-19_capability_preservation_design.md, section 8.2)?
# Every answer and verdict already exists (the E+C grids carry both strengths). For each strength s the caps of N1 are
# chosen on tune among the same candidates and by the same rule as main_table_same_n.py, restricted to strength s; the
# strong form is the must-answer prompt + E at s, keeping its N most confident answers (N = the answers of N1 at s).
# Several models together: per question the differences are summed and questions are resampled.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/dose_table.py
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_same_n as MT

ps, ro = MT.ps, MT.ro
CAPS = MT.CAPS


def one(name, tot):
    tune, test = MT.Data(name, 0), MT.Data(name, 1)
    tsp, xsp = tune.sp, test.sp
    base = MT.plain(test, ("base",))
    print("\n## %s: test %d questions (tune %d)" % (name, len(xsp.qs), len(tsp.qs)))
    print(MT.line("no write", *base))
    kept = {}
    for s in ("1", "2"):
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        pick = max(cands, key=lambda x: (ps.score(ro.states(tsp, tune.tx, x), 1), -cands.index(x)))
        e = MT.plain(test, ("E", s))
        n1 = MT.plain(test, pick)
        print(MT.line("single E, strength %s" % s, *e))
        print(MT.line("N1 at E %s (chosen on tune: %s)" % (s, ro.name_of(pick)), *n1))
        N = sum(n1[1])
        st, ans = MT.must(test, xsp.dE(s))
        order = MT.ranked(test, ans, test.rows("conf_E%s" % s, test.m["d"][1], xsp.dE(s)))
        strong = None
        if len(order) >= N:
            strong = MT.keep(st, order, N)
            print(MT.line("must-answer + E %s, its %d most confident answers" % (s, N), *strong))
        else:
            print("    must-answer + E %s has only %d answers to choose from (fewer than %d): not comparable" % (s, len(order), N))
        print("    at E %s: single E minus no write %s | N1 minus no write %s | N1 minus single E %s%s" % (
            s, ps.ci([x - y for x, y in zip(MT.val(e[0]), MT.val(base[0]))]),
            ps.ci([x - y for x, y in zip(MT.val(n1[0]), MT.val(base[0]))]),
            ps.ci([x - y for x, y in zip(MT.val(n1[0]), MT.val(e[0]))]),
            "" if strong is None else " | N1 minus the strong form %s" % ps.ci([x - y for x, y in zip(MT.val(n1[0]), MT.val(strong[0]))])))
        kept[s] = (e, n1, strong)
        for key, a, b in (("single E %s minus no write" % s, e, base), ("N1 at E %s minus no write" % s, n1, base),
                          ("N1 at E %s minus single E %s" % (s, s), n1, e), ("N1 at E %s minus the strong form at E %s" % (s, s), n1, strong)):
            if b is None:
                continue
            d = tot.setdefault(key, {})
            for p, x, y in zip(xsp.qs, MT.val(a[0]), MT.val(b[0])):
                d[p] = d.get(p, 0) + x - y
    for key, a, b in (("single E 1 minus single E 2", kept["1"][0], kept["2"][0]), ("N1 at E 1 minus N1 at E 2", kept["1"][1], kept["2"][1])):
        v = [x - y for x, y in zip(MT.val(a[0]), MT.val(b[0]))]
        print("    %s: %s" % (key, ps.ci(v)))
        d = tot.setdefault(key, {})
        for p, x in zip(xsp.qs, v):
            d[p] = d.get(p, 0) + x


if __name__ == "__main__":
    tot = {}
    for name in ro.FIVE:
        one(name, tot)
    print("\n## the five models together (net; per question the differences of the models are summed; questions resampled)")
    for k, d in tot.items():
        print("  %-44s %s (%d questions)" % (k, ps.ci([d[p] for p in sorted(d)]), len(d)))

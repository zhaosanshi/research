# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, test split, DeepSeek Pro, score = right - wrong: the main systems at E strength 1 next to E
# strength 2 (the frozen N1 uses 2, chosen on tune among 1 and 2). Question: how much of the gain is left at the lower
# dose, where writing E costs less on MMLU (novelty/ours/2026-11-19_capability_preservation_design.md, section 8.3)?
# Every answer and verdict already exists (the E+C grids carry both strengths). For each strength s the caps of N1 are
# chosen on tune among the same candidates and by the same rule as main_table_same_n.py, restricted to strength s; the
# strong form is the must-answer prompt + E at s, keeping its N most confident answers (N = the answers of N1 at s).
# Added the same day (the status board found no record of these at strength 1): prompt B at s (blanks under E re-asked
# with the must-answer prompt), and the prompt-only baseline "must-answer prompt + the honest sentence, nothing
# written" (condition baseline_conf_E2), against N1 and against must-answer + E at s with every question answered.
# Several models together: per question the differences are summed and questions are resampled; printed for the five
# models and for the three main-text models.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/dose_table.py
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_same_n as MT

ps, ro = MT.ps, MT.ro
CAPS = MT.CAPS
MAIN = ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B")


def sub(a, b):
    return [x - y for x, y in zip(MT.val(a[0]), MT.val(b[0]))]


def one(name, tots):
    tune, test = MT.Data(name, 0), MT.Data(name, 1)
    tsp, xsp = tune.sp, test.sp
    base = MT.plain(test, ("base",))
    print("\n## %s: test %d questions (tune %d)" % (name, len(xsp.qs), len(tsp.qs)))
    print(MT.line("no write", *base))
    try:
        da = MT.handed_in(test, ("DA",))
        print(MT.line("must-answer prompt + the honest sentence, nothing written", *da))
    except (Exception, SystemExit) as e:      # noqa: BLE001  (this run has no such condition)
        da = None
        print("    must-answer prompt + the honest sentence: not available (%s: %s)" % (type(e).__name__, str(e)[:80]))

    def add(key, v):
        for t in tots:
            d = t.setdefault(key, {})
            for p, x in zip(xsp.qs, v):
                d[p] = d.get(p, 0) + x

    kept = {}
    for s in ("1", "2"):
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        pick = max(cands, key=lambda x: (ps.score(ro.states(tsp, tune.tx, x), 1), -cands.index(x)))
        e = MT.plain(test, ("E", s))
        n1 = MT.plain(test, pick)
        b = MT.prompt_b(test, s)
        de = MT.must(test, xsp.dE(s))
        print(MT.line("single E, strength %s" % s, *e))
        print(MT.line("N1 at E %s (chosen on tune: %s)" % (s, ro.name_of(pick)), *n1))
        print(MT.line("prompt B at E %s (blanks re-asked with the must-answer prompt)" % s, *b))
        print(MT.line("must-answer + E %s, every question answered" % s, *de))
        N = sum(n1[1])
        order = MT.ranked(test, de[1], test.rows("conf_E%s" % s, test.m["d"][1], xsp.dE(s)))
        strong = None
        if len(order) >= N:
            strong = MT.keep(de[0], order, N)
            print(MT.line("must-answer + E %s, its %d most confident answers" % (s, N), *strong))
        else:
            print("    must-answer + E %s has only %d answers to choose from (fewer than %d): not comparable" % (s, len(order), N))
        print("    at E %s: single E minus no write %s | N1 minus no write %s | N1 minus single E %s%s" % (
            s, ps.ci(sub(e, base)), ps.ci(sub(n1, base)), ps.ci(sub(n1, e)),
            "" if strong is None else " | N1 minus the strong form %s" % ps.ci(sub(n1, strong))))
        print("    at E %s: N1 minus prompt B %s%s" % (s, ps.ci(sub(n1, b)), "" if da is None else (
            " | N1 minus must-answer + sentence %s | must-answer + E minus must-answer + sentence: %s" % (
                ps.ci(sub(n1, da)), MT.diff(de[0], da[0])))))
        kept[s] = (e, n1)
        add("single E %s minus no write" % s, sub(e, base))
        add("N1 at E %s minus no write" % s, sub(n1, base))
        add("N1 at E %s minus single E %s" % (s, s), sub(n1, e))
        if strong is not None:
            add("N1 at E %s minus the strong form at E %s" % (s, s), sub(n1, strong))
        add("N1 at E %s minus prompt B at E %s" % (s, s), sub(n1, b))
        if da is not None:
            add("N1 at E %s minus must-answer + sentence" % s, sub(n1, da))
            add("must-answer + E %s minus must-answer + sentence" % s, sub(de, da))
    for key, a, b2 in (("single E 1 minus single E 2", kept["1"][0], kept["2"][0]), ("N1 at E 1 minus N1 at E 2", kept["1"][1], kept["2"][1])):
        print("    %s: %s" % (key, ps.ci(sub(a, b2))))
        add(key, sub(a, b2))


if __name__ == "__main__":
    tot, tot3 = {}, {}
    for name in ro.FIVE:
        one(name, (tot, tot3) if name in MAIN else (tot,))
    for title, t in (("the five models together", tot), ("the three main-text models together (Llama-3.1-8B, Qwen3.5-9B, Qwen3.8-27B)", tot3)):
        print("\n## %s (net; per question the differences of the models are summed; questions resampled)" % title)
        for k, d in t.items():
            print("  %-50s %s (%d questions)" % (k, ps.ci([d[p] for p in sorted(d)]), len(d)))

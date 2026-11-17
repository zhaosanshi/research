# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# 赵磊 2026-11-17: "你现在告诉我现在最好的方案是什么，E+C强于单E和单C的方案。"
# The two kinds of blanks behave differently (novelty/ours/2026-11-17_silence_and_release.md, 4.3): the questions a model does not
# answer with no write are released mostly right; the questions E itself silences are released about half right.
# Scheme "two caps": E at s; a blank of the first kind gets C raised up to K2, a blank of the second kind up to K1
# (K = none: not released). The present method is K1 = K2; "own blanks only" (M1) is K1 = none. s, K1, K2 are chosen on the
# tune split by right - wrong (ties to the earlier, weaker setting) and reported on the test split (DeepSeek Pro).
# Also: the same chosen settings scored by the second judge (AllenAI) on the full test split, as an auxiliary check.
# Prediction written before the run (Fable 5.2, 16:2x): on every model, two caps is not more than 3 points below the better of
# the present method and M1 on the test split; its difference to single E has an interval above 0 on at least 4 of the 5
# models with blanks (Llama-3.1-8B, the two Qwen, the two Gemma-3).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/two_caps.py
import collections
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf

LAD = ps.LAD
CAPS = [None] + LAD


def lad(cap):
    return [] if cap is None else LAD[:LAD.index(cap) + 1]


def states(sp, cfg):
    if cfg[0] != "T2":
        return cf.states(sp, cfg)
    _, s, k1, k2 = cfg
    e = sp.cE(s)
    out = []
    for p in sp.qs:
        state = sp.st[(p, e)]
        if sp.blank[(p, e)]:
            for k in lad(k2 if sp.blank[(p, "baseline_x")] else k1):
                if not sp.blank[(p, sp.cEC(s, k))]:
                    state = sp.st[(p, sp.cEC(s, k))]
                    break
        out.append(state)
    return out


def name_of(cfg):
    if cfg[0] != "T2":
        return cf.describe(cfg)
    return "E %s, own blanks up to %s, E's blanks up to %s" % (cfg[1], ps.NAME[cfg[3]], ps.NAME[cfg[2]])


FAM = {
    "single E": [("E", "1"), ("E", "2")],
    "method": [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in LAD],
    "own only": [("E", s) for s in "12"] + [("BO", s, k) for s in "12" for k in LAD],
    "two caps": [("E", s) for s in "12"] + [("T2", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)],
}
AUX = {  # the full test split judged by AllenAI (auxiliary); None = the Pro test dir itself holds AllenAI verdicts for all questions
    "Llama-3.1-8B": None, "Qwen3.5-9B": None, "Qwen3.8-27B": None,
    "Gemma-3-12B": "Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b/eval_extra/judge_test_t25_test",
    "Gemma-3-27B": "Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b/eval_extra/judge_test_t31_test",
    "Gemma-4-12B": "Iteration_01/results/ff/gemma4/12b/e44/gemma4-12b/eval_extra/judge_test_t21_test",
    "Gemma-4-12B stronger caution": "Iteration_01/results/ff_xu1/gemma4/12b/e44/gemma4-12b/eval_extra/judge_test_t25_test",
    "Gemma-4-31B": "Iteration_01/results/ff/gemma4/31b/e44/gemma4-31b-dm/eval_extra/judge_test_t28_test",
}


def main():
    for name, m in cf.MODELS.items():
        tune, test = ps.Split(m, 0), ps.Split(m, 1)
        fam = dict(FAM)
        fam["single C"] = [("C", k) for k in LAD if k in tune.cks and k in test.cks]
        pick = {f: max(c, key=lambda x: (ps.score(states(tune, x), 1), -c.index(x))) for f, c in fam.items()}
        print("\n## %s (tune %d, test %d; score = right - wrong; Pro)" % (name, len(tune.qs), len(test.qs)))
        row = {}
        for f in ("single E", "single C", "method", "own only", "two caps"):
            st = states(test, pick[f])
            row[f] = st
            c = collections.Counter(st)
            print("  %-9s chosen on tune: %-46s | tune %5.1f | test: %s | score %4d" % (f, name_of(pick[f]), ps.score(states(tune, pick[f]), 1), cf.cnt(c), ps.score(st, 1)))
        for a in ("method", "own only", "two caps"):
            print("  %-9s minus single E: %-18s | minus single C: %-22s%s" % (
                a, ps.ci([x - y for x, y in zip(cf.val(row[a]), cf.val(row["single E"]))]),
                ps.ci([x - y for x, y in zip(cf.val(row[a]), cf.val(row["single C"]))]),
                "" if a != "two caps" else " | minus method: %s | minus own only: %s" % (
                    ps.ci([x - y for x, y in zip(cf.val(row[a]), cf.val(row["method"]))]),
                    ps.ci([x - y for x, y in zip(cf.val(row[a]), cf.val(row["own only"]))]))))
        # auxiliary: the same settings, the full test split, AllenAI verdicts
        aux = AUX.get(name)
        work = aux or m["ec"][1]
        if not os.path.exists(work + "/judge2/single_allenai.jsonl"):
            print("  auxiliary (AllenAI): no verdict file at %s" % work)
            continue
        m2 = dict(m, ec=(m["ec"][0], work), d=None, judge=(m["judge"][0], "allenai"))
        try:
            t2 = ps.Split(m2, 1)
        except SystemExit as err:
            print("  auxiliary (AllenAI): %s" % err)
            continue
        r2 = {f: states(t2, pick[f]) for f in ("single E", "method", "own only", "two caps")}
        print("  auxiliary, AllenAI on %d test questions, same settings: single E %d | method %d | own only %d | two caps %d" % (
            len(t2.qs), ps.score(r2["single E"], 1), ps.score(r2["method"], 1), ps.score(r2["own only"], 1), ps.score(r2["two caps"], 1)))
        for a in ("method", "own only", "two caps"):
            print("      %-9s minus single E (AllenAI): %s" % (a, ps.ci([x - y for x, y in zip(cf.val(r2[a]), cf.val(r2["single E"]))])))


if __name__ == "__main__":
    main()

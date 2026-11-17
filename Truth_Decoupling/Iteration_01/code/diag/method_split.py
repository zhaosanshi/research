# Iteration_01 analysis (Claude Opus 5.5, 2026-11-17), read-only: no model, no GPU, no judge call.
# Task E of the 11-17 offline batch: "method minus single E" on the test split, split over two kinds of blanks.
# Settings chosen on the tune split exactly as c_first.py (score = right - wrong; method family = E at s alone, or E at s
# with C raised on E's blanks up to K; single E family = E at 1 or 2; ties to the earlier, weaker setting).
# Per test question: d = score(method) - score(single E), score = +1 right / 0 hedge or blank / -1 wrong (Pro states).
#   (i)  answered with no write (rule detector: baseline_x not blank) and blank with E at the method's s
#   (ii) blank with no write (rule detector)
# Each part: its net score and a paired bootstrap interval (policy_select2.ci: 4000 resamples of ALL test questions,
# seed 20261116; questions outside the part count 0). Check: (i) + (ii) + rest = the total, where rest is the difference on
# the other questions (0 when single E and the method use the same s, because the method only touches E's blanks).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/method_split.py [MODEL ...]
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first

ORDER = ["Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B", "Gemma-3-12B", "Gemma-3-27B", "Gemma-4-12B",
         "Gemma-4-12B stronger caution", "Gemma-4-31B"]
LAM = 1


def main(names):
    print("| model | method (chosen on tune) | single E (chosen on tune) | test questions | (i) questions / net [95%] | (ii) questions / net [95%] | "
          "rest | total method - single E [95%] | (i) + (ii) + rest = total | method / single E test score |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        m = c_first.MODELS[name]
        tune, test = ps.Split(m, 0), ps.Split(m, 1)
        if m["judge"] != ("pro", "pro"):
            raise SystemExit("%s: not Pro on both splits" % name)
        fam = {"single E": [("E", "1"), ("E", "2")],
               "method": [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in ps.LAD]}
        pick = {f: max(c, key=lambda x: (ps.score(c_first.states(tune, x), LAM), -c.index(x))) for f, c in fam.items()}
        s = pick["method"][1]
        e = test.cE(s)
        vm = c_first.val(test.states(pick["method"]))
        ve = c_first.val(test.states(pick["single E"]))
        d = [a - b for a, b in zip(vm, ve)]
        in_i = [(not test.blank[(p, "baseline_x")]) and test.blank[(p, e)] for p in test.qs]
        in_ii = [test.blank[(p, "baseline_x")] for p in test.qs]
        di = [x if f else 0 for x, f in zip(d, in_i)]
        dii = [x if f else 0 for x, f in zip(d, in_ii)]
        rest = sum(x for x, a, b in zip(d, in_i, in_ii) if not a and not b)
        if pick["single E"][1] == s and rest != 0:
            raise SystemExit("%s: the method changes a question outside E's blanks" % name)
        ok = sum(di) + sum(dii) + rest == sum(d)
        if not ok:
            raise SystemExit("%s: parts do not add up" % name)
        print("| %s | %s | %s | %d | %d / %s | %d / %s | %+d | %s | %+d %+d %+d = %+d (%s) | %d / %d |" % (
            name, ps.describe(pick["method"]), ps.describe(pick["single E"]), len(test.qs), sum(in_i), ps.ci(di), sum(in_ii), ps.ci(dii),
            rest, ps.ci(d), sum(di), sum(dii), rest, sum(d), "ok" if ok else "MISMATCH", sum(vm), sum(ve)))


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)

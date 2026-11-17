# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# Numbers for section 4 of novelty/ours/2026-11-16_gemma3_27b_prediction.md: blanks with E, the best E + C cell against E alone (right answers,
# the metric the prediction was written in), and the must-answer E line; AllenAI on the 523 test questions, DeepSeek Pro on the 98.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/g27_prediction_check.py
import sys
sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
R = "Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b"
LAD = ps.LAD
def cE(s): return "E_L26_x-w-p%s" % s
def cEC(s, k): return "X_x--E26-w-p%s-and-C31-w-m%s" % (s, k)
for label, work, dwork, judge in (("AllenAI 523", R + "/eval_extra/judge_test_t31_test", R + "/judge_test_testD_E_L26", "allenai"),
                                  ("Pro 98", R + "/eval_extra/judge_test_t31_testdr100", R + "/judge_test_testD_E_L26dr100", "pro")):
    st, bl = ps.load(work, judge)
    need = ["baseline_x"] + [cE(s) for s in "12"] + [cEC(s, k) for s in "12" for k in LAD]
    qs0 = sorted(p for p, c in st if c == "baseline_x")
    qs = [p for p in qs0 if all(st.get((p, c), "?") != "?" for c in need)]
    print("\n## %s: %d questions (%d dropped for a missing verdict)" % (label, len(qs), len(qs0) - len(qs)))
    print("  rule blanks: no write %d | E 1 %d | E 2 %d" % tuple(sum(bl[(p, c)] for p in qs) for c in ("baseline_x", cE("1"), cE("2"))))
    for s in "12":
        e = [st[(p, cE(s))] for p in qs]
        line = "  E %s alone: right %d wrong %d, score %d |" % (s, e.count("R"), e.count("W"), e.count("R") - e.count("W"))
        for k in LAD:
            j = [st[(p, cEC(s, k))] for p in qs]
            d = [(a == "R") - (b == "R") for a, b in zip(j, e)]
            line += " + C %s on all: right %+d %s |" % (ps.NAME[k], sum(d), ps.ci(d).split(" ", 1)[1])
        print(line)
    dst, _ = ps.load(dwork, judge)
    dq = [p for p in qs if all(dst.get((p, c), "?") != "?" for c in ("baseline_conf", "baseline_conf_E2", "E_L26_conf-w-p1", "E_L26_conf-w-p2"))]
    conds = sorted({c for _, c in dst})
    print("  must-answer, %d questions; conditions: %s" % (len(dq), " ".join(conds)))
    base = [dst[(p, "baseline_conf")] for p in dq]
    for c in conds:
        if c == "baseline_conf":
            continue
        if any(dst.get((p, c), "?") == "?" for p in dq):
            print("    %s: has missing verdicts, skipped" % c); continue
        a = [dst[(p, c)] for p in dq]
        d = [(x == "R") - (y == "R") for x, y in zip(a, base)]
        v = [((x == "R") - (x == "W")) - ((y == "R") - (y == "W")) for x, y in zip(a, base)]
        print("    %-22s minus must-answer no write: right %s | score %s" % (c, ps.ci(d), ps.ci(v)))

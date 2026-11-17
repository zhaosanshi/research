# Iteration_01 analysis (Claude Opus 5.5, 2026-11-17), read-only: no model, no GPU, no judge call.
# Task C of the 11-17 offline batch: the law "the gain of E+C ~ 0.45 x the blanks after E", first reproduced as it was
# defined, then recomputed with score = right - wrong.
# Part 1, the original (工作上下文 11-16 00:37, made with subset_compare.py): 4 models x E at 1, 2 = 8 points on the test
#   split (523); judges Pro for Llama-3.1-8B / Qwen3.5-9B / Qwen3.8-27B, AllenAI for Gemma-4-12B (judge_test_t21_test).
#   x = questions NOT answered with E alone = n - (R + W) under that judge (X and H both count as not answered);
#   y = max over k in {0.25, 0.5, 1, 2} of [right answers of the joint arm E s + C k on EVERY question] - right answers of
#   E s alone. Questions as subset_compare.py: those holding the base cell, minus any question with a '?' in a used cell
#   under ANY judge file in the dir. Slope through the origin = sum(xy) / sum(x^2); Pearson r.
# Part 2, the recomputation: the 8 rows of policy_select2 / c_first (Pro on both splits), E at 1, 2, test (and tune).
#   method(s, K) = E at s; the questions E alone leaves blank (rule detector) get C raised 0.25 -> 0.5 -> 1 -> 2 until
#   answered, up to K (policy_select2.Split.states(("EC", s, K))); score = right - wrong.
#   y_max = max over K in {0.25, 0.5, 1, 2} of score(method s K) - score(E s)   (K chosen with hindsight on the split itself)
#   y_sel = score(method s K*) - score(E s), K* in {none, 0.25, 0.5, 1, 2} chosen on the TUNE split by the same score
#           (ties to the weaker K; K* = none gives 0)
#   x_a = questions blank (rule detector) with E s; x_b = of these, blank with no write too; x_c = x_a - x_b
#   (answered with no write and silenced by E, by the rule detector).
#   y_b / y_c = the part of y coming from the x_b / x_c questions (the method differs from E alone only on x_a questions,
#   so y = y_b + y_c exactly; checked).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/law_lam1.py
import json
import math
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first

ORDER = ["Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B", "Gemma-3-12B", "Gemma-3-27B", "Gemma-4-12B",
         "Gemma-4-12B stronger caution", "Gemma-4-31B"]
LAM = 1
G = "Iteration_01/results/ff/gemma4/12b/e44/gemma4-12b"
ORIG = [  # name, work dir of the test grid, judge, E layer, C layer, order  (as policy_select.py MODELS)
    ("Llama-3.1-8B", "results_e44_factorial_das/llama3.1-8b_side_E2/eval_extra/judge_test_tll19_test", "pro", 11, 19, "EC"),
    ("Qwen3.5-9B", "results_newmodels_ff_20261112/e44/qwen3.5-9b_E2/eval_extra/judge_test_t9b16_test", "pro", 15, 16, "EC"),
    ("Qwen3.8-27B", "results_newmodels_ff_20261112/e44/qwen3.8-27b-dm/eval_extra/judge_test_t2733_test", "pro", 28, 33, "EC"),
    ("Gemma-4-12B", G + "/eval_extra/judge_test_t21_test", "allenai", 24, 21, "CE"),
]
ORIG_KNOWN = {("Llama-3.1-8B", "1"): (134, 56), ("Llama-3.1-8B", "2"): (228, 114), ("Qwen3.5-9B", "1"): (84, 27), ("Qwen3.5-9B", "2"): (120, 39),
              ("Qwen3.8-27B", "1"): (185, 82), ("Qwen3.8-27B", "2"): (160, 76), ("Gemma-4-12B", "1"): (37, 11), ("Gemma-4-12B", "2"): (26, 5)}


def sc_state(r, judge):  # copied from subset_compare.py state()
    if r["status"] != "scored":
        return "X" if (r.get("basis") == "empty_output_no_judge_vote" and r.get("empty_output") is True) else "?"
    if judge == "allenai":
        return "X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]
    return {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]


def sc_load(work):  # copied from subset_compare.py load()
    lab = {}
    for j in ("allenai", "pro"):
        try:
            lab[j] = {}
            for l in open(work + "/judge2/single_%s.jsonl" % j, encoding="utf-8"):
                r = json.loads(l)
                lab[j][(r["pair_id"], r["condition"])] = sc_state(r, j)
        except FileNotFoundError:
            del lab[j]
    return lab


def fit(pts):
    """pts: [(x, y)]: slope through the origin and Pearson r (None when x or y does not vary)."""
    sxx = sum(x * x for x, _ in pts)
    slope = sum(x * y for x, y in pts) / sxx if sxx else None
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    vx = sum((x - mx) ** 2 for x, _ in pts)
    vy = sum((y - my) ** 2 for _, y in pts)
    r = sum((x - mx) * (y - my) for x, y in pts) / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None
    return slope, r


def f3(v):
    return "—" if v is None else "%.3f" % v


def part1():
    print("## Part 1: the original law reproduced (test 523; Pro for Llama / Qwen, AllenAI for Gemma-4-12B; right answers)")
    print("| model | E | questions | x = not answered with E alone (X + H) | y = max over k of right(E + C k on all) - right(E) | best k | known (x -> y) | match |")
    print("|---|---|---|---|---|---|---|---|")
    pts, ok = [], True
    for name, work, judge, e, c, order in ORIG:
        lab = sc_load(work)
        for s in "12":
            base = "E_L%d_x-w-p%s" % (e, s)
            arms = [("X_x--E%d-w-p%s-and-C%d-w-m%s" % (e, s, c, k)) if order == "EC" else ("X_x--C%d-w-m%s-and-E%d-w-p%s" % (c, k, e, s)) for k in ps.LAD]
            first = next(iter(lab.values()))
            have = {cc for _, cc in first}
            miss = [a for a in [base] + arms if a not in have]
            if miss:
                raise SystemExit("%s lacks %s" % (work, miss))
            used = [base] + arms
            qs = sorted({p for j in lab for p, cc in lab[j] if cc == base})
            bad = {p for j in lab for p in qs for cc in used if lab[j].get((p, cc), "?") == "?"}
            qs = [p for p in qs if p not in bad]
            L = lab[judge]
            x = sum(L[(p, base)] not in "RW" for p in qs)
            rE = sum(L[(p, base)] == "R" for p in qs)
            gains = [sum(L[(p, a)] == "R" for p in qs) - rE for a in arms]
            y = max(gains)
            kb = ps.NAME[ps.LAD[gains.index(y)]]
            kx, ky = ORIG_KNOWN[(name, s)]
            ok &= (x, y) == (kx, ky)
            pts.append((x, y))
            print("| %s | %s | %d | %d | %+d | %s | %d -> %+d | %s |" % (name, s, len(qs), x, y, kb, kx, ky, "yes" if (x, y) == (kx, ky) else "NO"))
    slope, r = fit(pts)
    print("- 8 points: slope through the origin %s, Pearson r %s (known: 0.45, 0.987)" % (f3(slope), f3(r)))
    return ok


def score_on(states, qs_mask):
    return sum((1 if s == "R" else -LAM if s == "W" else 0) for s, m in zip(states, qs_mask) if m)


def part2():
    rows = {"test": [], "tune": []}
    for name in ORDER:
        m = c_first.MODELS[name]
        tune, test = ps.Split(m, 0), ps.Split(m, 1)
        for s in "12":
            # K* chosen on tune
            cand = [None] + ps.LAD
            tsc = [ps.score(tune.states(("E", s) if k is None else ("EC", s, k)), LAM) for k in cand]
            kstar = cand[max(range(len(cand)), key=lambda j: (tsc[j], -j))]
            for label, sp in (("test", test), ("tune", tune)):
                e = sp.cE(s)
                xa = [sp.blank[(p, e)] for p in sp.qs]
                xb = [sp.blank[(p, e)] and sp.blank[(p, "baseline_x")] for p in sp.qs]
                xc = [a and not b for a, b in zip(xa, xb)]
                sE = sp.states(("E", s))
                base_e = ps.score(sE, LAM)
                per_k = {}
                for k in ps.LAD:
                    st = sp.states(("EC", s, k))
                    tot = ps.score(st, LAM) - base_e
                    yb = score_on(st, xb) - score_on(sE, xb)
                    yc = score_on(st, xc) - score_on(sE, xc)
                    outside = score_on(st, [not a for a in xa]) - score_on(sE, [not a for a in xa])
                    if outside != 0 or yb + yc != tot:
                        raise SystemExit("%s %s E %s C %s: the method differs from E alone outside its blanks (%d) or parts do not add up" % (name, label, s, k, outside))
                    per_k[k] = (tot, yb, yc)
                kmax = max(ps.LAD, key=lambda k: (per_k[k][0], -ps.LAD.index(k)))
                ysel = per_k[kstar] if kstar is not None else (0, 0, 0)
                rows[label].append(dict(model=name, s=s, n=len(sp.qs), xa=sum(xa), xb=sum(xb), xc=sum(xc), kmax=kmax, ymax=per_k[kmax],
                                        kstar=kstar, ysel=ysel))
    for label in ("test", "tune"):
        R = rows[label]
        print("\n## Part 2, %s split (Pro; score = right - wrong; blanks by the rule detector)" % label)
        print("| model | E | questions | x_a blank with E | x_b of which blank with no write | x_c of which answered with no write | "
              "K max | y_max | y_max from x_b | y_max from x_c | K chosen on tune | y_sel | y_sel from x_b | y_sel from x_c |")
        print("|" + "---|" * 14)
        for r in R:
            print("| %s | %s | %d | %d | %d | %d | %s | %+d | %+d | %+d | %s | %+d | %+d | %+d |" % (
                r["model"], r["s"], r["n"], r["xa"], r["xb"], r["xc"], ps.NAME[r["kmax"]], r["ymax"][0], r["ymax"][1], r["ymax"][2],
                ps.NAME[r["kstar"]], r["ysel"][0], r["ysel"][1], r["ysel"][2]))
        ys = {"y_max": lambda r: r["ymax"][0], "y_max from x_b": lambda r: r["ymax"][1], "y_max from x_c": lambda r: r["ymax"][2],
              "y_sel": lambda r: r["ysel"][0], "y_sel from x_b": lambda r: r["ysel"][1], "y_sel from x_c": lambda r: r["ysel"][2]}
        xs = {"x_a": "xa", "x_b": "xb", "x_c": "xc"}
        for scope, sub in (("all 16 points", R), ("the 6 points on 523 questions", [r for r in R if r["n"] > 400]),
                           ("the 10 points on 98 questions", [r for r in R if r["n"] < 100])) if label == "test" else (("all 16 points", R),):
            print("\n%s, %s: slope through the origin / Pearson r" % (label, scope))
            print("| y \\ x | " + " | ".join(xs) + " |")
            print("|---|" + "---|" * len(xs))
            for yn, yf in ys.items():
                cells = []
                for xn, xk in xs.items():
                    sl, rr = fit([(r[xk], yf(r)) for r in sub])
                    cells.append("%s / %s" % (f3(sl), f3(rr)))
                print("| %s | %s |" % (yn, " | ".join(cells)))


def main():
    ok = part1()
    if not ok:
        print("\nPart 1 did NOT reproduce the known numbers: Part 2 not run.")
        sys.exit(1)
    part2()


if __name__ == "__main__":
    main()

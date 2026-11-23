# Iteration_01 (Claude Opus 5.5, executor F, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_m.md section 1):
# the C row of the free-form 2x2, per model, per layer, per arm, DeepSeek Pro only. Read-only, no model, no judge call.
# Arms (frozen protocol of P2 / from_claude/45 R5): plain prompt x, C at 0.25, toward caution x.w.p0.25 and toward
# answering x.w.m0.25, each against the no-write answers baseline_x of the same work dir (same judge, same questions).
# Every number comes from the unchanged loader and helpers of scripts/review_checks/e44_round1_tables.py (load,
# per_question, boot, keep_interval; tolerance 0.03 as a rate; paired bootstrap 10,000 resamples, seed 20261111); the
# criterion items are the ones of its `cells` mode for family C (criterion v4, from_claude/30 section 4.2), item by item:
#   effect     A: answers move that way by more than 3% of the questions and abstentions move the other way
#              B: and both intervals lie on that side
#   random     A: the move is larger than that of each random arm x.rand.p0.25 / x.rand.m0.25 (point)
#              B: each paired difference interval lies on that side; "no rand" when the random arms were not run
#   retention  A: questions answered both times > 0 and at least half the base answers; answered -> hedge <= 3%;
#              right-rate change on the questions answered both times within +-3 pp (point)
#              B: and its whole interval within +-3 pp
# Answered = right or wrong (Pro). Net = right - wrong. Abstention-only-gold questions leave the tables (as the loader does).
# Printout: one block per split; one row per (model, layer, arm); nothing pooled over models.
# usage (scoring container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/c_row_layers.py [tune|test|all]
import os
import sys

sys.path.insert(0, "Iteration_01/code/scripts/review_checks")
import e44_round1_tables as T  # noqa: E402

LL = "results_e44_factorial_das/llama3.1-8b"
Q9 = "results_newmodels_ff_20261112/e44/qwen3.5-9b"
Q27 = "results_newmodels_ff_20261112/e44/qwen3.8-27b-dm"
G12 = "Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b"
G27 = "Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b"
CROW = "judge_tune_crow_C"
# model: (run dir, tune layers, tune layers judged in their own proS work dir, test layers, N1's C layer)
MODELS = (
    ("Llama-3.1-8B", LL, list(range(32)), (19, 31), (19, 31), 19),
    ("Qwen3.5-9B", Q9, list(range(32)), (12, 16, 17), (12, 16, 17), 16),
    ("Qwen3.8-27B", Q27, list(range(19, 38, 2)), (33, 35, 37), (33, 35, 37), 33),
    ("Gemma-3-12B", G12, list(range(15, 28, 2)), (), (25,), 25),
    ("Gemma-3-27B", G27, list(range(19, 38, 2)), (), (31,), 31),
)
ARMS = (("x.w.p0.25", -1, "caution"), ("x.w.m0.25", +1, "answering"))
RANDS = ("x.rand.p0.25", "x.rand.m0.25")
VAL = {"R": 1, "W": -1, "H": 0, "X": 0}
_CACHE = {}


def work_of(run, split, layer, own):
    if split == "test":
        return "%s/judge_test_testS_C_L%d" % (run, layer)
    return "%s/judge_tune_proS_C_L%d" % (run, layer) if layer in own else "%s/%s" % (run, CROW)


def loaded(work):
    if work not in _CACHE:
        _CACHE[work] = T.load(work, "pro")
    return _CACHE[work]


def iv(d):
    lo, hi = T.boot(d)
    return "%+d [%+d, %+d]" % (sum(d), lo, hi), lo, hi


def pp(t):
    return "undef" if t is None else "%+.1f [%+.1f, %+.1f]" % (100 * t[0], 100 * t[1], 100 * t[2])


def tf(x):
    return "no rand" if x is None else ("yes" if x else "no")


def rows_for(name, run, layer, split, own):
    w = work_of(run, split, layer, own)
    if not os.path.exists(os.path.join(w, "judge2", "single_pro.jsonl")):
        return ["| %s | %d | - | - | no Pro verdicts yet in %s |" % (name, layer, w)]
    gen, specs, ids, state, rule, words = loaded(w)
    N = len(ids)
    base = "baseline_x"
    rb, wb, hb, xb = T.counts(state, ids, base)
    base_ans = rb + wb
    K = {}
    for a in [x[0] for x in ARMS] + list(RANDS):
        if T.have(state, ids, T.cond("C", layer, a)):
            v, n = T.per_question(state, words, ids, base, T.cond("C", layer, a))
            K[a] = (v, n)
    out = []
    missing = [r for r in RANDS if r not in K]
    for arm, sign, label in ARMS:
        if arm not in K:
            out.append("| %s | %d | %s | %d | arm not generated / judged in %s |" % (name, layer, label, N, w))
            continue
        v, n = K[arm]
        c = T.cond("C", layer, arm)
        net = [VAL[state[(i, c)]] - VAL[state[(i, base)]] for i in ids]
        sA, loA, hiA = iv(v["dAns"])
        sN, _, _ = iv(net)
        _, loX, hiX = iv(v["dX"])
        dAns, dX = sum(v["dAns"]), sum(v["dX"])
        moved = sign * dAns
        own_a = moved / N > T.TOL and sign * dX < 0
        own_b = own_a and ((hiA < 0 and loX > 0) if sign < 0 else (loA > 0 and hiX < 0))
        vs = [iv(T.minus(v["dAns"], K[r][0]["dAns"])) for r in RANDS if r in K]
        beat_a = None if missing else all(sign * (dAns - sum(K[r][0]["dAns"])) > 0 for r in RANDS)
        beat_b = None if missing else all((hi < 0) if sign < 0 else (lo > 0) for _, lo, hi in vs)
        kr = T.keep_interval(v, n)
        enough = n["keep"] > 0 and 2 * n["keep"] >= base_ans
        hedge = n["toH"] / N <= T.TOL
        keep_a = enough and hedge and kr is not None and abs(kr[0]) <= T.TOL
        keep_b = keep_a and kr[1] >= -T.TOL and kr[2] <= T.TOL
        stA, stB = T.status([own_a, beat_a, keep_a]), T.status([own_b, beat_b, keep_b])
        if sign < 0:
            moved_rw = "suppressed R %d / W %d" % (n["lost_right"], n["lost_wrong"])
        else:
            moved_rw = "released R %d / W %d" % (n["gain_right"], n["gain_wrong"])
        rnd = "; ".join("vs rand %s %s" % (r.split(".")[2][0], s) for r, (s, _, _) in zip([r for r in RANDS if r in K], vs)) or "-"
        out.append("| %s | %d | %s | %d | %d / %+d | %s | %s | %s | %d, %s | %s | %d | %s / %s | %s / %s | %s / %s | %s / %s | %s |" % (
            name, layer, label, N, base_ans, rb - wb, sA, sN, "%+d" % dX, n["keep"], pp(kr), moved_rw, n["toH"],
            tf(own_a), tf(own_b), tf(beat_a), tf(beat_b), tf(keep_a), tf(keep_b), stA, stB, rnd))
    return out


HEAD = ("| model | layer | arm | N | no write: answered / net | d answered [95%] | d net [95%] | d abstained | answered both: n, "
        "right-rate change pp [95%] | moved answers (Pro) | answered->hedge | effect A / B | beats random A / B | retention A / B | "
        "status A / B | arm minus random, d answered [95%] |")
SEP = "|" + "---|" * 17


def block(split):
    print("## %s split (%s)\n" % (split, "192 questions" if split == "tune" else "529 questions minus the 6 abstention-only gold = 523"))
    for name, run, tl, own, xl, n1 in MODELS:
        layers = tl if split == "tune" else list(xl)
        print("### %s (N1 uses C layer %d)\n" % (name, n1))
        print(HEAD)
        print(SEP)
        for layer in layers:
            for r in rows_for(name, run, layer, split, own):
                print(r)
        print()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("# C row of the free-form 2x2 by layer: plain prompt, C at 0.25, DeepSeek Pro (2026-11-24, c_row_layers.py)\n")
    print("Tolerance 3%% of the questions (tune: 6 of 192 beyond, 5 inside; test: 16 of 523 beyond). Paired bootstrap %d "
          "resamples, seed %d. 'status' = met / not_met / incomplete (incomplete = no random arm, nothing else failed).\n" % (T.BOOT, T.SEED))
    for split in (("tune", "test") if which == "all" else (which,)):
        block(split)


if __name__ == "__main__":
    main()

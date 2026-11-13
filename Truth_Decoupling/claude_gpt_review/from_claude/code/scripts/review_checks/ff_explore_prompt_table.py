#!/usr/bin/env python3
"""Prompt-level table across models (Claude, 2026-11-13; EXPLORATORY, outside the frozen criteria).
Prompt against prompt, no write, both judges, on the 192 tune questions. Two judged work dirs per model:
  step0  the frozen 11-12 step-0 judge dir (x, x_E1, x_F, conf / caut and their E1 / E2 / M variants)
  diag   the prompt diagnosis (configs/prompts/e44_promptdiag_v1.json: x, x_E1, x_E2, x_F, x_F_E1, x_F_E2)
Uses the frozen tables module's own loader and per-question quantities (e44_round1_tables.py). CPU, standard library,
writes nothing. Also checks that the conditions present in both dirs (x, x_E1, x_F) came out the same twice.
  ff_explore_prompt_table.py            all models
  ff_explore_prompt_table.py MODEL ...  only these"""
import os, statistics, sys
from pathlib import Path

CODE = Path(os.environ.get("TD_FF_CODE", "/data/sanshizhao/Truth_Decoupling/code_newmodels_ff_20261112")) / "scripts" / "review_checks"
sys.path.insert(0, str(CODE))
import e44_round1_tables as T  # noqa: E402

W = Path(os.environ.get("TD_ROOT", "/data/sanshizhao/Truth_Decoupling"))
R = W / "results_newmodels_ff_20261112"
DIRS = {
    "llama3.1-8b": {"step0": W / "results_e44_factorial_das/llama3.1-8b/judge_tune_step0",
                    "diag": R / "promptdiag/llama3.1-8b/judge_tune_diag"},
    "qwen3-8b": {"step0": R / "e44/qwen3-8b/judge_tune_step0", "diag": R / "promptdiag/qwen3-8b/judge_tune_diag"},
    "qwen3.5-9b": {"step0": R / "e44/qwen3.5-9b/judge_tune_step0", "diag": R / "promptdiag/qwen3.5-9b/judge_tune_diag"},
}
PAIRS = [("diag", "x", "x_E1"), ("diag", "x", "x_E2"), ("diag", "x", "x_F"), ("diag", "x_F", "x_F_E1"), ("diag", "x_F", "x_F_E2"),
         ("step0", "x", "conf"), ("step0", "x", "caut"), ("step0", "caut", "conf"),
         ("step0", "conf", "conf_E1"), ("step0", "conf", "conf_E2"), ("step0", "caut", "caut_E1"), ("step0", "caut", "caut_E2"),
         ("step0", "conf", "conf_M")]


def row(L, b, c):
    gen, _, ids, st, _, words = L
    v, n = T.per_question(st, words, ids, b, c)
    kk = n["keep"]
    return "%s | %s | %+d | %+d | %+d | %d: %+d (%d / %d), %s | %d / %d | %.1f -> %.1f" % (
        T.fmt_counts(st, ids, b), T.fmt_counts(st, ids, c), sum(v["dR"]), sum(v["dAns"]), sum(v["dX"]), kk, sum(v["keep"]),
        n["up"], n["down"], ("%+.1f pp" % (100.0 * sum(v["keep"]) / kk)) if kk else "undefined", n["out"], n["in"],
        statistics.mean(words[(i, b)] for i in ids), statistics.mean(words[(i, c)] for i in ids))


def main(models):
    load = {}
    for m in models:
        for k, d in DIRS[m].items():
            for j in ("allenai", "pro"):
                load[(m, k, j)] = T.load(d, j)
        print("%s conditions: step0 %s | diag %s" % (m, " ".join(load[(m, "step0", "pro")][0]["conditions"]),
                                                     " ".join(load[(m, "diag", "pro")][0]["conditions"])))
    print("\n## Same prompt in both dirs (generated twice): questions whose text / state differs")
    for m in models:
        for c in ("x", "x_E1", "x_F"):
            c = "baseline_" + c
            out = []
            for j in ("allenai", "pro"):
                g0, _, ids, s0, _, _ = load[(m, "step0", j)]
                g1, _, ids1, s1, _, _ = load[(m, "diag", j)]
                if c not in g0["conditions"] or c not in g1["conditions"]:
                    out.append("%s missing" % j); continue
                assert ids == ids1, "question order differs"
                t0 = {r["pair_id"]: r["outputs"][c] for r in g0["evaluations"]}
                t1 = {r["pair_id"]: r["outputs"][c] for r in g1["evaluations"]}
                out.append("%s: text %d, state %d" % (j, sum(t0[i] != t1[i] for i in ids), sum(s0[(i, c)] != s1[(i, c)] for i in ids)))
            print("- %s %s: %s" % (m, c[9:], "; ".join(out)))
    for k, b, c in PAIRS:
        print("\n### %s -> %s (%s dir)" % (b, c, k))
        print("| model | judge | base R/W/H/X | prompt R/W/H/X | dR | dAns | dX | keep n: dR (W>R / R>W), keep right-rate | out / in | words |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for m in models:
            for j in ("allenai", "pro"):
                L = load[(m, k, j)]
                if "baseline_" + b not in L[0]["conditions"] or "baseline_" + c not in L[0]["conditions"]:
                    print("| %s | %s | missing |" % (m, j)); continue
                print("| %s | %s | %s |" % (m, j, row(L, "baseline_" + b, "baseline_" + c)))


if __name__ == "__main__":
    main(sys.argv[1:] or list(DIRS))

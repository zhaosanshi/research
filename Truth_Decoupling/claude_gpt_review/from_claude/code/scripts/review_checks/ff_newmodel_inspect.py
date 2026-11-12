#!/usr/bin/env python3
"""Read-only look at raw answers behind the 9B free-form tables (Claude, 2026-11-13; EXPLORATORY, outside the frozen criteria).
Uses the frozen tables module's own loader and state mapping (e44_round1_tables.py, sha256 f7a502d8...), prints raw-answer
examples and small strength tables. CPU, standard library, writes nothing. Run on the server with the container's python3:
  ff_newmodel_inspect.py e_dose LAYER          E strength curve on x / x_F / conf, both judges (proB + proA work dirs)
  ff_newmodel_inspect.py c_dose LAYER          C strength curve on x, both judges
  ff_newmodel_inspect.py e_ex LAYER ARM BASE WORKDIR   examples, Pro W->R / R->R / R->W (seed 0)
  ff_newmodel_inspect.py c_ex LAYER ARM        examples, Pro R->W / W->R (seed 0)
Paths: TD_FF_CODE (frozen code copy) and TD_FF_MAIN (the model's E44 run dir), server defaults below."""
import json, os, random, statistics, sys
from pathlib import Path

CODE = Path(os.environ.get("TD_FF_CODE", "/data/sanshizhao/Truth_Decoupling/code_newmodels_ff_20261112")) / "scripts" / "review_checks"
sys.path.insert(0, str(CODE))
import e44_round1_tables as T  # noqa: E402

MAIN = Path(os.environ.get("TD_FF_MAIN", "/data/sanshizhao/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b"))


def texts(gen):
    return {(r["pair_id"], c): (t or "") for r in gen["evaluations"] for c, t in r["outputs"].items()}


def qinfo(work):
    out = {}
    for l in (work / "judge2" / "cells.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        out.setdefault(r["pair_id"], (r["question"], r["best_answer"]))
    return out


def examples(work, base, arm, want, k, seed=0):
    """want = (base_state, arm_state) under Pro; print k random such questions with both judges' states."""
    gp, _, ids, sp, _, _ = T.load(work, "pro")
    _, _, _, sa, _, _ = T.load(work, "allenai")
    tx, qi = texts(gp), qinfo(work)
    hits = [i for i in ids if (sp[(i, base)], sp[(i, arm)]) == want]
    random.Random(seed).shuffle(hits)
    print("### %s: %s -> %s, Pro %s->%s: %d questions; %d shown" % (work.name, base, arm, want[0], want[1], len(hits), min(k, len(hits))))
    for i in hits[:k]:
        q, best = qi[i]
        print("- %s  Q: %s\n    best: %s\n    base [Pro %s / AllenAI %s] %s\n    arm  [Pro %s / AllenAI %s] %s" % (
            i, q, best, sp[(i, base)], sa[(i, base)], tx[(i, base)].replace("\n", " ")[:300],
            sp[(i, arm)], sa[(i, arm)], tx[(i, arm)].replace("\n", " ")[:300]))


def strength_table(work, base, arms):
    for judge in ("allenai", "pro"):
        gen, _, ids, st, _, words = T.load(work, judge)
        print("\n#### %s, judge %s, against %s (R/W/H/X %s; mean words %.1f)" % (
            work.name, judge, base, T.fmt_counts(st, ids, base), statistics.mean(words[(i, base)] for i in ids)))
        print("| arm | R / W / H / X | dR | dAns | keep dR (W>R / R>W) | G_same (n) | mean words | median words |")
        print("|---|---|---|---|---|---|---|---|")
        for a in arms:
            if a not in gen["conditions"]:
                print("| %s | missing |" % a); continue
            v, n = T.per_question(st, words, ids, base, a)
            w = [words[(i, a)] for i in ids]
            print("| %s | %s | %+d | %+d | %+d (%d / %d) | %+d (%d) | %.1f | %.0f |" % (
                a, T.fmt_counts(st, ids, a), sum(v["dR"]), sum(v["dAns"]), sum(v["keep"]), n["up"], n["down"],
                sum(v["same"]), n["same"], statistics.mean(w), statistics.median(w)))


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "e_dose":
        L = int(sys.argv[2])
        w = MAIN / ("judge_tune_proB_E_L%d" % L)
        arms = ["E_L%d_%s" % (L, a) for a in ("x-w-p0-25", "x-w-p0-5", "x-w-p0-75", "x-w-p1-5", "x-w-p2",
                                               "x-w-m0-5", "x-w-m0-75", "x-md-p0-5", "x-rand-p0-5", "x-rand-m0-5",
                                               "x-rand-p0-75", "x-rand-m0-75")]
        strength_table(w, "baseline_x", ["baseline_x_F"] + arms)
        wa = MAIN / ("judge_tune_proA_E_L%d" % L)
        strength_table(wa, "baseline_x", ["E_L%d_x-w-p1" % L])
        strength_table(w, "baseline_x_F", ["E_L%d_x_F-w-p0-5" % L, "E_L%d_x_F-w-p0-75" % L])
        strength_table(w, "baseline_conf", ["E_L%d_conf-w-p0-5" % L, "E_L%d_conf-w-p2" % L, "E_L%d_conf-rand-p2" % L,
                                            "E_L%d_conf-rand-m2" % L, "E_L%d_conf-md-p2" % L])
    elif what == "e_ex":
        L, arm, base = int(sys.argv[2]), sys.argv[3], sys.argv[4]
        w = MAIN / sys.argv[5]
        examples(w, base, "E_L%d_%s" % (L, arm), ("W", "R"), 6)
        examples(w, base, "E_L%d_%s" % (L, arm), ("R", "R"), 3)
        examples(w, base, "E_L%d_%s" % (L, arm), ("R", "W"), 3)
    elif what == "c_dose":
        L = int(sys.argv[2])
        wb, wa = MAIN / ("judge_tune_proB_C_L%d" % L), MAIN / ("judge_tune_proA_C_L%d" % L)
        order = [("m2", wb), ("m1", wa), ("m0-5", wb), ("m0-25", wb), (None, wb), ("p0-25", wb), ("p0-5", wb), ("p1", wa), ("p2", wb)]
        for judge in ("allenai", "pro"):
            loaded = {w: T.load(w, judge) for w in (wa, wb)}
            print("\n#### C layer %d, judge %s (+ = toward caution, - = toward confident; base x)" % (L, judge))
            print("| strength | R / W / H / X | answered | right among answered | mean words |")
            print("|---|---|---|---|---|")
            for s, w in order:
                gen, _, ids, st, _, words = loaded[w]
                c = "baseline_x" if s is None else "C_L%d_x-w-%s" % (L, s)
                r, wr, h, x = T.counts(st, ids, c)
                lab = "0" if s is None else ("+" if s[0] == "p" else "-") + s[1:].replace("-", ".")
                print("| %s | %d / %d / %d / %d | %d | %.1f%% | %.1f |" % (lab, r, wr, h, x, r + wr, 100.0 * r / (r + wr),
                                                                     statistics.mean(words[(i, c)] for i in ids)))
    elif what == "c_ex":
        L, arm = int(sys.argv[2]), sys.argv[3]
        w = MAIN / ("judge_tune_proA_C_L%d" % L)
        examples(w, "baseline_x", "C_L%d_%s" % (L, arm), ("R", "W"), 10)
        examples(w, "baseline_x", "C_L%d_%s" % (L, arm), ("W", "R"), 3)

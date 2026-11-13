# -*- coding: utf-8 -*-
"""Rule R3 of from_claude/45 (Claude, 2026-11-13 22:4x; read-only: no model, no judge call). The E layer of a model, from
the verdicts already on disk in the TAG=X work dirs RUN/judge_tune_proX_E_L<layer>: the learned E direction at strength 1
on the must-answer base (arm conf.w.p1, condition E_L<layer>_conf-w-p1) against the same base without any write
(baseline_conf), on the tune questions. Labels, rule abstentions and the paired bootstrap are e44_round1_tables.py's (as
ff_explore_dose_table.py), so every number equals the one in the per-layer tables.
  top   --run RUN --layers L ... [--k 3]
        AllenAI only, every candidate layer: dR = right answers with the write minus without; prints a table and the line
        "top: <the k layers with the largest dR, best first>" (ties: the lower layer first).
  pick  --run RUN --layers L ... --out JSON
        AllenAI and DeepSeek Pro on the same cells (both verdict files must be complete): the layer with the largest sum of
        the two dR (ties: the lower layer); prints a table and the line "chosen: <layer>"; writes JSON (an existing file
        that differs is never overwritten).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t  # noqa: E402

BASE = "baseline_conf"
ARM = "conf.w.p1"


def cell(run, layer, judge):
    work = Path(run) / ("judge_tune_proX_E_L%d" % layer)
    gen, specs, ids, state, rule, words = t.load(work, judge)
    c = "E_L%d_conf-w-p1" % layer
    for need in (BASE, c):
        if need not in gen["conditions"]:
            raise SystemExit("%s has no condition %s" % (work, need))
    v, n = t.per_question(state, words, ids, BASE, c)
    return {"dR": sum(v["dR"]), "ci": t.ci(v["dR"])[0], "dAns": sum(v["dAns"]), "G_same": sum(v["same"]),
            "n_same": n["same"], "words_base": sum(words[(i, BASE)] for i in ids) / len(ids),
            "words": sum(words[(i, c)] for i in ids) / len(ids), "n": len(ids)}


def row(layer, s):
    return "| %d | %s | %+d | %+d (%d) | %.1f -> %.1f |" % (layer, s["ci"], s["dAns"], s["G_same"], s["n_same"], s["words_base"], s["words"])


def top(a):
    if len(set(a.layers)) != len(a.layers) or len(a.layers) < a.k:
        raise SystemExit("need at least %d distinct layers, got %s" % (a.k, a.layers))
    S = {L: cell(a.run, L, "allenai") for L in a.layers}
    print("# R3 (from_claude/45): AllenAI, must-answer base, learned E direction at strength 1 against no write\n")
    print("tune questions: %d; run %s\n" % (S[a.layers[0]]["n"], a.run))
    print("| layer | dR [95%] | dAns | G_same (n) | mean words |")
    print("|---|---|---|---|---|")
    for L in sorted(a.layers):
        print(row(L, S[L]))
    best = sorted(a.layers, key=lambda L: (-S[L]["dR"], L))[:a.k]
    print("\ntop: %s" % " ".join(str(L) for L in best))


def pick(a):
    if len(set(a.layers)) != len(a.layers) or not a.layers:
        raise SystemExit("need distinct layers, got %s" % a.layers)
    S = {L: {j: cell(a.run, L, j) for j in ("allenai", "pro")} for L in a.layers}
    print("# R3 (from_claude/45): the E layer = largest AllenAI + Pro dR (must-answer base, strength 1); ties: the lower layer\n")
    print("| layer | AllenAI dR [95%] | Pro dR [95%] | sum | Pro: dAns, G_same (n), mean words |")
    print("|---|---|---|---|---|")
    for L in sorted(a.layers):
        al, pr = S[L]["allenai"], S[L]["pro"]
        print("| %d | %s | %s | %+d | %+d, %+d (%d), %.1f -> %.1f |" % (
            L, al["ci"], pr["ci"], al["dR"] + pr["dR"], pr["dAns"], pr["G_same"], pr["n_same"], pr["words_base"], pr["words"]))
    chosen = sorted(a.layers, key=lambda L: (-(S[L]["allenai"]["dR"] + S[L]["pro"]["dR"]), L))[0]
    print("\nchosen: %d" % chosen)
    out = {"rule": "R3 of from_claude/45: largest AllenAI + Pro dR of %s against %s on the tune questions, ties: the lower layer" % (ARM, BASE),
           "run": a.run, "work_pattern": "judge_tune_proX_E_L<layer>",
           "layers": {str(L): {"allenai_dR": S[L]["allenai"]["dR"], "pro_dR": S[L]["pro"]["dR"],
                               "sum": S[L]["allenai"]["dR"] + S[L]["pro"]["dR"]} for L in sorted(a.layers)},
           "chosen": chosen}
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    p = Path(a.out)
    if p.exists():
        if p.read_text(encoding="utf-8") != text:
            raise SystemExit("%s exists and differs: not overwritten" % p)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("top")
    s.add_argument("--run", required=True)
    s.add_argument("--layers", nargs="+", type=int, required=True)
    s.add_argument("--k", type=int, default=3)
    s = sub.add_parser("pick")
    s.add_argument("--run", required=True)
    s.add_argument("--layers", nargs="+", type=int, required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args()
    top(a) if a.mode == "top" else pick(a)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Read-only checks behind from_claude/44 section 5.9 (Claude, 2026-11-13 18:xx; 赵磊's questions of 18:0x). No model, no
judge call: only judge inputs (judge_input.json) and single-judge verdicts (judge2/single_<judge>.jsonl) already on disk.

  same-text  On the must-answer base: among the questions where adding the truth sentence to the prompt changed the answer
             text, how often the write along the learned direction (no sentence in the prompt) gives word for word the
             same text as the prompt with the sentence; controls: random directions and the mean-difference direction
             of the same strength.
             usage: same-text --run NAME:RUN_DIR:WORK_PATTERN:E_ON:LAYER[,LAYER...] [--run ...]
                    (WORK_PATTERN contains %d for the layer, e.g. judge_tune_proX_E_L%d)
  disagree   The E1 write of the 9B's round 1 on the must-answer base at strength 1: per layer, how the AllenAI-minus-Pro
             gap in right answers splits into questions only one judge counts as gained or lost, and how many of them
             got longer; then, for one layer, the questions AllenAI newly counts right while Pro does not and whose answer
             got longer, with three of them drawn at random (seed 20261113) for the human check of section 5.9.
             usage: disagree --work-pattern RUN_DIR/judge_tune_proA_E_L%d --layers 9 11 12 13 14 15 18 --show 15
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

AMAP = {"A": "R", "B": "W", "N": "H"}          # AllenAI single labels; a rule abstention is X
PMAP = {"A": "R", "B": "W", "H": "H", "C": "X"}  # Pro single labels


def words(t):
    return len((t or "").split())


def load(work):
    """(judge input, state[(judge, pair_id, condition)] = R / W / H / X), as e44_round1_tables.load reads them."""
    work = Path(work)
    g = json.loads((work / "judge_input.json").read_text(encoding="utf-8"))
    st = {}
    for j in ("allenai", "pro"):
        for line in (work / "judge2" / ("single_%s.jsonl" % j)).open(encoding="utf-8"):
            r = json.loads(line)
            if r["status"] != "scored":
                if r.get("basis") == "empty_output_no_judge_vote" and r.get("empty_output") is True:
                    st[(j, r["pair_id"], r["condition"])] = "X"   # empty answer = abstention (two-axis standard of 11-08)
                    continue
                raise SystemExit("unscored cell %s in %s" % (r["cell_id"], work))
            st[(j, r["pair_id"], r["condition"])] = ("X" if r["rule_abstain"] else AMAP[r["label"]]) if j == "allenai" else PMAP[r["label"]]
    return g, st


def same_text(a):
    print("| run | layer | answers the prompt changed | direction: same text | random +: same | random −: same | mean difference: same |")
    print("|---|---|---|---|---|---|---|")
    for spec in a.run:
        name, run, pattern, e_on, layers = spec.split(":")
        for L in (int(x) for x in layers.split(",")):
            g = json.loads((Path(run) / (pattern % L) / "judge_input.json").read_text(encoding="utf-8"))
            prompt = "baseline_conf_" + e_on
            changed = [ev for ev in g["evaluations"] if ev["outputs"]["baseline_conf"] != ev["outputs"][prompt]]
            cells = []
            for arm in ("conf-w-p1", "conf-rand-p1", "conf-rand-m1", "conf-md-p1"):
                c = "E_L%d_%s" % (L, arm)
                cells.append("—" if c not in g["conditions"] else str(sum(ev["outputs"][c] == ev["outputs"][prompt] for ev in changed)))
            print("| %s | %d | %d / %d | %s |" % (name, L, len(changed), len(g["evaluations"]), " | ".join(cells)))


def disagree(a):
    print("| layer | dR AllenAI / Pro | AllenAI-only gains (longer) | Pro-only gains (longer) | AllenAI-only losses (longer) | "
          "Pro-only losses (longer) | answers longer (of all) | words: no write -> write; prompt + E1 |")
    print("|---|---|---|---|---|---|---|---|")
    cands_by_layer, loaded = {}, {}
    for L in a.layers:
        g, st = load(a.work_pattern % L)
        loaded[L] = (g, st)
        arm, base = "E_L%d_conf-w-p1" % L, "baseline_conf"
        c = Counter()
        dA = dP = 0
        cands = []
        for e in g["evaluations"]:
            pid = e["pair_id"]
            lon = words(e["outputs"][arm]) > words(e["outputs"][base])
            c["longer"] += lon
            A = (st[("allenai", pid, base)] == "R", st[("allenai", pid, arm)] == "R")
            P = (st[("pro", pid, base)] == "R", st[("pro", pid, arm)] == "R")
            dA += A[1] - A[0]
            dP += P[1] - P[0]
            ga, gp, la, lp = (not A[0]) and A[1], (not P[0]) and P[1], A[0] and not A[1], P[0] and not P[1]
            for key, hit in (("ag", ga and not gp), ("pg", gp and not ga), ("al", la and not lp), ("pl", lp and not la)):
                if hit:
                    c[key] += 1
                    c[key + "_long"] += lon
            if ga and not P[1] and lon:      # AllenAI: not right -> right; Pro: not right after the write; longer
                cands.append(pid)
        n = len(g["evaluations"])
        mw = lambda cond: sum(words(e["outputs"][cond]) for e in g["evaluations"]) / n  # noqa: E731
        print("| %d | %+d / %+d | %d (%d) | %d (%d) | %d (%d) | %d (%d) | %d / %d | %.1f -> %.1f; %.1f |" % (
            L, dA, dP, c["ag"], c["ag_long"], c["pg"], c["pg_long"], c["al"], c["al_long"], c["pl"], c["pl_long"],
            c["longer"], n, mw(base), mw(arm), mw("baseline_conf_E1")))
        cands_by_layer[L] = sorted(cands)
    if a.show is None:
        return
    L = a.show
    g, st = loaded[L]
    ev = {e["pair_id"]: e for e in g["evaluations"]}
    cands = cands_by_layer[L]
    pick = random.Random(20261113).sample(cands, min(3, len(cands)))
    print("\nlayer %d: %d questions AllenAI newly counts right, Pro does not, answer longer; drawn (seed 20261113): %s" % (L, len(cands), pick))
    for pid in cands:
        e = ev[pid]
        print("\n#### %s%s\n" % (pid, " (drawn)" if pid in pick else ""))
        print("- question: %s\n- best answer: %s\n- correct: %s\n- incorrect: %s" % (e["question"], e["best_answer"], e["gold"], e["incorrect"]))
        for cond in ("baseline_conf", "E_L%d_conf-w-p1" % L, "baseline_conf_E1"):
            print("- `%s` (%d words; AllenAI %s, Pro %s): %s" % (cond, words(e["outputs"][cond]), st[("allenai", pid, cond)],
                                                                 st[("pro", pid, cond)], json.dumps(e["outputs"][cond], ensure_ascii=False)))
    print("\nlayer %d: questions Pro counts right without the write and not right with it, while AllenAI counts no loss:" % L)
    arm, base = "E_L%d_conf-w-p1" % L, "baseline_conf"
    for pid, e in ev.items():
        A = (st[("allenai", pid, base)] == "R", st[("allenai", pid, arm)] == "R")
        P = (st[("pro", pid, base)] == "R", st[("pro", pid, arm)] == "R")
        if P[0] and not P[1] and not (A[0] and not A[1]):
            print("- %s | %s | no write [AllenAI %s, Pro %s] %s -> write [AllenAI %s, Pro %s] %s" % (
                pid, e["question"], st[("allenai", pid, base)], st[("pro", pid, base)], json.dumps(e["outputs"][base], ensure_ascii=False),
                st[("allenai", pid, arm)], st[("pro", pid, arm)], json.dumps(e["outputs"][arm], ensure_ascii=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("same-text")
    s.add_argument("--run", action="append", required=True)
    d = sub.add_parser("disagree")
    d.add_argument("--work-pattern", required=True)
    d.add_argument("--layers", nargs="+", type=int, required=True)
    d.add_argument("--show", type=int, default=None)
    a = p.parse_args()
    same_text(a) if a.mode == "same-text" else disagree(a)


if __name__ == "__main__":
    main()

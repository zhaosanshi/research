# Iteration_01 diagnostic (Claude Opus 5.5, 2026-11-17), read-only: no model, no GPU, no judge call.
# Step 1 of novelty/ours/2026-11-17_e_doubt_c_commit.md section 7: does the doubt that E raises separate wrong answers from right
# ones? Doubt d = log p("I don't know.") with E at strength 1 written minus the same without a write, plain prompt x, tune split
# (the per-question `abstain_logp` of e44_factorial_das.py's mc readouts). Controls: d_rand (the random write of the same size
# minus no write), a0 (no write), aE (E written). Labels from the tune E+C grid (states as policy_select2.py's load):
#   T1  questions answered (R or W) with E written: wrong vs right
#   T2  questions answered without a write: wrong vs right
#   T3  questions answered without a write: blank with E (rule detector or empty output) vs still answered (R or W)
# AUROC = P(reading of a positive > reading of a negative) + half the ties; positive = wrong (T1, T2) or blank (T3).
# 95% interval: stratified bootstrap (positives and negatives resampled separately), 4000 resamples, seed 20261117.
# Criteria and predictions were written before this script ran (section 7, step 1, items 7-8); they are printed, not tuned.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/doubt_signal.py --gemma-judge allenai|pro [MODEL ...]
import argparse
import json
import os
import random
import re
import sys

SEED = 20261117
NB = 4000
DEN = re.compile(r"^\s*(none|nothing|no|nowhere|no one|nobody|never|neither|there (is|are) no|there's no|it (is|does) not|"
                 r"they (do|are) not)\b", re.I)          # copied from denial_split.py (importing it would run it)
I1 = "Iteration_01/results"
MODELS = [  # name, run dir, E layer, tune grid (relative to the run dir), judge family ("pro" fixed, or "gemma")
    ("Llama-3.1-8B", "results_e44_factorial_das/llama3.1-8b_side_E2", 11, "eval_extra/judge_tune_ecll19_tune", "pro"),
    ("Qwen3.5-9B", "results_newmodels_ff_20261112/e44/qwen3.5-9b_E2", 15, "eval_extra/judge_tune_ec9b_tune", "pro"),
    ("Qwen3.8-27B", "results_newmodels_ff_20261112/e44/qwen3.8-27b-dm", 28, "eval_extra/judge_tune_ec2733_tune", "pro"),
    ("Gemma-3-12B", I1 + "/ff/gemma3/12b/e44/gemma3-12b", 22, "eval_extra/judge_tune_ec25_tune", "gemma"),
    ("Gemma-4-12B", I1 + "/ff/gemma4/12b/e44/gemma4-12b", 24, "eval_extra/judge_tune_ec21_tune", "gemma"),
    ("Gemma-4-31B", I1 + "/ff/gemma4/31b/e44/gemma4-31b-dm", 24, "eval_extra/judge_tune_ec28_tune", "gemma"),
    ("Gemma-3-27B", I1 + "/ff/gemma3/27b/e44/gemma3-27b", 26, "eval_extra/judge_tune_ec31_tune", "gemma"),
]
PREDICTED_READ = {"Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B", "Gemma-3-12B", "Gemma-3-27B"}   # T3 above 0.70 (item 7)


class Stop(Exception):
    pass


def load_states(work, judge):
    """As policy_select2.py's load: state R/W/H/X/? and the blank flag (rule detector or empty output) per (pair, condition)."""
    st, blank = {}, {}
    for line in open(os.path.join(work, "judge2", "single_%s.jsonl" % judge), encoding="utf-8"):
        r = json.loads(line)
        k = (r["pair_id"], r["condition"])
        if r["status"] != "scored":
            s = "X" if (r.get("basis") == "empty_output_no_judge_vote" and r.get("empty_output") is True) else "?"
        elif judge == "allenai":
            s = "X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]
        else:
            s = {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]
        if k in st:
            raise Stop("%s: duplicate verdict row %s" % (work, k))
        st[k] = s
        blank[k] = bool(r.get("rule_abstain")) or bool(r.get("empty_output"))
    return st, blank


def auroc(pos, neg):
    """Mann-Whitney AUROC with half credit for ties; None when a class is empty."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0                                # average rank of the tie block (1-based)
        rank_sum += r * sum(1 for t in range(i, j + 1) if allv[t][1] == 1)
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


MIN_CLASS = 10                                              # below this many in a class: AUROC printed, interval not


def boot(pos, neg, seed=SEED):
    a = auroc(pos, neg)
    if a is None:
        return None, None, None
    if min(len(pos), len(neg)) < MIN_CLASS:
        return a, None, None
    rng = random.Random(seed)
    bs = sorted(auroc([pos[rng.randrange(len(pos))] for _ in pos], [neg[rng.randrange(len(neg))] for _ in neg]) for _ in range(NB))
    return a, bs[100], bs[3899]


def fmt(t, n1=None, n0=None):
    a, lo, hi = t
    if a is None:
        return "—"
    if lo is None:
        return "%.3f (class under %d: no interval)" % (a, MIN_CLASS)
    return "%.3f [%.3f, %.3f]" % (a, lo, hi)


def read_mc(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = {}
    for r in d["rows"]:
        if r["pair_id"] in rows:
            raise Stop("%s: duplicate pair_id %s" % (path, r["pair_id"]))
        rows[r["pair_id"]] = r
    return d, rows


def model_report(name, run, el, grid_rel, judge, out):
    gw = os.path.join(run, grid_rel)
    if not os.path.isfile(os.path.join(gw, "judge_input.json")):
        out.append("\n## %s: not run (no tune grid %s yet)" % (name, gw))
        return None
    if not os.path.isfile(os.path.join(gw, "judge2", "single_%s.jsonl" % judge)):
        out.append("\n## %s: not run (no %s verdicts in %s yet)" % (name, judge, gw))
        return None
    e_cond = "E_L%d_x-w-p1" % el
    base_d, base = read_mc(os.path.join(run, "mc", "base_x_tune.json"))
    e_d, ew = read_mc(os.path.join(run, "mc", "E_L%d_x-w-p1_tune.json" % el))
    r_d, er = read_mc(os.path.join(run, "mc", "E_L%d_x-rand-p1_tune.json" % el))
    g = json.load(open(os.path.join(gw, "judge_input.json"), encoding="utf-8"))
    # ---- the checks of item 5: any mismatch stops this model
    if not (base_d.get("prompt") == "x" and base_d.get("kind") == "baseline" and base_d.get("split") == "tune"):
        raise Stop("%s: base_x_tune.json is not the no-write readout of prompt x on tune" % name)
    for d, arm in ((e_d, "x.w.p1"), (r_d, "x.rand.p1")):
        if d.get("arm") != arm or d.get("base_prompt") != "x" or d.get("family") != "E" or d.get("layer") != el or d.get("split") != "tune":
            raise Stop("%s: mc file of %s is not E%d %s on x / tune" % (name, arm, el, arm))
    ids = set(g["question_ids"])
    if len(ids) != len(g["question_ids"]):
        raise Stop("%s: the grid repeats a question id" % name)
    for d, rows, what in ((base_d, base, "base"), (e_d, ew, "E"), (r_d, er, "rand")):
        if set(rows) != ids:
            raise Stop("%s: the %s readout and the grid hold different questions" % (name, what))
        if d.get("identity_sha256") != g.get("identity_sha256"):
            raise Stop("%s: the %s readout belongs to another run identity" % (name, what))
    spec = {c["name"]: c for c in g["condition_specs"]}.get(e_cond)
    if spec is None or "baseline_x" not in g["conditions"] or e_cond not in g["conditions"]:
        raise Stop("%s: the grid lacks baseline_x or %s" % (name, e_cond))
    parts = spec["parts"]
    if len(parts) != 1 or parts[0]["kind"] != "w" or parts[0]["direction_file_sha256"] != e_d["direction_file_sha256"] \
            or abs(float(parts[0]["amount"]) - float(e_d["write"])) > 1e-3 or float(parts[0]["beta"]) != float(e_d["beta"]):
        raise Stop("%s: the grid's %s was made with another E write than the readout's (direction / amount / strength)" % (name, e_cond))
    gen_name = "x--E%d-w-p1_tune.json" % el
    if gen_name not in g["sources"]["gen"]:
        raise Stop("%s: the grid's sources list no %s" % (name, gen_name))
    gen = json.load(open(os.path.join(run, "eval_extra", "gen", gen_name), encoding="utf-8"))
    if gen.get("positions") != e_d.get("positions") or gen["parts"][0]["direction_file_sha256"] != e_d["direction_file_sha256"] \
            or abs(float(gen["parts"][0]["amount"]) - float(e_d["write"])) > 1e-3:
        raise Stop("%s: the generation of %s wrote at other positions or another write (%s vs %s)" % (name, e_cond, gen.get("positions"), e_d.get("positions")))
    st, bl = load_states(gw, judge)
    unscored = sorted(p for p in ids if not (base[p].get("abstain_scored") and ew[p].get("abstain_scored") and er[p].get("abstain_scored")))
    qs = sorted(p for p in ids if p not in unscored)
    miss = [p for p in qs if (p, "baseline_x") not in st or (p, e_cond) not in st]
    if miss:
        raise Stop("%s: %d questions have no verdict row for baseline_x or %s, e.g. %s" % (name, len(miss), e_cond, miss[:3]))
    # ---- readings
    a0 = {p: float(base[p]["abstain_logp"]) for p in qs}
    aE = {p: float(ew[p]["abstain_logp"]) for p in qs}
    aR = {p: float(er[p]["abstain_logp"]) for p in qs}
    read = {"d": {p: aE[p] - a0[p] for p in qs}, "d_rand": {p: aR[p] - a0[p] for p in qs}, "a0": a0, "aE": aE}
    # ---- kind of question (denial_split.py): from the baseline_x cell's gold answers
    cells = {}
    for line in open(os.path.join(gw, "judge2", "cells.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] == "baseline_x":
            cells[r["pair_id"]] = r
    kind = {p: ("D" if cells[p].get("gold_is_abstention") or any(DEN.match(x) for x in cells[p]["gold"]) else "S") for p in qs}
    # ---- labels
    sB = {p: st[(p, "baseline_x")] for p in qs}
    sE = {p: st[(p, e_cond)] for p in qs}
    bE = {p: bl[(p, e_cond)] for p in qs}
    unknown = sum(1 for p in qs if sB[p] == "?" or sE[p] == "?")
    T = {
        "T1": ([p for p in qs if sE[p] == "W"], [p for p in qs if sE[p] == "R"]),
        "T2": ([p for p in qs if sB[p] == "W"], [p for p in qs if sB[p] == "R"]),
        "T3": ([p for p in qs if sB[p] in "RW" and bE[p]], [p for p in qs if sB[p] in "RW" and not bE[p] and sE[p] in "RW"]),
        "T1 D": ([p for p in qs if sE[p] == "W" and kind[p] == "D"], [p for p in qs if sE[p] == "R" and kind[p] == "D"]),
        "T1 S": ([p for p in qs if sE[p] == "W" and kind[p] == "S"], [p for p in qs if sE[p] == "R" and kind[p] == "S"]),
    }
    read["-a0"] = {p: -a0[p] for p in qs}
    res = {}
    for t, (pos, neg) in T.items():
        for k, v in read.items():
            res[(k, t)] = boot([v[p] for p in pos], [v[p] for p in neg])
    # d compared only inside thirds of a0 (pairs of a wrong and a right answer from the same third)
    pos, neg = T["T1"]
    cut = sorted(a0[p] for p in pos + neg)
    third = lambda p: 0 if a0[p] <= cut[len(cut) // 3] else (1 if a0[p] <= cut[2 * len(cut) // 3] else 2)
    num = den = 0.0
    for z in range(3):
        ps, ns = [read["d"][p] for p in pos if third(p) == z], [read["d"][p] for p in neg if third(p) == z]
        if ps and ns:
            num += auroc(ps, ns) * len(ps) * len(ns)
            den += len(ps) * len(ns)
    strat = num / den if den else None
    # ---- report
    out.append("\n## %s (judge %s; E layer %d; grid %s)" % (name, judge, el, grid_rel))
    out.append("- checks passed: same %d questions in the three readouts and the grid; one run identity %s; the grid's %s = the readout's write "
               "(direction %s, amount %s, strength %s, positions %s)" % (len(ids), g["identity_sha256"][:12], e_cond, e_d["direction_file_sha256"][:12],
                                                                       e_d["write"], e_d["beta"], e_d["positions"]))
    out.append("- excluded: %d with an unscored readout; %d questions with a state '?' in baseline_x or %s (left out of the labels)" % (len(unscored), unknown, e_cond))
    cnt = lambda s, c: sum(1 for p in qs if s[p] == c)
    out.append("- states without a write R/W/H/X %d/%d/%d/%d; with E %d/%d/%d/%d; blank with E (rule or empty) %d; questions D %d, S %d" % (
        cnt(sB, "R"), cnt(sB, "W"), cnt(sB, "H"), cnt(sB, "X"), cnt(sE, "R"), cnt(sE, "W"), cnt(sE, "H"), cnt(sE, "X"),
        sum(bE.values()), sum(1 for p in qs if kind[p] == "D"), sum(1 for p in qs if kind[p] == "S")))
    out.append("")
    out.append("| reading | T1 wrong vs right with E (%d / %d) | T2 wrong vs right without (%d / %d) | T3 blank vs answered (%d / %d) | T1 on D (%d / %d) | T1 on S (%d / %d) |" % (
        len(T["T1"][0]), len(T["T1"][1]), len(T["T2"][0]), len(T["T2"][1]), len(T["T3"][0]), len(T["T3"][1]),
        len(T["T1 D"][0]), len(T["T1 D"][1]), len(T["T1 S"][0]), len(T["T1 S"][1])))
    out.append("|---|---|---|---|---|---|")
    for k in read:
        out.append("| %s | %s |" % (k, " | ".join(fmt(res[(k, t)]) for t in ("T1", "T2", "T3", "T1 D", "T1 S"))))
    out.append("")
    out.append("- -a0 is the starting level with its sign flipped (no E needed). d within thirds of a0 on T1 (wrong and right answers "
               "compared only inside the same third): %s" % ("—" if strat is None else "%.3f" % strat))
    # quintiles of d over the questions answered with E
    ans = sorted((read["d"][p], p) for p in qs if sE[p] in "RW")
    out.append("")
    out.append("Questions answered with E (%d), split into fifths by d (low doubt first): right / wrong per fifth" % len(ans))
    fifths = []
    for q in range(5):
        part = ans[q * len(ans) // 5:(q + 1) * len(ans) // 5]
        fifths.append("%d / %d (d %.2f to %.2f)" % (sum(sE[p] == "R" for _, p in part), sum(sE[p] == "W" for _, p in part), part[0][0], part[-1][0]) if part else "—")
    out.append("- " + " | ".join(fifths))
    # criteria, as written before the run
    a, lo, hi = res[("d", "T1")]
    ra, rlo, rhi = res[("d_rand", "T1")]
    usable = a is not None and a >= 0.60 and lo > 0.50 and rlo is not None and rlo <= 0.50 <= rhi
    t3 = res[("d", "T3")][0]
    out.append("- criterion 'usable' (T1: AUROC of d >= 0.60, its lower bound > 0.50, the interval of d_rand holding 0.50): %s" % ("MET" if usable else "not met"))
    n3 = len(T["T3"][0])
    out.append("- read check (T3: does the change d predict which answered questions go blank; predicted above 0.70 for Llama, the two Qwen "
               "and Gemma-3): %s, %d blank%s%s; the level aE on T3: %s (whether the reading itself reflects blanks)" % (
        "—" if t3 is None else "%.3f" % t3, n3, "" if n3 >= MIN_CLASS else " (too few for an interval)",
        "" if name not in PREDICTED_READ or t3 is None else ("; prediction held" if t3 > 0.70 else "; prediction failed"),
        fmt(res[("aE", "T3")])))
    out.append("- predictions (Fable 11-17): T2 AUROC of d > 0.55 -> %s; T1 > 0.55 for Llama / Qwen -> %s" % (
        "—" if res[("d", "T2")][0] is None else ("held" if res[("d", "T2")][0] > 0.55 else "failed"),
        ("held" if a is not None and a > 0.55 else "failed") if name in ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B") else "n/a"))
    return {"usable": usable, "T1": res[("d", "T1")], "T1r": res[("d_rand", "T1")], "T3": t3, "n3": n3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-judge", required=True, choices=("allenai", "pro"))
    ap.add_argument("models", nargs="*")
    args = ap.parse_args()
    want = args.models or [m[0] for m in MODELS]
    unknown = [w for w in want if w not in {m[0] for m in MODELS}]
    if unknown:
        sys.exit("unknown model(s) %s" % unknown)
    out = ["# Doubt signal on the tune split (step 1 of novelty/ours/2026-11-17_e_doubt_c_commit.md); Gemma labels: %s" % args.gemma_judge,
           "AUROC [95%% interval] (stratified bootstrap, %d resamples, seed %d); positive = wrong (T1, T2) or blank (T3)." % (NB, SEED)]
    summary = []
    for name, run, el, grid, fam in MODELS:
        if name not in want:
            continue
        judge = "pro" if fam == "pro" else args.gemma_judge
        try:
            r = model_report(name, run, el, grid, judge, out)
        except Stop as e:
            out.append("\n## %s: STOPPED: %s" % (name, e))
            r = None
        if r is not None:
            summary.append((name, judge, r))
    out.append("\n## Summary")
    for name, judge, r in summary:
        prov = " (provisional: the criterion counts only with Pro labels, section 7 step 0 item 6)" if judge == "allenai" else ""
        out.append("- %s (%s labels): T1 AUROC of d %s, of d_rand %s -> %s%s; T3 %s with %d blank" % (
            name, judge, fmt(r["T1"]), fmt(r["T1r"]), "usable" if r["usable"] else "not usable", prov,
            "—" if r["T3"] is None else "%.3f" % r["T3"], r["n3"]))
    print("\n".join(out))


if __name__ == "__main__":
    main()

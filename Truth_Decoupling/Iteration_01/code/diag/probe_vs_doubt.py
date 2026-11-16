# Iteration_01 diagnostic (Claude Opus 5.5, 2026-11-17), read-only: no model, no GPU, no judge call.
# Step 5 item 3 of novelty/ours/2026-11-17_e_doubt_c_commit.md section 7: per question, the change E causes in the downstream
# C readout of the prompt forward (the fixed probe, ff_ec_probe.py protocol v2: delta over gap_C minus the ACTUAL direct part,
# last prompt token) against the doubt d of step 1, and its AUROC on the labels T1 / T2 / T3 of step 1. A correlation here
# does not make C a causal mediator (GPT review S3); it only says whether the two readings move together.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/probe_vs_doubt.py --gemma-judge allenai|pro
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doubt_signal as S   # noqa: E402  (main() guarded)

RUNS = [  # model, run dir, E layer, tune grid, probe name
    ("Gemma-3-12B", S.I1 + "/ff/gemma3/12b/e44/gemma3-12b", 22, "eval_extra/judge_tune_ec25_tune", "g3plain"),
    ("Gemma-4-12B", S.I1 + "/ff/gemma4/12b/e44/gemma4-12b", 24, "eval_extra/judge_tune_ec21_tune", "g4plain"),
]


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-judge", required=True, choices=("allenai", "pro"))
    args = ap.parse_args()
    out = ["# The E-caused change of the downstream C readout vs the doubt d (tune 192; Gemma labels: %s)" % args.gemma_judge,
           "dC = (coordinate with the write - without) / gap_C - actual direct, last prompt token; > 0 = toward caution. "
           "Correlation: Spearman over questions. AUROC as step 1 (positive = wrong for T1 / T2, blank for T3)."]
    for name, run, el, grid, probe in RUNS:
        pj = json.load(open(os.path.join(run, "eval_extra", "probe_c_under_e_%s.json" % probe), encoding="utf-8"))
        if pj.get("partial") or pj.get("split") != "tune":
            sys.exit("%s: the probe file is partial or not of the tune split" % name)
        gaps = {str(k): float(v["gap"]) for k, v in pj["c_directions"].items()}
        rows = {r["pair_id"]: r for r in pj["rows"]}
        _, base = S.read_mc(os.path.join(run, "mc", "base_x_tune.json"))
        _, ew = S.read_mc(os.path.join(run, "mc", "E_L%d_x-w-p1_tune.json" % el))
        if set(rows) != set(base) or set(rows) != set(ew):
            sys.exit("%s: the probe and the readouts hold different questions" % name)
        qs = sorted(rows)
        d = {p: float(ew[p]["abstain_logp"]) - float(base[p]["abstain_logp"]) for p in qs}
        st, bl = S.load_states(os.path.join(run, grid), args.gemma_judge)
        e = "E_L%d_x-w-p1" % el
        sB = {p: st[(p, "baseline_x")] for p in qs}
        sE = {p: st[(p, e)] for p in qs}
        bE = {p: bl[(p, e)] for p in qs}
        T = {"T1": ([p for p in qs if sE[p] == "W"], [p for p in qs if sE[p] == "R"]),
             "T2": ([p for p in qs if sB[p] == "W"], [p for p in qs if sB[p] == "R"]),
             "T3": ([p for p in qs if sB[p] in "RW" and bE[p]], [p for p in qs if sB[p] in "RW" and not bE[p] and sE[p] in "RW"])}
        out.append("\n## %s (E%d; probe %s; labels %s)" % (name, el, probe, args.gemma_judge))
        out.append("| C layer | arm | mean dC | Spearman(dC, d) | AUROC T1 (%d / %d) | AUROC T2 (%d / %d) | AUROC T3 (%d / %d) |" % (
            len(T["T1"][0]), len(T["T1"][1]), len(T["T2"][0]), len(T["T2"][1]), len(T["T3"][0]), len(T["T3"][1])))
        out.append("|---|---|---|---|---|---|---|")
        for cl in sorted(gaps, key=int):
            for arm in ("E1", "E2", "rand1"):
                dc = {p: (rows[p]["coordinates"][arm][cl][0] - rows[p]["coordinates"]["none"][cl][0]) / gaps[cl]
                      - rows[p]["actual_direct"][arm][cl][0] for p in qs}
                cells = [S.fmt(S.boot([dc[p] for p in T[t][0]], [dc[p] for p in T[t][1]])) for t in ("T1", "T2", "T3")]
                out.append("| %s | %s | %+.3f | %+.3f | %s |" % (cl, arm, sum(dc.values()) / len(qs), spearman([dc[p] for p in qs], [d[p] for p in qs]),
                                                              " | ".join(cells)))
    print("\n".join(out))


if __name__ == "__main__":
    main()

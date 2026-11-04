#!/usr/bin/env python3
"""E12 (teacher's steps 6-7, within-dataset version): does the knowledge-prediction gap predict the steering gain on
subsets that were not used to choose anything? Pure post-processing of the E10 per-item file (no GPU).

For every TruthfulQA category (n >= --min-n) and for the Adversarial / Non-Adversarial split, compute on the baseline:
  out   = output AUROC (P(True) of the unsteered pass)
  probe = AUROC of the clean knowledge-probe score s (probe fitted on calib+dev only, never on test)
  gap   = probe - out
and the realised effect of the ungated evidence write (global_a3): dAUROC, d accuracy over all items, corrected, damaged.
Then rank-correlate (Spearman) each of three predictors (out alone, probe alone, gap) with each realised effect across subsets.
The alpha (3) and layers (12+13+14) were chosen on the whole test set, not per subset, so subset-level gains are not tuned.

Usage: python3 scripts/e12_subset_prediction.py --npz results_tqa/E10/llama3.1-8b/gating.npz --test data/processed/truthfulqa_tf/test.jsonl
Output: <npz dir>/subset_prediction.{md,json}
"""
import argparse
import json
import os

import numpy as np

from e10_splithalf import auroc, state


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--cond", default="global_a3")
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    meta = {}
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta[r["id"]] = r.get("meta", {})
    ids = [str(i) for i in z["ids"]]
    cat = np.array([meta.get(i, {}).get("category", "?") for i in ids])
    typ = np.array([meta.get(i, {}).get("type", "?") for i in ids])
    labels = z["labels"].astype(bool)
    s, bp, bpred = z["probe_s"], z["baseline__p"], z["baseline__pred"]
    cp, cpred = z[args.cond + "__p"], z[args.cond + "__pred"]
    sb, sc = state(bpred, labels), state(cpred, labels)

    def row(name, m):
        n = int(m.sum())
        out, pr = auroc(bp[m], labels[m]), auroc(s[m], labels[m])
        acc_b, acc_c = float((sb[m] == "known").mean()), float((sc[m] == "known").mean())
        return {"subset": name, "n": n, "out_auroc": out, "probe_auroc": pr, "gap": pr - out, "d_auroc": auroc(cp[m], labels[m]) - out,
                "acc_base": acc_b, "d_acc": acc_c - acc_b, "corrected": int(((sb[m] == "wrong") & (sc[m] == "known")).sum()),
                "damaged": int(((sb[m] == "known") & (sc[m] == "wrong")).sum()), "disagree_frac": float(z["disagree"][m].mean())}

    rows = []
    for c in sorted(set(cat)):
        m = cat == c
        if m.sum() >= args.min_n:
            rows.append(row(c, m))
    type_rows = [row(t, typ == t) for t in sorted(set(typ))]
    preds = {"output AUROC alone (lower = more room)": [-r["out_auroc"] for r in rows], "probe AUROC alone": [r["probe_auroc"] for r in rows], "gap = probe - output": [r["gap"] for r in rows],
             "disagreement share": [r["disagree_frac"] for r in rows]}
    targets = {"dAUROC": [r["d_auroc"] for r in rows], "d acc (all items)": [r["d_acc"] for r in rows], "corrected - damaged": [r["corrected"] - r["damaged"] for r in rows]}
    corr = {p: {t: spearman(pv, tv) for t, tv in targets.items()} for p, pv in preds.items()}

    out_dir = os.path.dirname(os.path.abspath(args.npz))
    stem = os.path.join(out_dir, "subset_prediction" + (("_" + args.tag) if args.tag else ""))
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"cond": args.cond, "min_n": args.min_n, "categories": rows, "types": type_rows, "spearman": corr}, f, indent=1, ensure_ascii=False)
    lines = ["# E12 subset-level prediction of the steering gain (%s, condition %s; categories with n >= %d)" % (os.path.basename(args.npz), args.cond, args.min_n), "",
             "| subset | n | output AUROC | clean probe AUROC | gap | realised dAUROC | base acc | d acc (all) | corrected | damaged | disagree share |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["gap"]) + type_rows:
        lines.append("| %s | %d | %.3f | %.3f | %+.3f | %+.3f | %.3f | %+.3f | %d | %d | %.2f |" % (
            r["subset"], r["n"], r["out_auroc"], r["probe_auroc"], r["gap"], r["d_auroc"], r["acc_base"], r["d_acc"], r["corrected"], r["damaged"], r["disagree_frac"]))
    lines += ["", "## Spearman rank correlation across the %d categories" % len(rows), "", "| predictor | vs dAUROC | vs d acc | vs corrected - damaged |", "|---|---|---|---|"]
    for p, d in corr.items():
        lines.append("| %s | %+.2f | %+.2f | %+.2f |" % (p, d["dAUROC"], d["d acc (all items)"], d["corrected - damaged"]))
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()

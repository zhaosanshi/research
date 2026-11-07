#!/usr/bin/env python3
"""For every C condition of the 2x2 C sweep: test-half d answer rate, dJ, dAUROC, dB with pair-bootstrap 95% CIs, and the
equivalence margin = smallest delta such that the whole 95% CI lies within +-delta. Then the Pareto view: among conditions
that move the answer rate by at least T, the smallest achievable margins. usage: c_equivalence.py MODEL_DIR [min_shift]"""
import json, sys
import numpy as np

d = sys.argv[1]; T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
z = np.load(d + "/knob_sweep_das_tfce_C.npz", allow_pickle=True)
labels = z["labels"].astype(bool); test = ~z["half_a"].astype(bool)
ids = z["ids"]; pair = np.array([i.rsplit(":", 1)[0] for i in ids])
rng = np.random.default_rng(0)
tp = np.array(sorted(set(pair[test]))); where = {p: np.where((pair == p) & test)[0] for p in tp}
boots = [np.concatenate([where[p] for p in rng.choice(tp, len(tp), replace=True)]) for _ in range(2000)]
tidx = np.where(test)[0]

def auroc(s, y):
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); s_sorted = s[order]
    # average ranks with ties
    i = 0; ranks = np.empty(len(s))
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    npos = y.sum(); nneg = len(y) - npos
    return (ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def mets(c, idx):
    a = z[c + "__answer"][idx]; p = z[c + "__ptrue"][idx]; y = labels[idx]; pred = p > 0.5
    return np.array([a.mean(), pred[y].mean() - pred[~y].mean(), auroc(p, y), pred.mean()])

base = mets("baseline", tidx); baseb = [mets("baseline", b) for b in boots]
conds = sorted({k.rsplit("__", 1)[0] for k in z.files if k.endswith("__answer") and k != "baseline__answer"})
rows = []
for c in conds:
    dm = mets(c, tidx) - base
    if abs(dm[0]) < T: continue
    bd = np.array([mets(c, b) - bb for b, bb in zip(boots, baseb)])
    lo, hi = np.percentile(bd, 2.5, axis=0), np.percentile(bd, 97.5, axis=0)
    rows.append({"cond": c, "d_ans": dm[0], "dJ": dm[1], "dJ_ci": [lo[1], hi[1]], "dJ_margin": max(abs(lo[1]), abs(hi[1])),
                 "dAUROC": dm[2], "dAUROC_ci": [lo[2], hi[2]], "dAUROC_margin": max(abs(lo[2]), abs(hi[2])),
                 "dB": dm[3], "dB_ci": [lo[3], hi[3]]})
print("%s: %d C conditions with |d answer rate| >= %.2f on the test half (baseline ans %.3f, J %.3f, AUROC %.3f)" % (d.rstrip("/").split("/")[-1], len(rows), T, base[0], base[1], base[2]))
for key in ("dJ_margin", "dAUROC_margin"):
    best = sorted(rows, key=lambda r: r[key])[:5]
    print("  smallest %s:" % key)
    for r in best:
        print("    %-20s d_ans %+.3f | dJ %+.4f [%+.4f, %+.4f] | dAUROC %+.4f [%+.4f, %+.4f] | dB %+.3f" % (
            r["cond"], r["d_ans"], r["dJ"], *r["dJ_ci"], r["dAUROC"], *r["dAUROC_ci"], r["dB"]))
for sgn, nm in ((-1, "abst"), (1, "ans")):
    sub = [r for r in rows if np.sign(r["d_ans"]) == sgn]
    if sub:
        bj = min(sub, key=lambda r: r["dJ_margin"]); ba = min(sub, key=lambda r: r["dAUROC_margin"])
        print("  %s: best J-margin %.4f (%s, d_ans %+.3f); best AUROC-margin %.4f (%s, d_ans %+.3f)" % (nm, bj["dJ_margin"], bj["cond"], bj["d_ans"], ba["dAUROC_margin"], ba["cond"], ba["d_ans"]))
json.dump(rows, open(d + "/c_equivalence_test.json", "w"), indent=1, default=float)

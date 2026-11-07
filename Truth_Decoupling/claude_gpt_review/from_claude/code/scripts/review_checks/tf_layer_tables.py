#!/usr/bin/env python3
"""Markdown tables for the Dr.'s package, from the 2x2 sweep files (test half = 529 pairs = 1058 statements).
usage: make_layer_tables.py MODEL_DIR MODEL_LABEL CHOSEN_E C_ESTIMATOR(Cp|Cs) CHOSEN_C_ABST CHOSEN_C_ANS CLEAN_C_ABST CLEAN_C_ANS
Writes MODEL_DIR/layer_tables.md and MODEL_DIR/c_per_layer_test.json."""
import json, sys
import numpy as np

d, label, chosen_e, cest, c_abst, c_ans, clean_abst, clean_ans = sys.argv[1:9]

def auroc(s, y):
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ss = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    npos = y.sum(); return (ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))

def mets(z, c, idx, labels):
    a = z[c + "__answer"][idx]; p = z[c + "__ptrue"][idx]; y = labels[idx]; pred = p > 0.5
    return np.array([a.mean(), pred[y].mean() - pred[~y].mean(), auroc(p, y), pred.mean()])

ze = np.load(d + "/knob_sweep_das_tfce_full.npz", allow_pickle=True); zc = np.load(d + "/knob_sweep_das_tfce_C.npz", allow_pickle=True)
labels = ze["labels"].astype(bool); test = ~ze["half_a"].astype(bool); tidx = np.where(test)[0]
assert np.array_equal(zc["ids"], ze["ids"]) and np.array_equal(zc["half_a"], ze["half_a"])
base = mets(ze, "baseline", tidx, labels); basec = mets(zc, "baseline", tidx, labels)
layers = sorted({int(k.split("_L")[1].split("_")[0]) for k in ze.files if k.startswith("E_L") and k.endswith("__answer")}); nL = len(layers)
out = []
# ---- E per layer (full) ----
out += ["### %s：E 逐层（%d 层；原模型 自报作答率 %.3f、J %.3f、AUROC %.3f；规则选中 %s）" % (label, nL, base[0], base[1], base[2], chosen_e), "",
        "| 层 | 深度 | α2 ΔJ | α2 ΔAUROC | α2 Δ作答率 | α3 ΔJ | α3 ΔAUROC | α3 Δ作答率 |", "|---|---|---|---|---|---|---|---|"]
for l in layers:
    cells = []
    for a in (2, 3):
        m = mets(ze, "E_L%d_a%d" % (l, a), tidx, labels) - base
        leak = "（漏）" if abs(m[0]) > 0.03 else ""
        cells += ["%+.3f" % m[1], "%+.3f" % m[2], "%+.3f%s" % (m[0], leak)]
    mark = " **←选中**" if any(chosen_e == "E_L%d_a%d" % (l, a) for a in (2, 3)) else ""
    out.append("| L%d%s | %.2f | %s |" % (l, mark, (l + 1) / nL, " | ".join(cells)))
# ---- C per layer (protocol estimator, alpha 2, both signs) ----
cl = []
out += ["", "### %s：C 逐层（协议估计器 %s，α = 2；原模型同上）" % (label, "提示对比" if cest == "Cp" else "自报"), "",
        "| 层 | 深度 | 推弃答 Δ作答率 | ΔAUROC | ΔJ | Δ偏置 | 推作答 Δ作答率 | ΔAUROC | ΔJ | Δ偏置 |", "|---|---|---|---|---|---|---|---|---|---|"]
for l in layers:
    row = {"layer": l, "depth": (l + 1) / nL}; cells = []
    for nm in ("abst", "ans"):
        c = "%s_L%d_%s_a2" % (cest, l, nm)
        if c + "__answer" not in zc.files:
            cells += ["—"] * 4; continue
        m = mets(zc, c, tidx, labels) - basec; row[nm] = {"d_ans": m[0], "dJ": m[1], "dAUROC": m[2], "dB": m[3]}
        cells += ["%+.3f" % m[0], "%+.4f" % m[2], "%+.3f" % m[1], "%+.3f" % m[3]]
    mark = " **←选中**" if any(x.startswith("%s_L%d_" % (cest, l)) for x in (c_abst, c_ans)) else ""
    out.append("| L%d%s | %.2f | %s |" % (l, mark, (l + 1) / nL, " | ".join(cells))); cl.append(row)
json.dump(cl, open(d + "/c_per_layer_test.json", "w"), indent=1, default=float)
for nm, sgn in (("abst", -1), ("ans", 1)):
    strong = [r["layer"] for r in cl if nm in r and sgn * r[nm]["d_ans"] >= 0.20]
    out.append("")
    out.append("推%s：作答率移动 ≥ 20 个百分点的层：%s（深度 %.2f–%.2f）" % ("弃答" if nm == "abst" else "作答", ("L%d–L%d，共 %d 层" % (min(strong), max(strong), len(strong))) if strong else "无",
               (min(strong) + 1) / nL if strong else 0, (max(strong) + 1) / nL if strong else 0))
# ---- strict rows with bootstrap ----
pair = np.array([i.rsplit(":", 1)[0] for i in zc["ids"]]); rng = np.random.default_rng(0)
tp = np.array(sorted(set(pair[test]))); where = {p: np.where((pair == p) & test)[0] for p in tp}
boots = [np.concatenate([where[p] for p in rng.choice(tp, len(tp), replace=True)]) for _ in range(2000)]
baseb = [mets(zc, "baseline", b, labels) for b in boots]
def describe(c, tag):
    m = mets(zc, c, tidx, labels) - basec
    bd = np.array([mets(zc, c, b, labels) - bb for b, bb in zip(boots, baseb)])
    lo, hi = np.percentile(bd, 2.5, axis=0), np.percentile(bd, 97.5, axis=0)
    pb = zc["baseline__ptrue"][tidx] > 0.5; pc = zc[c + "__ptrue"][tidx] > 0.5; ab = zc["baseline__answer"][tidx]; ac = zc[c + "__answer"][tidx]
    return "| %s（%s） | %+.0f 点 [%+.0f, %+.0f] | %d / 1058 | %d / 1058 | %+.4f [%+.4f, %+.4f] | %+.3f [%+.3f, %+.3f] | %+.3f [%+.3f, %+.3f] |" % (
        tag, c, 100 * m[0], 100 * lo[0], 100 * hi[0], int((ab != ac).sum()), int((pb != pc).sum()), m[2], lo[2], hi[2], m[1], lo[1], hi[1], m[3], lo[3], hi[3])
out += ["", "### %s：最优层上 C 对判别的全部影响（测试集 1058 条陈述）" % label, "",
        "| C | 自报作答率变化 [95% 区间] | 自报翻转数 | 真假结论被改数 | ΔAUROC [95% 区间] | ΔJ [95% 区间] | Δ偏置 [95% 区间] |", "|---|---|---|---|---|---|---|",
        describe(c_abst, "规则选中，推弃答"), describe(c_ans, "规则选中，推作答"), describe(clean_abst, "AUROC 最干净层，推弃答"), describe(clean_ans, "AUROC 最干净层，推作答")]
conds = sorted({k.rsplit("__", 1)[0] for k in zc.files if k.endswith("__answer") and k != "baseline__answer"})
strong = [c for c in conds if abs((mets(zc, c, tidx, labels) - basec)[0]) >= 0.20]
dA = np.array([abs((mets(zc, c, tidx, labels) - basec)[2]) for c in strong]); dJ = np.array([abs((mets(zc, c, tidx, labels) - basec)[1]) for c in strong])
out.append(""); out.append("全部 %d 个强 C 条件（两种估计器、两档力度、两个方向，作答率移动 ≥ 20 点，不挑）：|ΔAUROC| 中位数 %.4f、最大 %.4f；|ΔJ| 中位数 %.3f、最大 %.3f。" % (len(strong), np.median(dA), dA.max(), np.median(dJ), dJ.max()))
open(d + "/layer_tables.md", "w").write("\n".join(out) + "\n"); print("\n".join(out[-8:]))

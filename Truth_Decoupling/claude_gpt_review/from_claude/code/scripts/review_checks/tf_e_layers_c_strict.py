#!/usr/bin/env python3
"""(a) E per-layer table (test half): d ans / dJ / dAUROC for alpha 2 and 3 at every layer, chosen layer marked, depth summary.
(b) C 'strict' table: protocol-chosen C and tune-selected min-AUROC-margin C (both signs): test deltas with 95% CIs, flips.
usage: e_layers_c_strict.py MODEL_DIR CHOSEN_E CHOSEN_C_ABST CHOSEN_C_ANS"""
import json, sys
import numpy as np

d, chosen_e, c_abst, c_ans = sys.argv[1:5]
model = d.rstrip("/").split("/")[-1]

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

# ---------------- (a) E per layer ----------------
ze = np.load(d + "/knob_sweep_das_tfce_full.npz", allow_pickle=True)
labels = ze["labels"].astype(bool); test = ~ze["half_a"].astype(bool); tidx = np.where(test)[0]
base = mets(ze, "baseline", tidx, labels)
layers = sorted({int(k.split("_L")[1].split("_")[0]) for k in ze.files if k.startswith("E_L") and k.endswith("__answer")})
nL = len(layers); rows = []
print("## %s  E per layer (test half, d vs baseline; baseline ans %.3f J %.3f AUROC %.3f); chosen %s" % (model, base[0], base[1], base[2], chosen_e))
print("| layer | depth | a2: d ans | a2: dJ | a2: dAUROC | a3: d ans | a3: dJ | a3: dAUROC |")
print("|---|---|---|---|---|---|---|---|")
for l in layers:
    r = {"layer": l, "depth": (l + 1) / nL}
    cells = []
    for a in (2, 3):
        m = mets(ze, "E_L%d_a%d" % (l, a), tidx, labels) - base
        r["a%d" % a] = {"d_ans": m[0], "dJ": m[1], "dAUROC": m[2]}
        cells += ["%+.3f" % m[0], "%+.3f" % m[1], "%+.3f" % m[2]]
    mark = " **←选中**" if any(chosen_e == "E_L%d_a%d" % (l, a) for a in (2, 3)) else ""
    print("| L%d%s | %.2f | %s |" % (l, mark, r["depth"], " | ".join(cells))); rows.append(r)
for a in (2, 3):
    top = sorted(rows, key=lambda r: -r["a%d" % a]["dJ"])[:5]
    print("a%d top-5 by dJ: %s" % (a, ", ".join("L%d (%.2f) %+.3f/%+.3f" % (r["layer"], r["depth"], r["a%d" % a]["dJ"], r["a%d" % a]["dAUROC"]) for r in top)))
    topA = sorted(rows, key=lambda r: -r["a%d" % a]["dAUROC"])[:5]
    print("a%d top-5 by dAUROC: %s" % (a, ", ".join("L%d (%.2f) %+.3f" % (r["layer"], r["depth"], r["a%d" % a]["dAUROC"]) for r in topA)))
    thirds = [[r["a%d" % a]["dJ"] for r in rows if lo <= r["depth"] < hi] for lo, hi in ((0, 1/3), (1/3, 2/3), (2/3, 1.01))]
    print("a%d mean dJ by depth third (early/mid/late): %s" % (a, " / ".join("%+.3f" % np.mean(t) for t in thirds)))
    leak = [r for r in rows if abs(r["a%d" % a]["d_ans"]) > 0.03]
    print("a%d layers whose d ans exceeds 0.03: %s" % (a, ", ".join("L%d(%+.3f)" % (r["layer"], r["a%d" % a]["d_ans"]) for r in leak) or "none"))
json.dump(rows, open(d + "/e_per_layer_test.json", "w"), indent=1, default=float)

# ---------------- (b) C strict table ----------------
zc = np.load(d + "/knob_sweep_das_tfce_C.npz", allow_pickle=True)
assert np.array_equal(zc["ids"], ze["ids"]) and np.array_equal(zc["half_a"], ze["half_a"])
tune = zc["half_a"].astype(bool); tuidx = np.where(tune)[0]
pair = np.array([i.rsplit(":", 1)[0] for i in zc["ids"]]); rng = np.random.default_rng(0)
tp = np.array(sorted(set(pair[test]))); where = {p: np.where((pair == p) & test)[0] for p in tp}
boots = [np.concatenate([where[p] for p in rng.choice(tp, len(tp), replace=True)]) for _ in range(2000)]
baseb = [mets(zc, "baseline", b, labels) for b in boots]; basec = mets(zc, "baseline", tidx, labels)
conds = sorted({k.rsplit("__", 1)[0] for k in zc.files if k.endswith("__answer") and k != "baseline__answer"})
baset = mets(zc, "baseline", tuidx, labels)
# tune-selected: among conditions moving tune answer rate by >= 0.20 in the requested direction, minimise the TUNE |dAUROC| then |dJ|
sel = {}
for sgn, nm in ((-1, "abst"), (1, "ans")):
    cands = []
    for c in conds:
        m = mets(zc, c, tuidx, labels) - baset
        if sgn * m[0] >= 0.20: cands.append((abs(m[2]), abs(m[1]), c))
    sel[nm] = min(cands)[2] if cands else None
def describe(c):
    m = mets(zc, c, tidx, labels) - basec
    bd = np.array([mets(zc, c, b, labels) - bb for b, bb in zip(boots, baseb)])
    lo, hi = np.percentile(bd, 2.5, axis=0), np.percentile(bd, 97.5, axis=0)
    pb = zc["baseline__ptrue"][tidx] > 0.5; pc = zc[c + "__ptrue"][tidx] > 0.5
    flips = int((pb != pc).sum()); ab = zc["baseline__answer"][tidx]; ac = zc[c + "__answer"][tidx]
    return ("| %s | %+.3f [%+.3f, %+.3f] | %+.4f [%+.4f, %+.4f] | %+.4f [%+.4f, %+.4f] | %+.3f [%+.3f, %+.3f] | %d / %d | %d / %d |"
            % (c, m[0], lo[0], hi[0], m[1], lo[1], hi[1], m[2], lo[2], hi[2], m[3], lo[3], hi[3], flips, len(tidx), int((ab != ac).sum()), len(tidx)))
print("\n## %s  C: full situation on the test half (%d statements)" % (model, len(tidx)))
print("| condition | d answer rate [95% CI] | dJ [95% CI] | dAUROC [95% CI] | d bias B [95% CI] | True/False预测翻转 | Yes/No自报翻转 |")
print("|---|---|---|---|---|---|---|")
print("protocol-chosen (2x2 rule on tune):"); print(describe(c_abst)); print(describe(c_ans))
print("tune-selected min |dAUROC| among |d ans| >= 0.20 on tune:"); print(describe(sel["abst"])); print(describe(sel["ans"]))
strong = [c for c in conds if abs((mets(zc, c, tidx, labels) - basec)[0]) >= 0.20]
dA = np.array([abs((mets(zc, c, tidx, labels) - basec)[2]) for c in strong]); dJ = np.array([abs((mets(zc, c, tidx, labels) - basec)[1]) for c in strong])
print("all %d strong C conditions (|d ans| >= 0.20, test): |dAUROC| median %.4f, max %.4f; |dJ| median %.4f, max %.4f" % (len(strong), np.median(dA), dA.max(), np.median(dJ), dJ.max()))

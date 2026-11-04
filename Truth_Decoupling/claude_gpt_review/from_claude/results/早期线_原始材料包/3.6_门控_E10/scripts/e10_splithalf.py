#!/usr/bin/env python3
"""E10 post-hoc: split-half selection check and pair-clustered CIs for the gating conditions (no GPU; reads gating.npz).

Why: every gate in E10 was evaluated on the full test set, so picking the best one there carries selection optimism.
Here the pair hash (same sha1-parity rule as E8) splits the test questions into half A / half B; the gate is chosen on
half A by the penalised score (correct - wrong)/n and reported on half B. Pair-clustered bootstrap CIs are given for
the key contrasts (gated vs ungated, gated vs baseline) on the full set.

Usage: python3 scripts/e10_splithalf.py --npz results_tqa/E10/llama3.1-8b/gating.npz
Output: <npz dir>/gating_splithalf.{md,json}
"""
import argparse
import hashlib
import json
import os

import numpy as np


def auroc(score, y):
    score, y = np.asarray(score, dtype=np.float64), np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def state(pred, labels):
    pred = np.asarray(pred)
    ans = pred != "Abstain"
    corr = ans & ((pred == "True") == labels)
    return np.where(~ans, "abstain", np.where(corr, "known", "wrong"))


def metrics(labels, p, pred, base_pred, base_p, m):
    """Metrics restricted to mask m."""
    lab, pr, bp = labels[m], np.asarray(pred)[m], np.asarray(base_pred)[m]
    sb, sc = state(bp, lab), state(pr, lab)
    ans = pr != "Abstain"
    n_known = max(1, int((sb == "known").sum()))
    tpr = float(((pr == "True") & lab & ans).sum() / max(1, (lab & ans).sum()))
    fpr = float(((pr == "True") & ~lab & ans).sum() / max(1, (~lab & ans).sum()))
    return {"n": int(m.sum()), "corrected": int(((sb == "wrong") & (sc == "known")).sum()), "damaged": int(((sb == "known") & (sc == "wrong")).sum()),
            "known_damage_rate": float(((sb == "known") & (sc == "wrong")).sum() / n_known),
            "acc_all": float((sc == "known").mean()), "wrong_all": float((sc == "wrong").mean()), "abstain_all": float((sc == "abstain").mean()),
            "penalised": float(((sc == "known").sum() - (sc == "wrong").sum()) / len(lab)),
            "J": tpr - fpr, "B": float(((pr == "True") & ans).sum() / max(1, ans.sum())), "answer_rate": float(ans.mean()),
            "d_auroc": auroc(p[m], lab) - auroc(base_p[m], lab)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    labels = z["labels"].astype(bool)
    ids = [str(i) for i in z["ids"]]
    pair_ids = np.array([i.rsplit(":", 1)[0] for i in ids])
    half_a = np.array([int(hashlib.sha1(p.encode()).hexdigest(), 16) % 2 == 0 for p in pair_ids])
    conds = [k[:-3] for k in z.files if k.endswith("__p") and k != "baseline__p"]
    base_p, base_pred = z["baseline__p"], z["baseline__pred"]
    masks = {"ALL": np.ones(len(labels), dtype=bool), "halfA": half_a, "halfB": ~half_a}
    res = {c: {g: metrics(labels, z[c + "__p"], z[c + "__pred"], base_pred, base_p, m) for g, m in masks.items()} for c in ["baseline"] + conds}

    # selection on half A, report on half B
    sel = {}
    for crit in ("penalised", "J", "acc_all"):
        best = max(conds, key=lambda c: res[c]["halfA"][crit])
        sel[crit] = {"chosen": best, "halfA": res[best]["halfA"][crit], "halfB": res[best]["halfB"][crit], "halfB_baseline": res["baseline"]["halfB"][crit],
                     "halfB_global_a3": res["global_a3"]["halfB"][crit] if "global_a3" in res else None}

    # pair-clustered bootstrap for key contrasts on the full set
    rng = np.random.default_rng(args.seed)
    uniq, inv = np.unique(pair_ids, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    contrasts = [(a, b) for a, b in (("disagree_a5", "global_a3"), ("disagree_a3", "global_a3"), ("transfer_disagree_g2", "transfer_global_g2"),
                                     ("probeconf_t1_a3", "baseline"), ("disagree_a5", "baseline"), ("disagree_a5", "prompt_only_cautious"),
                                     ("soft_unc_a8", "global_a3"), ("hard_unc_t0.5_a3", "global_a3")) if a in res and b in res]
    boot = {}
    for a, b in contrasts:
        sa, sb_ = state(z[a + "__pred"], labels), state(z[b + "__pred"], labels)
        s0 = state(base_pred, labels)
        pen_a, pen_b = (sa == "known").astype(float) - (sa == "wrong"), (sb_ == "known").astype(float) - (sb_ == "wrong")
        acc_a, acc_b = (sa == "known").astype(float), (sb_ == "known").astype(float)
        dmg_a, dmg_b = ((s0 == "known") & (sa == "wrong")).astype(float), ((s0 == "known") & (sb_ == "wrong")).astype(float)
        d_pen, d_acc, d_dmg = [], [], []
        for _ in range(args.n_boot):
            gi = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[g] for g in gi])
            d_pen.append(pen_a[idx].mean() - pen_b[idx].mean())
            d_acc.append(acc_a[idx].mean() - acc_b[idx].mean())
            d_dmg.append(dmg_a[idx].sum() / max(1, (s0[idx] == "known").sum()) - dmg_b[idx].sum() / max(1, (s0[idx] == "known").sum()))
        q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
        boot["%s vs %s" % (a, b)] = {"d_penalised": float(pen_a.mean() - pen_b.mean()), "ci_penalised": q(d_pen),
                                     "d_acc_all": float(acc_a.mean() - acc_b.mean()), "ci_acc_all": q(d_acc),
                                     "d_known_damage_rate": float(dmg_a.sum() / max(1, (s0 == "known").sum()) - dmg_b.sum() / max(1, (s0 == "known").sum())), "ci_known_damage_rate": q(d_dmg)}

    out_dir = os.path.dirname(os.path.abspath(args.npz))
    stem = os.path.join(out_dir, os.path.basename(args.npz)[:-4] + "_splithalf")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"n": int(len(labels)), "n_halfA": int(half_a.sum()), "n_halfB": int((~half_a).sum()), "selection": sel, "bootstrap": boot, "per_condition": res}, f, indent=1, ensure_ascii=False)
    lines = ["# E10 split-half check (%s; n=%d, half A %d / half B %d items, split by pair hash as in E8)" % (os.path.basename(args.npz), len(labels), half_a.sum(), (~half_a).sum()), "",
             "## Gate chosen on half A, reported on half B", "", "| criterion | chosen on A | value on A | value on B | baseline on B | ungated global_a3 on B |", "|---|---|---|---|---|---|"]
    for crit, s in sel.items():
        lines.append("| %s | %s | %+.3f | %+.3f | %+.3f | %+.3f |" % (crit, s["chosen"], s["halfA"], s["halfB"], s["halfB_baseline"], s["halfB_global_a3"]))
    lines += ["", "## Half B only (no selection), all conditions", "", "| condition | fixed | damaged | known dmg % | acc all | wrong all | abstain all | penalised | J | B | answer rate | dAUROC |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in ["baseline"] + conds:
        r = res[c]["halfB"]
        lines.append("| %s | %d | %d | %.1f | %.3f | %.3f | %.3f | %+.3f | %.3f | %.3f | %.3f | %+.3f |" % (c, r["corrected"], r["damaged"], 100 * r["known_damage_rate"], r["acc_all"], r["wrong_all"], r["abstain_all"], r["penalised"], r["J"], r["B"], r["answer_rate"], r["d_auroc"]))
    lines += ["", "## Pair-clustered bootstrap (full set, %d resamples of questions)" % args.n_boot, "", "| contrast | d penalised [95% CI] | d acc all [95% CI] | d known damage rate [95% CI] |", "|---|---|---|---|"]
    for k, b in boot.items():
        lines.append("| %s | %+.3f [%+.3f, %+.3f] | %+.3f [%+.3f, %+.3f] | %+.3f [%+.3f, %+.3f] |" % (k, b["d_penalised"], *b["ci_penalised"], b["d_acc_all"], *b["ci_acc_all"], b["d_known_damage_rate"], *b["ci_known_damage_rate"]))
    lines += ["", "## Full-set sanity (should match gating.md)", "", "| condition | fixed | damaged | known dmg % | acc all | penalised | J | dAUROC |", "|---|---|---|---|---|---|---|---|"]
    for c in ["baseline", "global_a3", "disagree_a5", "probeconf_t1_a3", "transfer_disagree_g2"]:
        if c in res:
            r = res[c]["ALL"]
            lines.append("| %s | %d | %d | %.1f | %.3f | %+.3f | %.3f | %+.3f |" % (c, r["corrected"], r["damaged"], 100 * r["known_damage_rate"], r["acc_all"], r["penalised"], r["J"], r["d_auroc"]))
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()

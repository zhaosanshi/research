#!/usr/bin/env python3
"""Pre-declared analysis of E45 (from_claude/40 section 3), written before any E45 result existed (Claude, 2026-11-12).

usage: e45_analysis.py E45_JSON --out OUT_PREFIX [--n-boot 2000] [--seed 0]
Reads the E45 json + npz (same stem). Selection only on the tuning statements and only on answer COUNTS (no correctness
label); every comparison is reported on the test statements. Pair bootstrap = resample whole true/false pairs of the test
half with replacement, recomputing every K and every top-K selection inside each resample (as e41_analysis.py).

natural(X) = statements condition X answers (three-way arg-max is True or False, not Unknown); K = their count; correct =
the True/False judgement among them is right. Comparisons (positive = the combination is better):
  P1  / P2   E+C natural  vs  E alone, thresholded to the same K on its gate (answer-vs-Unknown margin) / on |margin|
  P5  / P5b  E+C natural  vs  C alone (same C, same strength), thresholded the same way
Matched point (per C variant): the C side opposite to E's own answer-rate change on the tuning half (E raises the answer
rate -> push C toward abstention), and the C strength whose E+C answer count on the tuning half is closest to the
baseline's; ties -> the smaller strength. Primary family per model and C variant: P1, P2, P5, P5b at the matched point,
Holm-adjusted. Strong form ("E+C beats any single direction at the same answer rate") at a model: P2 and P5b both
positive with Holm p < 0.05; weak form: P5b positive with Holm p < 0.05 and P2 not significantly negative.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.metrics import auroc  # noqa: E402


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); m = len(ps); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i]); adj[i] = min(1.0, running)
    return adj


def boot_p(diffs):
    """two-sided bootstrap p: 2 * min(P(diff <= 0), P(diff >= 0)), floored at 1/n_boot"""
    d = np.asarray(diffs); n = len(d)
    return float(min(1.0, max(1.0 / n, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def topk_correct(score, correct, idx, k):
    """correct count of the k highest-scoring statements among the (possibly repeated) resampled index array idx"""
    if k <= 0:
        return 0
    order = np.argsort(-score[idx], kind="stable")
    return int(correct[idx][order[:k]].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    J = json.load(open(args.json)); Z = np.load(os.path.splitext(args.json)[0] + ".npz", allow_pickle=True)
    if not J.get("completed"):
        raise SystemExit("E45 run not completed")
    labels = Z["labels"].astype(bool); tune = Z["half_a"].astype(bool); test = ~tune
    conds = [c for c in J["conditions"] if c + "__ptrue" in Z.files]
    if len(conds) != len(J["conditions"]):
        raise SystemExit("npz misses conditions")
    A = {}
    for c in conds:
        a = {"ans": Z[c + "__answer"].astype(bool), "gate": Z[c + "__gate"], "p": Z[c + "__ptrue"], "mg": Z[c + "__margin"]}
        a["correct"] = (a["p"] > 0.5) == labels
        a["conf"] = np.abs(a["mg"])
        if c + "__lpT" in Z.files:   # probability mass of the three label sets (red team 40 item 3)
            lp3 = np.stack([Z[c + "__lpT"], Z[c + "__lpF"], Z[c + "__lpU"]])
            mx = lp3.max(0)
            a["label_mass_median"] = float(np.median(np.exp(mx + np.log(np.exp(lp3 - mx).sum(0)))))
        else:
            a["label_mass_median"] = float("nan")
        A[c] = a
    pair = np.array([i.rsplit(":", 1)[0] for i in Z["ids"]])
    rng = np.random.default_rng(args.seed)
    test_pairs = np.array(sorted(set(pair[test]))); where = {p: np.where((pair == p) & test)[0] for p in test_pairs}
    boots = [np.concatenate([where[p] for p in rng.choice(test_pairs, len(test_pairs), replace=True)]) for _ in range(args.n_boot)]
    test_idx = np.where(test)[0]; tune_idx = np.where(tune)[0]

    def K(c, idx):
        return int(A[c]["ans"][idx].sum())

    def summary(c, idx):
        a = A[c]; pred = a["p"][idx] > 0.5; lab = labels[idx]; yes = a["ans"][idx]
        return {"ans": float(yes.mean()), "J": float(pred[lab].mean() - pred[~lab].mean()), "AUROC": float(auroc(a["p"][idx], lab)),
                "B": float(pred.mean()), "K": int(yes.sum()), "correct_natural": int(a["correct"][idx][yes].sum()),
                "acc_answered": float(a["correct"][idx][yes].mean()) if yes.any() else float("nan"),
                "label_mass_median_all": a["label_mass_median"]}

    def v2(x, y, idx, score):
        k = K(x, idx)
        return int(A[x]["correct"][idx][A[x]["ans"][idx]].sum()) - topk_correct(A[y][score], A[y]["correct"], idx, k), k

    def contrast(x, y, score):
        d, k = v2(x, y, test_idx, score)
        bd = np.array([v2(x, y, b, score)[0] for b in boots])
        return {"diff": d, "K": k, "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))], "boot_p": boot_p(bd)}

    def delta(x, y):
        sx, sy = summary(x, test_idx), summary(y, test_idx); out = {}
        for key in ("ans", "J", "AUROC", "acc_answered"):
            bd = np.array([summary(x, b)[key] - summary(y, b)[key] for b in boots])
            out[key] = {"diff": sx[key] - sy[key], "ci95": [float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))]}
        return out

    E = "E_a%g" % J["e_alpha"]
    base_unknown_tune = 1.0 - summary("baseline", tune_idx)["ans"]
    R = {"model": J["model"], "E": E, "n_test": int(test.sum()), "n_tune": int(tune.sum()), "n_boot": args.n_boot,
         "baseline_unknown_share_tune": base_unknown_tune, "ceiling_flag": bool(base_unknown_tune < 0.05),
         "E_diagnostics": J.get("E_diagnostics"), "E_alone": {}, "variants": {}}
    for e in [c for c in conds if c.startswith("E_a") and "+" not in c]:
        R["E_alone"][e] = {"tune_dK": K(e, tune_idx) - K("baseline", tune_idx), "test_vs_baseline": delta(e, "baseline")}
    dK_E = K(E, tune_idx) - K("baseline", tune_idx)
    side = "abst" if dK_E >= 0 else "ans"
    for variant in ("C", "CpI"):
        labs = [k for k, v in J["C"].items() if v["variant"] == variant]
        lab = [k for k in labs if k.endswith("_" + side)]
        if len(lab) != 1:
            raise SystemExit("cannot find the %s %s C" % (variant, side))
        lab = lab[0]
        grid = []
        for a in J["alphas_C"]:
            ec = "%s+%s_a%g" % (E, lab, a)
            grid.append((abs(K(ec, tune_idx) - K("baseline", tune_idx)), a, ec, "%s_a%g" % (lab, a)))
        grid.sort(key=lambda g: (g[0], g[1]))
        dist, a_m, ec, c = grid[0]
        prim = {"P1 vs E@gate": contrast(ec, E, "gate"), "P2 vs E@conf": contrast(ec, E, "conf"),
                "P5 vs C@gate": contrast(ec, c, "gate"), "P5b vs C@conf": contrast(ec, c, "conf")}
        for (k, v), adj in zip(prim.items(), holm([v["boot_p"] for v in prim.values()])):
            v["holm_p"] = float(adj)
        sig = lambda v, s: v["holm_p"] < 0.05 and np.sign(v["diff"]) == s  # noqa: E731
        strong = sig(prim["P2 vs E@conf"], 1) and sig(prim["P5b vs C@conf"], 1)
        weak = sig(prim["P5b vs C@conf"], 1) and not sig(prim["P2 vs E@conf"], -1)
        # applicability (red team 40 item 1, fixed before any full result): the comparison is only about C restoring the
        # answer rate when there is something to restore and C actually restores it
        kb, ke, kec, kc = K("baseline", tune_idx), K(E, tune_idx), K(ec, tune_idx), K(c, tune_idx)
        recovery = (ke - kec) / (ke - kb) if ke != kb else 0.0
        e2 = "%s+%s_a%g" % (E, lab, max(J["alphas_C"]))
        recovery_at_max = (ke - K(e2, tune_idx)) / (ke - kb) if ke != kb else 0.0
        reasons = []
        if R["ceiling_flag"]:
            reasons.append("ceiling: baseline Unknown share on the tuning half %.1f%% < 5%%" % (100 * base_unknown_tune))
        if abs(ke - kb) < 0.03 * len(tune_idx):
            reasons.append("E changes the tuning answer count by %+d (< 3%% of %d)" % (ke - kb, len(tune_idx)))
        if recovery < 0.5:
            reasons.append("the matched C takes back only %.0f%% of E's change" % (100 * recovery))
        if (side == "abst" and not kc < kb) or (side == "ans" and not kc > kb):
            reasons.append("C alone does not move the tuning answer count toward %s (%d vs baseline %d)" % (side, kc, kb))
        low = [x for x in (E, c, ec) if A[x]["label_mass_median"] < 0.5]
        if low:
            reasons.append("label probability mass below 0.5 (median) in %s" % ", ".join(low))
        applicable = not reasons
        curve = []
        for s in ("abst", "ans"):
            l2 = [k for k in labs if k.endswith("_" + s)][0]
            for a in J["alphas_C"]:
                e2, c2 = "%s+%s_a%g" % (E, l2, a), "%s_a%g" % (l2, a)
                curve.append({"side": s, "alpha": a, "K_tune_EC": K(e2, tune_idx), "K_test_EC": K(e2, test_idx), "K_test_C": K(c2, test_idx),
                              "P2": v2(e2, E, test_idx, "conf")[0], "P5b": v2(e2, c2, test_idx, "conf")[0],
                              "dans_C_alone": summary(c2, test_idx)["ans"] - summary("baseline", test_idx)["ans"]})
        R["variants"][variant] = {
            "C": lab, "side": side, "E_tune_dK": dK_E, "matched_alpha": a_m, "matched_tune_K_distance": dist,
            "matched_EC": ec, "K_tune": {"baseline": K("baseline", tune_idx), "E": K(E, tune_idx), "EC": K(ec, tune_idx)},
            "primary": prim, "applicable": bool(applicable), "not_applicable_because": reasons,
            "recovery_tune": recovery, "recovery_tune_at_max_strength": recovery_at_max,
            "strong_form": bool(strong) if applicable else None, "weak_form": bool(weak) if applicable else None,
            "label_mass_median": {x: A[x]["label_mass_median"] for x in ("baseline", E, c, ec)},
            "secondary": {"EC vs baseline@conf": contrast(ec, "baseline", "conf"), "C natural vs baseline@conf (P3b)": contrast(c, "baseline", "conf"),
                          "E natural vs baseline@conf (P4b)": contrast(E, "baseline", "conf"), "EC minus E": delta(ec, E), "EC minus C": delta(ec, c),
                          "EC vs baseline": delta(ec, "baseline")},
            "summaries_test": {x: summary(x, test_idx) for x in ("baseline", E, c, ec)},
            "curve": curve}
    # designated C (red team 40 item 2, fixed before any full result): the 2x2's C decides; CpI only if the 2x2's C cannot
    # take back half of E's tuning answer-count change even at the largest strength
    vC = R["variants"]["C"]
    designated = "C" if vC["recovery_tune_at_max_strength"] >= 0.5 else "CpI"
    dv = R["variants"][designated]
    R["designated_variant"] = designated
    R["verdict"] = ("n/a: " + "; ".join(dv["not_applicable_because"])) if not dv["applicable"] else (
        "strong form" if dv["strong_form"] else ("weak form" if dv["weak_form"] else "neither form"))
    json.dump(R, open(args.out + ".json", "w"), indent=1)
    L = ["# E45 analysis %s (test %d statements; matched on the tuning half by answer counts only)" % (R["model"], R["n_test"]), "",
         "Baseline Unknown share on the tuning half: %.1f%%%s" % (100 * base_unknown_tune, " -> CEILING: few Unknowns, same-K comparisons degenerate" if R["ceiling_flag"] else ""), "",
         "## E alone under the Unknown-allowed prompt (test, vs baseline)", "", "| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |", "|---|---|---|---|---|"]
    for e, v in R["E_alone"].items():
        t = v["test_vs_baseline"]
        L.append("| %s | %+.3f [%+.3f, %+.3f] | %+.3f [%+.3f, %+.3f] | %+.4f [%+.4f, %+.4f] | %+.3f [%+.3f, %+.3f] |" % (
            e, t["ans"]["diff"], *t["ans"]["ci95"], t["J"]["diff"], *t["J"]["ci95"], t["AUROC"]["diff"], *t["AUROC"]["ci95"], t["acc_answered"]["diff"], *t["acc_answered"]["ci95"]))
    for variant, v in R["variants"].items():
        L += ["", "## C variant %s: %s (side %s, chosen because E changes the tuning answer count by %+d); matched strength %g (tuning K: baseline %d, E %d, E+C %d)" % (
            variant, v["C"], v["side"], v["E_tune_dK"], v["matched_alpha"], v["K_tune"]["baseline"], v["K_tune"]["E"], v["K_tune"]["EC"]), "",
              "| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |", "|---|---|---|---|---|"]
        for k, c in v["primary"].items():
            L.append("| %s | %+d | [%+.0f, %+.0f] | %d | %.4f |" % (k, c["diff"], *c["ci95"], c["K"], c["holm_p"]))
        L.append("")
        L.append("applicable: %s%s; recovery of E's tuning answer-count change: %.0f%% (at the largest C strength %.0f%%); strong form: %s; weak form: %s; label mass medians %s" % (
            v["applicable"], "" if v["applicable"] else " (" + "; ".join(v["not_applicable_because"]) + ")", 100 * v["recovery_tune"],
            100 * v["recovery_tune_at_max_strength"], v["strong_form"], v["weak_form"],
            ", ".join("%s %.3f" % (k, m) for k, m in v["label_mass_median"].items())))
        L += ["", "| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |", "|---|---|---|---|---|---|---|---|"]
        for r in v["curve"]:
            L.append("| %s | %g | %d | %d | %d | %+d | %+d | %+.3f |" % (r["side"], r["alpha"], r["K_tune_EC"], r["K_test_EC"], r["K_test_C"], r["P2"], r["P5b"], r["dans_C_alone"]))
    L += ["", "## Pre-registered reading (from_claude/40 section 2.1)", "",
          "designated C variant: %s; verdict: %s" % (R["designated_variant"], R["verdict"])]
    open(args.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

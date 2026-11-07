#!/usr/bin/env python3
"""Pre-declared analysis of E41 (fixed 2x2 C x E at every layer), written before any E41 result existed (Claude, 2026-11-07).

usage: e41_analysis.py E41_JSON --chosen-e E_L11_a3 --out OUT_PREFIX [--n-boot 2000] [--seed 0]
Reads the E41 json + npz (same stem). Test statements only, except where a tuning selection is stated. Pair bootstrap =
resample whole true/false pairs of the test half with replacement, recomputing every selection inside each resample.

Views (all per fixed C sign s in {abst, ans}):
  V1 additivity   : do the two knobs compose? (E+C_s) - E on answer rate / J / AUROC / B, and (E+C_s) - C_s. C's answer-rate
                    shift should survive E and E's J gain should survive C.
  V2 same number of self-selected answers (the fair "same answer rate" test of a threshold-type C):
                    natural(X) = statements where pass A says Yes; K = their count; correct = pass-B right among them.
                    Compare E+C_s natural against E with the SAME K reached by a threshold on E's own gate margin (P1) or on
                    E's True/False confidence |margin| (P2); C_s natural against baseline @gate K (P3); E natural against
                    baseline @gate K (P4). Positive = the steered selection beats the thresholded one.
  V3 exact quota by True/False confidence at 25/50/75 % of the test statements (the E26 / round-20 view).
Primary family (Holm within model): P1 and P2 at the pre-chosen E (the 2x2's E), both C signs -> 4 comparisons. Every
other layer is exploratory; the tuning-selected best layer is reported on test and labelled as such.
"""
import argparse
import json
import math
import os
import sys

import numpy as np


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
    ap.add_argument("--chosen-e", required=True, help="the 2x2's E condition for this model, e.g. E_L11_a3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    J = json.load(open(args.json)); Z = np.load(os.path.splitext(args.json)[0] + ".npz", allow_pickle=True)
    if not J.get("completed"):
        print("WARNING: run not marked completed; analysing what exists", file=sys.stderr)
    labels = Z["labels"].astype(bool); test = ~Z["half_a"].astype(bool); tune = Z["half_a"].astype(bool)
    conds = [c for c in J["conditions"] if c + "__ptrue" in Z.files]
    c_names = list(J["fixed_C"].keys()); c_abst = [c for c in c_names if J["fixed_C"][c]["sign"] == -1][0]; c_ans = [c for c in c_names if J["fixed_C"][c]["sign"] == 1][0]
    e_names = [c for c in conds if c.startswith("E_") and "+" not in c]
    if args.chosen_e not in e_names:
        raise SystemExit("--chosen-e %s not in this run" % args.chosen_e)
    A = {c: {"ans": Z[c + "__answer"].astype(bool), "gate": Z[c + "__gate"], "p": Z[c + "__ptrue"], "mg": Z[c + "__margin"]} for c in conds}
    for c in conds:
        A[c]["correct"] = (A[c]["p"] > 0.5) == labels
        A[c]["conf"] = np.abs(A[c]["mg"])
        if not np.array_equal(A[c]["ans"], A[c]["gate"] > 0):
            print("note: %s pass-A Yes differs from gate > 0 on %d statements (exact ties)" % (c, int((A[c]["ans"] != (A[c]["gate"] > 0)).sum())), file=sys.stderr)
    pair = np.array([i.rsplit(":", 1)[0] for i in Z["ids"]])
    rng = np.random.default_rng(args.seed)
    test_pairs = np.array(sorted(set(pair[test]))); where = {p: np.where((pair == p) & test)[0] for p in test_pairs}
    boots = [np.concatenate([where[p] for p in rng.choice(test_pairs, len(test_pairs), replace=True)]) for _ in range(args.n_boot)]
    test_idx = np.where(test)[0]; tune_idx = np.where(tune)[0]

    def summary(c, idx):
        a = A[c]; pred = a["p"][idx] > 0.5; lab = labels[idx]; yes = a["ans"][idx]
        tpr = pred[lab].mean(); fpr = pred[~lab].mean()
        from core.metrics import auroc
        return {"ans": float(yes.mean()), "J": float(tpr - fpr), "AUROC": float(auroc(a["p"][idx], lab)), "B": float(pred.mean()),
                "K": int(yes.sum()), "correct_natural": int(a["correct"][idx][yes].sum())}

    def v2(x, y, idx, score):
        """correct(natural x) - correct(y thresholded on `score` at x's K), on index array idx"""
        k = int(A[x]["ans"][idx].sum())
        return int(A[x]["correct"][idx][A[x]["ans"][idx]].sum()) - topk_correct(A[y][score], A[y]["correct"], idx, k), k

    def contrast(x, y, score, idx=test_idx):
        d, k = v2(x, y, idx, score)
        bd = np.array([v2(x, y, b, score)[0] for b in boots])
        return {"diff": d, "K": k, "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))], "boot_p": boot_p(bd)}

    def delta(x, y, idx=test_idx):
        sx, sy = summary(x, idx), summary(y, idx); out = {}
        for key in ("ans", "J", "AUROC", "B"):
            bd = np.array([summary(x, b)[key] - summary(y, b)[key] for b in boots])
            out[key] = {"diff": sx[key] - sy[key], "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
        return out

    E = args.chosen_e; R = {"model": J["model"], "e_estimator": J["e_estimator"], "chosen_E": E, "fixed_C": J["fixed_C"], "n_test": int(test.sum()), "n_boot": args.n_boot,
                            "primary": {}, "secondary": {}, "exploratory": {}, "quota": {}}
    # ---- primary family: P1, P2 at the chosen E, both C signs; Holm within model ------------------------------------- #
    prim = {}
    for c in (c_abst, c_ans):
        ec = "%s+%s" % (E, c)
        prim["P1 %s vs E@gate" % ec] = contrast(ec, E, "gate")
        prim["P2 %s vs E@conf" % ec] = contrast(ec, E, "conf")
    adj = holm([v["boot_p"] for v in prim.values()])
    for (k, v), a in zip(prim.items(), adj):
        v["holm_p"] = float(a)
    R["primary"] = prim
    # ---- secondary: P3, P4, additivity, baseline/E/C summaries ------------------------------------------------------- #
    sec = {"P4 E vs baseline@gate": contrast(E, "baseline", "gate"), "P4b E vs baseline@conf": contrast(E, "baseline", "conf")}
    for c in (c_abst, c_ans):
        sec["P3 %s vs baseline@gate" % c] = contrast(c, "baseline", "gate")
        sec["P3b %s vs baseline@conf" % c] = contrast(c, "baseline", "conf")
        ec = "%s+%s" % (E, c)
        sec["additivity %s minus E" % ec] = delta(ec, E); sec["additivity %s minus %s" % (ec, c)] = delta(ec, c)
        sec["%s vs baseline@gate" % ec] = contrast(ec, "baseline", "gate"); sec["%s vs baseline@conf" % ec] = contrast(ec, "baseline", "conf")
    sec["summaries"] = {c: summary(c, test_idx) for c in ["baseline", E, c_abst, c_ans, "%s+%s" % (E, c_abst), "%s+%s" % (E, c_ans)]}
    R["secondary"] = sec
    # ---- exploratory: every layer / alpha (point estimates on test; tuning-selected best reported on test) ------------ #
    rows = []
    for e in e_names:
        b = summary(e, test_idx); row = {"E": e, "E_dJ": b["J"] - summary("baseline", test_idx)["J"], "E_dans": b["ans"] - summary("baseline", test_idx)["ans"]}
        for tag, c in (("abst", c_abst), ("ans", c_ans)):
            ec = "%s+%s" % (e, c)
            if ec not in A:
                continue
            s = summary(ec, test_idx)
            row[tag] = {"dans_vs_E": s["ans"] - b["ans"], "dJ_vs_E": s["J"] - b["J"], "dAUROC_vs_E": s["AUROC"] - b["AUROC"], "K": s["K"],
                        "P1": v2(ec, e, test_idx, "gate")[0], "P2": v2(ec, e, test_idx, "conf")[0],
                        "P2_tune": v2(ec, e, tune_idx, "conf")[0], "vs_baseline_conf": v2(ec, "baseline", test_idx, "conf")[0]}
        rows.append(row)
    R["exploratory"]["per_layer"] = rows
    for tag in ("abst", "ans"):
        have = [r for r in rows if tag in r]
        R["exploratory"]["n_layers_P2_positive_%s" % tag] = sum(1 for r in have if r[tag]["P2"] > 0)
        R["exploratory"]["n_layers_P2_negative_%s" % tag] = sum(1 for r in have if r[tag]["P2"] < 0)
        best = max(have, key=lambda r: (r[tag]["P2_tune"], -abs(float(r["E"].split("_a")[1])))) if have else None
        if best:
            ec = "%s+%s" % (best["E"], c_abst if tag == "abst" else c_ans)
            R["exploratory"]["tune_selected_best_%s" % tag] = {"E": best["E"], "P2_tune": best[tag]["P2_tune"], "test": contrast(ec, best["E"], "conf")}
    # ---- V3: exact quota by True/False confidence at 25 / 50 / 75 % of the test statements -------------------------- #
    n = len(test_idx); q = {}
    for frac in (0.25, 0.5, 0.75):
        k = int(math.floor(frac * n)); row = {}
        for c in ["baseline", E, c_abst, c_ans, "%s+%s" % (E, c_abst), "%s+%s" % (E, c_ans)]:
            row[c] = topk_correct(A[c]["conf"], A[c]["correct"], test_idx, k)
        for tag, c in (("abst", c_abst), ("ans", c_ans)):
            ec = "%s+%s" % (E, c); bd = np.array([topk_correct(A[ec]["conf"], A[ec]["correct"], b, int(math.floor(frac * len(b)))) -
                                                 topk_correct(A[E]["conf"], A[E]["correct"], b, int(math.floor(frac * len(b)))) for b in boots])
            row["EC_%s_minus_E" % tag] = {"diff": row[ec] - row[E], "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))], "boot_p": boot_p(bd)}
        q["%d%%" % int(frac * 100)] = {"k": k, **row}
    R["quota"] = q
    json.dump(R, open(args.out + ".json", "w"), indent=1, ensure_ascii=False)
    # ---- markdown ------------------------------------------------------------------------------------------------------ #
    L = ["# E41 analysis: %s (%s E), chosen E %s, fixed C %s / %s; %d test statements, %d pair-bootstrap resamples" % (
        J["model"], J["e_estimator"], E, c_abst, c_ans, int(test.sum()), args.n_boot), "",
        "## Test-half summaries", "", "| condition | answer rate | J | AUROC | B | K answered | correct among answered |", "|---|---|---|---|---|---|---|"]
    for c, s in sec["summaries"].items():
        L.append("| %s | %.3f | %+.3f | %.3f | %.3f | %d | %d |" % (c, s["ans"], s["J"], s["AUROC"], s["B"], s["K"], s["correct_natural"]))
    L += ["", "## Primary (pre-declared): E+C natural vs E thresholded to the same K (positive = C's selection beats a threshold)", "",
          "| contrast | K | diff (correct) | 95% CI | bootstrap p | Holm p |", "|---|---|---|---|---|---|"]
    for k, v in prim.items():
        L.append("| %s | %d | %+d | [%+.1f, %+.1f] | %.4f | %.4f |" % (k, v["K"], v["diff"], v["ci95"][0], v["ci95"][1], v["boot_p"], v["holm_p"]))
    L += ["", "## Secondary", "", "| contrast | K | diff | 95% CI | bootstrap p |", "|---|---|---|---|---|"]
    for k, v in sec.items():
        if isinstance(v, dict) and "K" in v:
            L.append("| %s | %d | %+d | [%+.1f, %+.1f] | %.4f |" % (k, v["K"], v["diff"], v["ci95"][0], v["ci95"][1], v["boot_p"]))
    L += ["", "### Additivity at the chosen E (test; diff with 95% CI)", "", "| pair | d answer rate | d J | d AUROC | d B |", "|---|---|---|---|---|"]
    for k, v in sec.items():
        if k.startswith("additivity"):
            L.append("| %s | " % k + " | ".join("%+.3f [%+.3f, %+.3f]" % (v[m]["diff"], v[m]["ci95"][0], v[m]["ci95"][1]) for m in ("ans", "J", "AUROC", "B")) + " |")
    L += ["", "## Exploratory: every E layer / alpha (test point estimates; P1 = vs E@gate, P2 = vs E@conf at E+C's own K)", "",
          "| E | E dJ | E d ans | +C_abst: d ans / dJ vs E | K | P1 | P2 | +C_ans: d ans / dJ vs E | K | P1 | P2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = ["%s" % r["E"], "%+.3f" % r["E_dJ"], "%+.3f" % r["E_dans"]]
        for tag in ("abst", "ans"):
            if tag in r:
                t = r[tag]; cells += ["%+.3f / %+.3f" % (t["dans_vs_E"], t["dJ_vs_E"]), "%d" % t["K"], "%+d" % t["P1"], "%+d" % t["P2"]]
            else:
                cells += ["—"] * 4
        L.append("| " + " | ".join(cells) + " |")
    for tag in ("abst", "ans"):
        L.append("")
        L.append("C_%s: P2 > 0 at %d layer/alpha points, < 0 at %d of %d." % (tag, R["exploratory"]["n_layers_P2_positive_%s" % tag], R["exploratory"]["n_layers_P2_negative_%s" % tag], len(rows)))
        tb = R["exploratory"].get("tune_selected_best_%s" % tag)
        if tb:
            L.append("Tuning-selected best for C_%s (max P2 on tune): %s, tune P2 %+d -> test P2 %+d [%+.1f, %+.1f], bootstrap p %.3f (exploratory, selected on seen data)." % (
                tag, tb["E"], tb["P2_tune"], tb["test"]["diff"], tb["test"]["ci95"][0], tb["test"]["ci95"][1], tb["test"]["boot_p"]))
    L += ["", "## V3: exact quota by True/False confidence (test; delivered correct)", "", "| quota | k | baseline | E | C_abst | C_ans | E+C_abst | E+C_ans | E+C_abst − E | E+C_ans − E |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in q.items():
        da, dn = row["EC_abst_minus_E"], row["EC_ans_minus_E"]
        L.append("| %s | %d | %d | %d | %d | %d | %d | %d | %+d [%+.1f, %+.1f] p %.3f | %+d [%+.1f, %+.1f] p %.3f |" % (
            name, row["k"], row["baseline"], row[E], row[c_abst], row[c_ans], row["%s+%s" % (E, c_abst)], row["%s+%s" % (E, c_ans)],
            da["diff"], da["ci95"][0], da["ci95"][1], da["boot_p"], dn["diff"], dn["ci95"][0], dn["ci95"][1], dn["boot_p"]))
    open(args.out + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()

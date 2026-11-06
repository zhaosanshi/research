#!/usr/bin/env python3
"""EXPLORATORY analysis of the Qwen E26 run with E27 generation adapters (Claude, 2026-11-06).

usage: e26_adapter_analysis.py RUN_DIR OUT_JSON
Free-generation QA only (adapters are QA-only arms). For every coverage and every adapter method
(das / md / random _generation_adapter) compare, on the delivered sets of the exact-quota table:
  A_E          vs baseline_commitment   (both ranked by their own commitment score)
  A_E_simple   vs baseline_simple       (both ranked by simple confidence)
  A_EC         vs A_E                   (does C add anything on top of the adapter)
  A_E_simple   vs das_E_simple          (adapter versus the untrained DAS direction)
Counts are given (i) as judged, with the report's own exact paired p, and (ii) with a delivered "correct" answer that
contains U+FFFD counted as wrong (the delivered sets do not depend on labels, so recounting inside them is valid), with
an exact two-sided sign test over questions. Holm within each comparison family. This run's adapters were chosen after
the E27 test result had been seen and the test questions were viewed before: exploratory, not confirmatory.
"""
import json
import sys
from math import comb
from pathlib import Path

run, out_path = Path(sys.argv[1]), sys.argv[2]
rep = json.loads((run / "report.json").read_text())
delivered = json.loads((run / "delivered_answers.json").read_text())["qa"]


def sign_p(up, down):
    n = up + down
    return None if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(up, down) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run_max = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run_max
    return adj


def correct_map(arm_rows, rule):
    out = {}
    for r in arm_rows:
        ok = r["delivered_outcome"] == "correct"
        if rule == "fffd" and ok and "�" in (r.get("output") or ""):
            ok = False
        out[r["id"]] = ok
    return out


methods = ["das_generation_adapter", "md_generation_adapter", "random_generation_adapter"]
pairs = []
for m in methods:
    pairs += [(m + "_E", "baseline_commitment"), (m + "_E_simple", "baseline_simple"), (m + "_EC", m + "_E"), (m + "_E_simple", "das_E_simple")]
families = {}
for cov in sorted(rep["tables"]["qa"], key=float):
    entry = rep["tables"]["qa"][cov]["quota"]["coverages"][0]
    arms, comps = entry["arms"], entry["comparisons"]
    for a, b in pairs:
        fam = "%s - %s" % (a.replace("das_generation_adapter", "A").replace("md_generation_adapter", "A").replace("random_generation_adapter", "A"), b.replace("das_generation_adapter", "A").replace("md_generation_adapter", "A").replace("random_generation_adapter", "A"))
        rec = {"coverage": float(cov), "a": a, "b": b}
        fa, fb = arms.get(a, {}).get("feasible"), arms.get(b, {}).get("feasible")
        if not (fa and fb) or cov not in delivered or a not in delivered[cov] or b not in delivered[cov]:
            rec.update({"note": "infeasible or missing", "p_judged": 1.0, "p_fffd": 1.0})
        else:
            hit = [c for c in comps if {c["arm_a"], c["arm_b"]} == {a, b}]
            ex = (hit[0].get("exact_fixed_deliveries") or {}) if hit else {}
            sign = 1 if hit and hit[0]["arm_a"] == a else -1
            rec.update({"k": entry["k"], "n": entry["n"], "correct_a": arms[a]["correct"], "correct_b": arms[b]["correct"],
                        "delta_judged": sign * ex.get("delta_correct_count", 0), "p_judged": ex.get("p_value", 1.0)})
            ca, cb = correct_map(delivered[cov][a], "fffd"), correct_map(delivered[cov][b], "fffd")
            up = sum(ca[k] and not cb[k] for k in ca); down = sum(cb[k] and not ca[k] for k in ca)
            rec.update({"correct_a_fffd": sum(ca.values()), "correct_b_fffd": sum(cb.values()), "delta_fffd": up - down,
                        "p_fffd": sign_p(up, down) if up + down else 1.0,
                        "fffd_in_delivered_correct_a": sum(r["delivered_outcome"] == "correct" and "�" in (r.get("output") or "") for r in delivered[cov][a])})
        families.setdefault(fam, []).append(rec)
for fam, recs in families.items():
    for key in ("judged", "fffd"):
        for r, h in zip(recs, holm([r["p_" + key] for r in recs])):
            r["p_holm_" + key] = h
json.dump({"exploratory": True, "families": families}, open(out_path, "w"), indent=1, ensure_ascii=False)
for fam, recs in families.items():
    print("\n" + fam)
    for r in recs:
        if "note" in r:
            print("  %.2f %-28s %s" % (r["coverage"], r["a"][:28], r["note"]))
            continue
        print("  %.2f %-28s k=%d | judged %d vs %d (%+d, p %.3g, holm %.3g) | U+FFFD rule %d vs %d (%+d, p %.3g, holm %.3g) | fffd-in-correct %d" % (
            r["coverage"], r["a"][:28], r["k"], r["correct_a"], r["correct_b"], r["delta_judged"], r["p_judged"], r["p_holm_judged"],
            r["correct_a_fffd"], r["correct_b_fffd"], r["delta_fffd"], r["p_fffd"], r["p_holm_fffd"], r["fffd_in_delivered_correct_a"]))
print("saved", out_path)

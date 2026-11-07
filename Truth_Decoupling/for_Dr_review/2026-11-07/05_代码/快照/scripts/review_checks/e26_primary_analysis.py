#!/usr/bin/env python3
"""E26 analysis following the pre-test plan in from_claude/13 section 4.
The analysis script was committed before QA grading completed; TF test outputs already existed.

usage: e26_primary_analysis.py OUT_JSON RUN_DIR [RUN_DIR ...]   (each RUN_DIR holds an E26 report.json)
Primary family: DAS E+C minus DAS E (both ranked by their own commitment score), delivered correct count at each exact
quota, for every model x task x coverage cell; exact two-sided group-swap test from the report (fixed deliveries);
Holm correction over the cells of the family. A cell where either arm is infeasible enters Holm with p = 1.
Secondary families (each Holm-corrected on its own): DAS E+C minus DAS C, DAS E_simple, DAS C_simple,
baseline_commitment, baseline_simple; and the analogous MD contrasts. The original script only emitted MD EC-vs-E;
the remaining MD contrasts are labeled a supplemental completion, not a new primary analysis.
Decision rule (pre-declared): "combination beats single direction" only if, in a majority of cells, the primary contrast
is Holm-significant and positive AND E+C is not significantly worse than any simple-confidence arm.
"""
import json
import sys
from pathlib import Path
from e26_independent_recount import audit_run, require

if len(sys.argv) < 4:
    raise SystemExit("usage: e26_primary_analysis.py OUT_JSON LLAMA_RUN QWEN_RUN (both pre-declared models required)")
out_path, run_dirs = sys.argv[1], [Path(p) for p in sys.argv[2:]]
models = [json.loads((rd / "report.json").read_text())["model"] for rd in run_dirs]
require(sorted(models) == ["llama3.1-8b", "qwen3-8b"], "primary family requires exactly one run for each pre-declared model")
for rd in run_dirs:
    require(not json.loads((rd / "bank.json").read_text()).get("generation_adapters"), "exploratory adapters cannot enter the primary family")
    audit_run(rd)  # independent raw-row checks; malformed results must never silently become p=1
FAMILIES = {"primary: das_EC - das_E": ("das_EC", "das_E"),
            "das_EC - das_C": ("das_EC", "das_C"), "das_EC - das_E_simple": ("das_EC", "das_E_simple"),
            "das_EC - das_C_simple": ("das_EC", "das_C_simple"), "das_EC - baseline_commitment": ("das_EC", "baseline_commitment"),
            "das_EC - baseline_simple": ("das_EC", "baseline_simple"), "md_EC - md_E": ("md_EC", "md_E"),
            "supplemental: md_EC - md_C": ("md_EC", "md_C"),
            "supplemental: md_EC - md_E_simple": ("md_EC", "md_E_simple"),
            "supplemental: md_EC - md_C_simple": ("md_EC", "md_C_simple"),
            "supplemental: md_EC - baseline_commitment": ("md_EC", "baseline_commitment"),
            "supplemental: md_EC - baseline_simple": ("md_EC", "baseline_simple")}
SHOW = ["baseline_commitment", "baseline_simple", "baseline_random", "das_E", "das_C", "das_EC", "das_E_simple", "das_C_simple",
        "md_E", "md_C", "md_EC"]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


cells, families = [], {f: [] for f in FAMILIES}
for rd in run_dirs:
    rep = json.loads((rd / "report.json").read_text())
    if not rep.get("completed") or rep.get("pilot"):
        raise SystemExit("%s: report incomplete or a pilot" % rd)
    for task in ("tf", "qa"):
        require(set(rep["tables"][task]) == {"0.25", "0.5", "0.75"}, "primary coverage family changed")
        for cov in sorted(rep["tables"][task], key=float):
            entry = rep["tables"][task][cov]["quota"]["coverages"][0]
            arms, comps = entry["arms"], entry["comparisons"]
            cell = {"model": rep["model"], "run": rd.name, "task": task, "coverage": float(cov), "n": entry["n"], "k": entry["k"],
                    "arms": {a: ({"correct": arms[a]["correct"], "answered": arms[a]["answered"],
                                  "accuracy_answered": arms[a]["accuracy_answered"]} if arms.get(a, {}).get("feasible") else
                                 {"infeasible": (arms.get(a) or {}).get("reason", "absent")}) for a in SHOW}}
            for fam, (a, b) in FAMILIES.items():
                hit = [c for c in comps if {c["arm_a"], c["arm_b"]} == {a, b}]
                rec = {"model": rep["model"], "task": task, "coverage": float(cov), "a": a, "b": b}
                feasible = arms.get(a, {}).get("feasible") and arms.get(b, {}).get("feasible")
                require(len(hit) <= 1, "duplicate comparison: " + fam)
                if feasible:
                    require(len(hit) == 1 and (hit[0].get("exact_fixed_deliveries") or {}).get("p_value") is not None,
                            "missing exact comparison for feasible arms: " + fam)
                ex = (hit[0].get("exact_fixed_deliveries") or {}) if hit else {}
                if hit and ex.get("p_value") is not None:
                    sign = 1 if hit[0]["arm_a"] == a else -1
                    acc = (hit[0].get("point_delta") or {}).get("accuracy_answered")
                    ci = ((hit[0].get("ci95") or {}).get("accuracy_answered")) or None
                    rec.update({"delta_correct": sign * ex["delta_correct_count"], "p": ex["p_value"],
                                "delta_accuracy": None if acc is None else sign * acc,
                                "ci95_accuracy": None if ci is None or None in ci else sorted([sign * ci[0], sign * ci[1]])})
                else:
                    rec.update({"delta_correct": None, "p": 1.0, "note": "infeasible; retained in the pre-declared Holm family with p=1"})
                families[fam].append(rec)
            cells.append(cell)

for fam, recs in families.items():
    for r, a in zip(recs, holm([r["p"] for r in recs])):
        r["p_holm"] = a
prim = families["primary: das_EC - das_E"]
sig_pos = [r for r in prim if r["delta_correct"] is not None and r["delta_correct"] > 0 and r["p_holm"] < 0.05]
worse_simple = [r for f in ("das_EC - das_E_simple", "das_EC - das_C_simple", "das_EC - baseline_simple")
                for r in families[f] if r["delta_correct"] is not None and r["delta_correct"] < 0 and r["p_holm"] < 0.05]
decision = {"cells": len(prim), "primary_holm_significant_positive": len(sig_pos),
            "ec_significantly_worse_than_a_simple_arm": len(worse_simple),
            "predeclared_ec_vs_e_rule_met": len(sig_pos) > len(prim) / 2 and not worse_simple,
            "combination_beats_single_direction": len(sig_pos) > len(prim) / 2 and not worse_simple,
            "interpretation": "Pre-declared EC-vs-E decision rule only; superiority to every single direction additionally requires EC-vs-C evidence. Non-significance is not equivalence."}
json.dump({"pre_declared_in": "from_claude/13 section 4", "cells": cells, "families": families, "decision": decision},
          open(out_path, "w"), indent=1, ensure_ascii=False)

for c in cells:
    print("%-11s %-2s %.2f k=%3d/%-4d | %s" % (c["model"], c["task"], c["coverage"], c["k"], c["n"], " ".join(
        "%s=%s" % (a.replace("baseline_", "b_"), (c["arms"][a]["correct"] if "correct" in c["arms"][a] else "inf")) for a in SHOW)))
for fam, recs in families.items():
    print("\n%s" % fam)
    for r in recs:
        print("  %-11s %-2s %.2f  delta %s  p %.3g  holm %.3g" % (r["model"], r["task"], r["coverage"],
              ("%+d" % r["delta_correct"]) if r["delta_correct"] is not None else "n/a", r["p"], r["p_holm"]))
print("\nDECISION:", decision)
print("saved", out_path)

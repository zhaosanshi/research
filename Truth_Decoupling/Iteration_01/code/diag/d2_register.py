# Iteration_01 analysis (Claude Opus 5.5, 2026-11-19), read-only, no model, no judge call.
# D2 registration (novelty/ours/2026-11-19_new_datasets_pilot_design.md section 7, end): for the ten cells (two tasks x
# five models) the prediction of the sign of the TEST gain of must-answer + E 2 over must-answer (Pro, the fixed 100),
# written before any test answer exists:
#   gap rule (October P3)   gap > +0.03 -> "+", gap < -0.03 -> "-", otherwise "0"; no estimate -> "not estimable"
#   reference predictor     the sign of the same gain on the tune split (Pro, 192), "about zero" = within +-2 net points
#                           per 100 questions (the same range as for the test gain)
# Reads Iteration_01/newsets/gap/<task>/<model>/tune_gap.json and the verified tune votes; refuses to run if any test
# generation or test judge input already exists. Prints the table for the design file and writes
# Iteration_01/newsets/gap/d2_predictions.json (never overwritten).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/d2_register.py
import glob
import json
import os
import sys

sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_votes as NV  # noqa: E402

NS = "Iteration_01/newsets"
CFG = "Iteration_01/code/configs/newsets/frozen_models_20261119.json"
TASKS = ("triviaqa", "popqa")
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
NEAR_ZERO = 2


def gap_sign(g):
    if g is None:
        return "not estimable"
    return "+" if g > 0.03 else ("-" if g < -0.03 else "0")


def gain_sign(per100):
    return "+" if per100 > NEAR_ZERO else ("-" if per100 < -NEAR_ZERO else "0")


def main():
    if glob.glob(NS + "/gen/*/*/test") or glob.glob(NS + "/judge/*/*/test_pro100"):
        raise SystemExit("test generations or test judge inputs exist already: the D2 predictions can no longer be registered blind")
    out = NS + "/gap/d2_predictions.json"
    if os.path.exists(out):
        raise SystemExit("%s exists" % out)
    with open(CFG, encoding="utf-8") as f:
        M = json.load(f)["models"]
    cells, lines = [], ["| task | model | gap (readout AUROC - confidence AUROC) | gap prediction | tune gain per 100 | reference prediction |", "|---|---|---|---|---|---|"]
    for task in TASKS:
        for name in ORDER:
            with open("%s/gap/%s/%s/tune_gap.json" % (NS, task, name), encoding="utf-8") as f:
                g = json.load(f)
            table, _, _ = NV.load_votes("%s/judge/%s/%s/tune" % (NS, task, name), "%s/gen/%s/%s/tune" % (NS, task, name))
            ec = "E_L%d_conf-w-p2" % M[name]["el"]
            qs = sorted(p for p, c in table if c == "baseline_conf" and table[(p, c)]["state"] != "?" and table.get((p, ec), {}).get("state", "?") != "?")
            v = lambda s: (s == "R") - (s == "W")  # noqa: E731
            gain = sum(v(table[(p, ec)]["state"]) - v(table[(p, "baseline_conf")]["state"]) for p in qs)
            per100 = 100.0 * gain / len(qs)
            cell = {"task": task, "model": name, "gap": g["gap"], "readout_auroc": g["readout_auroc"],
                    "confidence_auroc": g["confidence_auroc"], "gap_counts": g["counts"], "gap_prediction": gap_sign(g["gap"]),
                    "tune_gain": gain, "tune_questions": len(qs), "tune_gain_per100": per100, "reference_prediction": gain_sign(per100)}
            cells.append(cell)
            lines.append("| %s | %s | %s | %s | %+.1f (%+d of %d) | %s |" % (
                task, name, "not estimable" if g["gap"] is None else "%+.3f (%.3f - %.3f)" % (g["gap"], g["readout_auroc"], g["confidence_auroc"]),
                cell["gap_prediction"], per100, gain, len(qs), cell["reference_prediction"]))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"near_zero_net_points_per_100": NEAR_ZERO, "cells": cells}, f, indent=1, ensure_ascii=False)
    print("\n".join(lines))
    print("\nwritten", out)


if __name__ == "__main__":
    main()

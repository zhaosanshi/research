#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Opus 5.5, 2026-11-28). Read-only: existing answers and Pro verdicts of
# FalseQA held-out 600; no model, no judge call. Not reviewed by GPT. Zhao Lei 01:46 asked whether Qwen3.5-9B on FalseQA
# against single C is the only cell to fix. Per model and premise group: what single E and single C change against no write,
# and E+C against C at the same c (the single-C c chosen on the development questions in diag/simple_ec_1127.py).
# Labels: R right, W wrong, H hedge, N blank by the rule.
# usage (container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/fqa_premise_split_1128.py
import collections, sys
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import falseqa_read as F
ok, rows = F.check_bank()
prem = {r["id"]: r["premise"] for r in rows}
for m in ("qwen3.5-9b", "llama3.1-8b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b"):
    keep = {r["id"] for r in rows if r["set"] == "heldout"}
    R, _ = F.load_rows(m, keep)
    st = F.setting(m); s = st[1]
    lab = lambda arm, q: ("N" if R.blank(arm, q) else {1: "R", -1: "W", 0: "H"}[R.cell(arm, q)])
    def tr(a, b, qs):
        c = collections.Counter(lab(a, q) + lab(b, q) for q in qs)
        same = c["RR"] + c["WW"] + c["NN"] + c["HH"]
        return "changed %d of %d | R->W %d W->R %d | answered->blank %d (R %d, W %d) blank->answered %d (R %d, W %d)" % (
            len(qs) - same, len(qs), c["RW"], c["WR"], c["RN"] + c["WN"] + c["HN"], c["RN"], c["WN"], c["NR"] + c["NW"] + c["NH"], c["NR"], c["NW"])
    ids = list(R.ids)
    cbest = {"qwen3.5-9b": "0.5", "llama3.1-8b": "2", "qwen3.8-27b": "0.25", "gemma3-12b": "2", "gemma3-27b": "0.25"}[m]
    print("== %s (E %s, C %s)" % (m, s, cbest))
    for name, qs in (("premise false", [q for q in ids if prem[q] == "false"]), ("premise valid", [q for q in ids if prem[q] == "valid"])):
        net = lambda arm: sum(R.cell(arm, q) for q in qs)
        print("  %s %d: nets O %+d | E %+d | C %+d | EC(same c) %+d" % (name, len(qs), net("x"), net(R.e(s)), net(R.c(cbest)), net(R.ec(s, cbest))))
        print("     E vs no write:   " + tr("x", R.e(s), qs))
        print("     C vs no write:   " + tr("x", R.c(cbest), qs))
        print("     E+C vs C:        " + tr(R.c(cbest), R.ec(s, cbest), qs))

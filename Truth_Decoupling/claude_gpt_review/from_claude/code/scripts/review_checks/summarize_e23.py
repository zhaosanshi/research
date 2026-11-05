#!/usr/bin/env python3
"""Compact E-row table from E23 knob sweeps: the pre-declared exact-integer rule on the 192 tuning pairs, test deltas.

usage: summarize_e23.py CODE_DIR MODEL ROOT:TAG [ROOT:TAG ...]
For each sweep: the condition chosen by e23_compare_estimators.rule_pick (|d answer rate| <= 0.03 on tuning, max dJ as an
exact integer count, sweep-order tie-break), its tuning and test deltas of answer rate, J, AUROC and bias B, plus the
best-on-test condition (post hoc, for the selection-error column only) and the per-layer test dJ at the chosen alpha.
"""
import json
import os
import sys

import numpy as np

CODE = sys.argv[1]
MODEL = sys.argv[2]
sys.path[:0] = [os.path.join(CODE, "scripts"), CODE]
from e23_compare_estimators import rule_pick  # noqa: E402

for spec in sys.argv[3:]:
    root, tag = spec.split(":")
    d = os.path.join(root, "E23", MODEL)
    j = json.load(open(os.path.join(d, "knob_sweep_%s.json" % tag)))
    z = np.load(os.path.join(d, "knob_sweep_%s.npz" % tag), allow_pickle=True)
    lab, tune = z["labels"].astype(bool), z["half_a"].astype(bool)
    scale = int(lab[tune].sum()) * int((~lab[tune]).sum())
    names = [c for c in j["conds"] if c.startswith("E_")]
    ch = rule_pick(j["conds"], names, scale)
    b = j["conds"]["baseline"]

    def dl(name, split, k):
        return j["conds"][name][split][k] - b[split][k]

    best_test = max(names, key=lambda n: dl(n, "B", "J"))
    print("== %s %-18s estimator=%s variant=%s n_conds=%d" % (MODEL, tag, j.get("e_estimator"), (j.get("das") or {}).get("variant"), len(names)))
    if ch is None:
        print("   no feasible E condition on tuning")
        continue
    print("   chosen %-12s tune: d_ans %+.3f dJ %+.3f dAUROC %+.3f dB %+.3f | TEST: d_ans %+.3f dJ %+.3f dAUROC %+.3f dB %+.3f" % (
        ch, dl(ch, "A", "ans"), dl(ch, "A", "J"), dl(ch, "A", "AUROC"), dl(ch, "A", "B"),
        dl(ch, "B", "ans"), dl(ch, "B", "J"), dl(ch, "B", "AUROC"), dl(ch, "B", "B")))
    print("   post-hoc best on test %-12s dJ %+.3f dAUROC %+.3f d_ans %+.3f (selection error %.3f)" % (
        best_test, dl(best_test, "B", "J"), dl(best_test, "B", "AUROC"), dl(best_test, "B", "ans"), dl(best_test, "B", "J") - dl(ch, "B", "J")))
    a = ch.split("_a")[1]
    row = []
    for n in names:
        if n.endswith("_a" + a) and "-" not in n:
            row.append("L%s %+.3f" % (n[3:].split("_a")[0], dl(n, "B", "J")))
    print("   test dJ per layer at alpha %s: %s" % (a, "  ".join(row)))

#!/usr/bin/env python3
"""Is the 2x2 dissociation a property of the whole sweep, or only of the conditions the rule picked? (Claude, 2026-11-05)

usage: sweep_dissociation.py ROOT [--e-tag das_tfce_full] [--c-tag das_tfce_C] [--out OUT_JSON]
ROOT holds <model>/E23/knob_sweep_<tag>.json for llama3.1-8b, qwen3-8b, gemma3-4b (the synced results_das_tfce E23 folders).
Every number is on the TEST half (529 pairs, key "B"), relative to the baseline of the same sweep; the two sweeps' baselines
must be identical (checked). No selection rule is applied: all single-layer E conditions and all C conditions are counted.
  E : share of conditions with |d answer rate| <= 0.03 (leaves the answer rate alone), with d J > 0, with d AUROC > 0
  C : among conditions that move the answer rate by >= 10 points, share with |d J| <= 0.02, |d J| <= 0.04, |d AUROC| <= 0.02
The protocol C estimator of round 5 is Cp for Llama / Gemma and Cs for Qwen; both are reported.
"""
import argparse
import json
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--e-tag", default="das_tfce_full")
ap.add_argument("--c-tag", default="das_tfce_C")
ap.add_argument("--out")
args = ap.parse_args()
PRIMARY = {"llama3.1-8b": "Cp", "qwen3-8b": "Cs", "gemma3-4b": "Cp"}
out = {}
for m, prim in PRIMARY.items():
    E = json.load(open("%s/%s/E23/knob_sweep_%s.json" % (args.root, m, args.e_tag)))["conds"]
    C = json.load(open("%s/%s/E23/knob_sweep_%s.json" % (args.root, m, args.c_tag)))["conds"]
    if any(abs(E["baseline"]["B"][k] - C["baseline"]["B"][k]) > 1e-12 for k in ("ans", "J", "AUROC", "B")):
        raise SystemExit("%s: the E and C sweeps do not share the test baseline" % m)

    def d(cs, n, k):
        return cs[n]["B"][k] - cs["baseline"]["B"][k]

    en = [n for n in E if re.fullmatch(r"E_L\d+_a[\d.]+", n)]
    ea = np.array([d(E, n, "ans") for n in en])
    ej = np.array([d(E, n, "J") for n in en])
    eu = np.array([d(E, n, "AUROC") for n in en])
    r = {"E": {"n": len(en), "ans_within_0.03": int((abs(ea) <= 0.03).sum()), "dJ_pos": int((ej > 0).sum()),
               "dAUROC_pos": int((eu > 0).sum()), "max_abs_dans": float(abs(ea).max())}, "protocol_C": prim}
    print("== %s | E single-layer: %d conds; |d ans| <= 0.03: %d; d J > 0: %d; d AUROC > 0: %d; max |d ans| %.3f"
          % (m, len(en), r["E"]["ans_within_0.03"], r["E"]["dJ_pos"], r["E"]["dAUROC_pos"], r["E"]["max_abs_dans"]))
    for kind in ("Cp", "Cs"):
        cn = [n for n in C if n.startswith(kind + "_L")]
        a = np.array([d(C, n, "ans") for n in cn])
        j = np.array([d(C, n, "J") for n in cn])
        u = np.array([d(C, n, "AUROC") for n in cn])
        mv = abs(a) >= 0.10
        r[kind] = {"n": len(cn), "movers_10pts": int(mv.sum()), "J_within_0.02": int((mv & (abs(j) <= 0.02)).sum()),
                   "J_within_0.04": int((mv & (abs(j) <= 0.04)).sum()), "AUROC_within_0.02": int((mv & (abs(u) <= 0.02)).sum()),
                   "dJ_movers_median": float(np.median(j[mv])) if mv.any() else None}
        print("   %s%s: %d conds; move answer rate >= 10 pts: %d; of those |d J| <= 0.02: %d, <= 0.04: %d; |d AUROC| <= 0.02: %d; median d J %+.3f"
              % (kind, " (protocol C)" if kind == prim else "", len(cn), r[kind]["movers_10pts"], r[kind]["J_within_0.02"],
                 r[kind]["J_within_0.04"], r[kind]["AUROC_within_0.02"], r[kind]["dJ_movers_median"]))
    out[m] = r
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)

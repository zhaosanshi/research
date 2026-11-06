#!/usr/bin/env python3
"""Label-free dose probe of the E26 commitment direction C (Claude, 2026-11-06).

usage: e26_c_dose_probe.py MODEL RUN_DIR OUT_JSON [--n 64] [--alphas -2 -1 1 2]
RUN_DIR is a prepared E26 run (bank.json / bank.npz / data.json). For each task and method, at the C layer that prepare
chose, add alpha x scale x C at the gate prompt end (exactly the E26 'C' condition) and measure on the first N tuning
items: the mean shift of the first-token Yes-minus-No gate margin, that shift in units of the unsteered margin SD, and
the share of items whose Yes/No decision flips. No labels, references or test items are read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("run_dir", type=Path)
ap.add_argument("out")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--alphas", type=float, nargs="+", default=[-2.0, -1.0, 1.0, 2.0])
args = ap.parse_args()
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]
import e26_matched_coverage as e26  # noqa: E402

meta = e26.read(args.run_dir / "bank.json")
if meta["model"] != args.model:
    raise SystemExit("run dir belongs to %s" % meta["model"])
z = np.load(args.run_dir / "bank.npz")
bank = {k: z[k] for k in z.files}
data = e26.read(args.run_dir / "data.json")
bundle = e26.open_bundle(meta)
out = {"model": args.model, "run_dir": str(args.run_dir), "n": args.n, "alphas": args.alphas, "tasks": {}}
for task in e26.TASKS:
    rows = data[task]["tune"][: args.n]
    texts = [e26.gate_text(bundle, meta, task, r) for r in rows]
    lm = e26.gate_map(bundle, texts[0], meta["config"])
    base = {"name": "baseline", "kind": "base", "task": task, "method": "base", "e": None, "c": None}
    g0 = e26.one_step(bundle, texts, e26.intervention_specs(bank, base, "gate"))
    m0 = g0[:, lm["yes"]] - g0[:, lm["no"]]
    sd = float(np.std(m0))
    out["tasks"][task] = {"baseline_yes_share": float((m0 > 0).mean()), "baseline_margin_sd": sd, "methods": {}}
    for method, layers in meta["c_layers"][task].items():
        for layer in layers:
            res = {}
            for a in args.alphas:
                cond = {"name": "probe", "kind": "C", "task": task, "method": method, "e": None, "c": {"layer": int(layer), "alpha": float(a)}}
                g = e26.one_step(bundle, texts, e26.intervention_specs(bank, cond, "gate"))
                m = g[:, lm["yes"]] - g[:, lm["no"]]
                res[str(a)] = {"mean_shift": float((m - m0).mean()), "shift_in_sd": float((m - m0).mean() / sd) if sd > 0 else None,
                               "yes_share": float((m > 0).mean()), "flip_share": float(((m > 0) != (m0 > 0)).mean()),
                               "rank_spearman_vs_base": float(np.corrcoef(np.argsort(np.argsort(m)), np.argsort(np.argsort(m0)))[0, 1])}
                print("%s %s C L%d alpha %+g: shift %+.3f (%.2f sd) | yes share %.2f -> %.2f | flips %.2f | rank corr %.3f" % (
                    task, method, layer, a, res[str(a)]["mean_shift"], res[str(a)]["shift_in_sd"] or 0, (m0 > 0).mean(),
                    res[str(a)]["yes_share"], res[str(a)]["flip_share"], res[str(a)]["rank_spearman_vs_base"]), flush=True)
            out["tasks"][task]["methods"]["%s_L%d" % (method, layer)] = res
json.dump(out, open(args.out, "w"), indent=1)
print("saved", args.out)

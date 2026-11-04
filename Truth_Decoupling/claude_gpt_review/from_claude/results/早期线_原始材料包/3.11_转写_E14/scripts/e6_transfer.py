#!/usr/bin/env python3
"""E6: can the evidence direction be WRITTEN? KAPPA-style knowledge -> readout transfer (rank-1).

For every evaluation statement:
  1. read a knowledge score s_i from a linear probe at layer l_r (probe fitted on the direction-estimation
     statements only, never on evaluation items; standardised with the estimation-set mean/std, clipped to +-3);
  2. at write layer l_w, clamp the projection of the last-token residual onto the evidence direction w_w
     (E1 mean-diff, centred) to  z_i = g * half_gap_w * s_i   (alpha = 1 -> full clamp).
If the probe ranks items better than the model's own projection, J / AUROC must rise (the plan's H1 row);
if the model already "says what it knows", nothing can improve.

Controls: shuffled (s_i permuted across items: same marginal, no item knowledge), own_g (the model's own
projection scaled by g = plain amplify), baseline. Metrics and per-dataset breakdown as in E2b.

Usage: python3 scripts/e6_transfer.py --model llama3.1-8b [--read-layers 12 16 20] [--write-layers 18 16 20] [--gains 1 2]
Output: results/E6/<model>/transfer.json (+ .md)
"""
import argparse
import os
import time

from common import RESULTS_DIR, load_split, tf_prompt
from e2b_h1_rescue import metrics, run

import numpy as np
import torch

from core.config import load_experiment, load_model_config
from core.directions import fit_logreg, logreg_predict_proba
from core.intervene import SteerSpec
from core.io_utils import atomic_json_dump, load_json, set_seed
from core.metrics import auroc
from core.model_io import capture_last_token, load_model
from core.readout import build_label_token_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_h1_rescue", help="eval datasets / limits / batch size")
    ap.add_argument("--e1-exp", default="E1_directions", help="which statements the probes are fitted on")
    ap.add_argument("--model", default=None)
    ap.add_argument("--read-layers", nargs="+", type=int, default=None)
    ap.add_argument("--write-layers", nargs="+", type=int, default=None)
    ap.add_argument("--gains", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-oracle", action="store_true",
                    help="positive control only: clamp the projection to the GOLD label (does the readout follow the direction?)")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()

    exp = load_experiment(args.exp)
    if args.model:
        exp["model"] = load_model_config(args.model)
    mcfg, prompts = exp["model"], exp["prompts"]
    seed = exp.get("seed", 42)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    bs = exp.get("batch_size", 8)
    n_boot = exp.get("n_boot", 300)

    e1_dir = os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    e1 = load_json(os.path.join(e1_dir, "summary.json"))
    z = np.load(os.path.join(e1_dir, "directions_meandiff.npz"))
    wL = e1["E_write_layer"]
    e5_path = os.path.join(RESULTS_DIR, "E5", mcfg["name"], "knowledge_gap_%s.json" % exp.get("eval_split", "test"))
    e5_best = load_json(e5_path).get("best_probe_layer") if os.path.exists(e5_path) else None
    read_layers = sorted(set(args.read_layers or [12, 16, 20] + ([e5_best] if e5_best is not None else [])))
    write_layers = sorted(set(args.write_layers or [wL, 16, 20]))
    layers_needed = sorted(set(read_layers) | set(write_layers))

    bundle = load_model(mcfg)

    # ---- probes on the direction-estimation statements (never the eval items) --------------------- #
    e1_cfg = load_experiment(args.e1_exp)
    est = []
    for ds in e1_cfg["e_datasets"]:
        for sp in e1_cfg.get("direction_splits", ["calib", "dev"]):
            est += load_split(ds, sp)
    est_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est]
    y_est = np.array([1 if r["label"] else 0 for r in est])
    acts_est, _ = capture_last_token(bundle, est_texts, layers_needed, components=("resid",), batch_size=bs)
    probes, half_gap = {}, {}
    for l in read_layers:
        H = acts_est[("resid", l)].numpy().astype(np.float64)
        fit = fit_logreg(H, y_est, l2=args.l2, standardize=True)
        logit = H @ fit["w"] + fit["b"]
        probes[l] = {"fit": fit, "mu": float(logit.mean()), "sd": float(logit.std() + 1e-6),
                     "auc_train": float(auroc(logit, y_est == 1))}
    for l in write_layers:
        H = acts_est[("resid", l)].numpy().astype(np.float64)
        proj = (H - z["E_center_L%d" % l]) @ z["E_L%d" % l]
        half_gap[l] = float((proj[y_est == 1].mean() - proj[y_est == 0].mean()) / 2)
    print("probes (train AUC): %s | half-gap at write layers: %s" % (
        {l: round(p["auc_train"], 3) for l, p in probes.items()}, {l: round(v, 2) for l, v in half_gap.items()}), flush=True)

    # ---- evaluation items ------------------------------------------------------------------------- #
    recs = []
    per_ds = exp.get("eval_limit_per_dataset")
    for ds in exp["eval_datasets"]:
        r = load_split(ds, exp.get("eval_split", "test"))
        recs += r[:per_ds] if per_ds else r
    if args.limit:
        recs = recs[: args.limit]
    labels = [r["label"] for r in recs]
    datasets = [r["dataset"] for r in recs]
    y = np.array([1 if l else 0 for l in labels])
    texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in recs]
    lab_map = build_label_token_map(bundle.tokenizer, texts[0], prompts["statement"]["label_variants"])
    enc_batches = [bundle.encode(texts[i:i + bs]) for i in range(0, len(texts), bs)]
    slices = [slice(i, min(i + bs, len(texts))) for i in range(0, len(texts), bs)]
    acts_ev, _ = capture_last_token(bundle, texts, read_layers, components=("resid",), batch_size=bs)
    scores = {}
    for l in read_layers:
        H = acts_ev[("resid", l)].numpy().astype(np.float64)
        logit = H @ probes[l]["fit"]["w"] + probes[l]["fit"]["b"]
        s = np.clip((logit - probes[l]["mu"]) / probes[l]["sd"], -3.0, 3.0)
        scores[l] = s
        per_ds_auc = {d: round(float(auroc(s[np.array(datasets) == d], y[np.array(datasets) == d] == 1)), 3) for d in sorted(set(datasets))}
        print("  probe L%d on eval items: AUROC %.3f %s" % (l, auroc(s, y == 1), per_ds_auc), flush=True)
    print("E6 %s: n=%d read %s write %s gains %s" % (mcfg["name"], len(recs), read_layers, write_layers, args.gains), flush=True)

    t0 = time.time()
    base_p, base_pred = run(bundle, enc_batches, lab_map, None)
    results = {"model": mcfg["name"], "n": len(recs), "read_layers": read_layers, "write_layers": write_layers,
               "probe_train_auc": {str(l): p["auc_train"] for l, p in probes.items()},
               "probe_eval_auc": {str(l): float(auroc(scores[l], y == 1)) for l in read_layers},
               "half_gap": {str(l): v for l, v in half_gap.items()}, "conditions": {}}
    results["conditions"]["baseline"] = metrics(labels, base_p, base_pred, base_p, base_pred, datasets, rng, 1)
    print("baseline AUROC %.3f J %.3f B %.3f ans %.3f" % tuple(results["conditions"]["baseline"][k] for k in ("auroc", "J", "B", "answer_rate")), flush=True)

    conds = []
    perm = np.random.default_rng(seed + 7).permutation(len(recs))
    gold = 2.0 * y - 1.0
    for lw in write_layers:
        w, c = z["E_L%d" % lw], z["E_center_L%d" % lw]
        for g in args.gains:
            # positive control: write the GOLD label into the projection. If J does not follow, the write
            # mechanism (not the knowledge source) is broken.
            tgt_or = g * half_gap[lw] * gold
            conds.append(("oracle_w%d_g%g" % (lw, g),
                          lambda bi, w=w, lw=lw, tgt_or=tgt_or: [SteerSpec(w, lw, mode="clamp", alpha=args.alpha, z_target=torch.tensor(tgt_or[slices[bi]], dtype=torch.float32))]))
            if args.only_oracle:
                continue
            for lr in read_layers:
                tgt = g * half_gap[lw] * scores[lr]
                conds.append(("transfer_r%d_w%d_g%g" % (lr, lw, g),
                              lambda bi, w=w, lw=lw, tgt=tgt: [SteerSpec(w, lw, mode="clamp", alpha=args.alpha, z_target=torch.tensor(tgt[slices[bi]], dtype=torch.float32))]))
                if g == args.gains[0]:
                    tgt_sh = tgt[perm]
                    conds.append(("shuffled_r%d_w%d_g%g" % (lr, lw, g),
                                  lambda bi, w=w, lw=lw, tgt_sh=tgt_sh: [SteerSpec(w, lw, mode="clamp", alpha=args.alpha, z_target=torch.tensor(tgt_sh[slices[bi]], dtype=torch.float32))]))
            # own projection scaled by g (= amplify with alpha g-1)
            conds.append(("own_w%d_g%g" % (lw, g), lambda bi, w=w, lw=lw, c=c, g=g: [SteerSpec(w, lw, mode="amplify", alpha=g - 1.0, center=c)]))

    for i, (name, mk) in enumerate(conds):
        p, pred = run(bundle, enc_batches, lab_map, mk)
        m = metrics(labels, p, pred, base_p, base_pred, datasets, rng, n_boot)
        results["conditions"][name] = m
        print("  [%2d/%d] %-26s AUROC %.3f (d %+.3f [%+.3f,%+.3f])  J %.3f  B %.3f  ans %.3f  J_pair %.3f/%.3f  (%.0fs)" % (
            i + 1, len(conds), name, m["auroc"], m["auroc"] - base_p_auroc(results), m["d_auroc_ci95"][0], m["d_auroc_ci95"][1],
            m["J"], m["B"], m["answer_rate"], m["J_paired"], m["J_paired_base"], time.time() - t0), flush=True)

    out_dir = os.path.join(RESULTS_DIR, "E6", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    results["elapsed_s"] = time.time() - t0
    stem = ("transfer_oracle" if args.only_oracle else "transfer") + (("_" + args.tag) if args.tag else "")
    atomic_json_dump(results, os.path.join(out_dir, stem + ".json"))
    b = results["conditions"]["baseline"]
    lines = ["# E6 transfer: %s (n=%d)" % (mcfg["name"], len(recs)), "",
             "probe train AUC %s | probe eval AUC %s | half-gap %s" % (results["probe_train_auc"], {k: round(v, 3) for k, v in results["probe_eval_auc"].items()}, {k: round(v, 2) for k, v in results["half_gap"].items()}), "",
             "| condition | AUROC | dAUROC [95% CI] | J | B | answer rate | J_paired (cond/base) | per-dataset AUROC |", "|---|---|---|---|---|---|---|---|"]
    for name, m in results["conditions"].items():
        lines.append("| %s | %.3f | %+.3f [%+.3f, %+.3f] | %.3f | %.3f | %.3f | %.3f / %.3f | %s |" % (
            name, m["auroc"], m["auroc"] - b["auroc"], m["d_auroc_ci95"][0], m["d_auroc_ci95"][1], m["J"], m["B"], m["answer_rate"],
            m["J_paired"], m["J_paired_base"], {d: round(v["auroc"], 3) for d, v in m["per_dataset"].items()}))
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(out_dir, stem + ".md"))


def base_p_auroc(results):
    return results["conditions"]["baseline"]["auroc"]


if __name__ == "__main__":
    main()

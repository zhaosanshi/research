#!/usr/bin/env python3
"""E13: the teacher's "one table" (her step 1) under a leakage-free protocol, plus the pre-registered prediction (step 6).

Everything on ONE set of evaluation items (the same test statements the dose curves use), ONE metric (AUROC over all
items; the model's score is P(True | answered) from the unsteered pass) and probes that never saw an evaluation label:
  probe_auc      logistic probe fitted on the estimation statements only (calib+dev of the same dataset), scored on eval
  meandiff_auc   projection on the mean-difference direction estimated on the same estimation statements (what E writes)
  e1dir_auc      projection on the saved E1 direction (consistency check; equals meandiff_auc up to the estimator options)
  out_auc        the model's own output
E5 used 5-fold CV on the evaluation items (a probe trained on other eval items); its numbers are therefore slightly
optimistic and are NOT used here.

Rooms (per dataset):  up_room = max_{l in room-layers} meandiff_auc(l) - out_auc   (what a write along E could add)
                      down_room = out_auc - 0.5                                   (what an erase could destroy)
Pre-registered prediction rule (fixed before any steering on a new set; evaluated afterwards by E11):
  amplify gain at alpha=3 on layers 12+13+14: "+" and within [0.3, 0.7] x up_room if up_room > 0.03; "~0" if |up_room| <= 0.03;
  "-" if up_room < -0.03.  Erase (alpha=-1): drop roughly proportional to down_room; "none" if down_room < 0.05.

Usage: python3 scripts/e13_clean_gap.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b
Output: results/E13/<model>/clean_gap[_tag].{json,md,npz}
"""
import argparse
import os
import time

from common import RESULTS_DIR, load_split, tf_prompt
from e2b_h1_rescue import run

import numpy as np

from core.config import load_experiment, load_model_config
from core.directions import fit_logreg, mean_diff_direction
from core.io_utils import atomic_json_dump, set_seed
from core.metrics import auroc, confusion_tf
from core.model_io import capture_last_token, load_model
from core.readout import build_label_token_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--e1-exp", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="default: all layers from 2 to num_layers-2")
    ap.add_argument("--room-layers", nargs="+", type=int, default=None, help="default: layers between 30%% and 60%% depth")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    exp = load_experiment(args.exp)
    if args.model:
        exp["model"] = load_model_config(args.model)
    mcfg, prompts = exp["model"], exp["prompts"]
    set_seed(exp.get("seed", 42))
    bs = exp.get("batch_size", 8)
    bundle = load_model(mcfg)
    if args.layers is None:
        args.layers = list(range(2, bundle.num_layers - 1))
    if args.room_layers is None:
        args.room_layers = [l for l in args.layers if 0.3 * bundle.num_layers <= l <= 0.6 * bundle.num_layers]

    e1_cfg = load_experiment(args.e1_exp)
    est = []
    for ds in e1_cfg["e_datasets"]:
        for sp in e1_cfg.get("direction_splits", ["calib", "dev"]):
            est += load_split(ds, sp)
    y_est = np.array([1 if r["label"] else 0 for r in est])
    recs = []
    per_ds = exp.get("eval_limit_per_dataset")
    for ds in exp["eval_datasets"]:
        r = load_split(ds, exp.get("eval_split", "test"))
        recs += r[:per_ds] if per_ds else r
    labels = np.array([bool(r["label"]) for r in recs])
    dsets = np.array([r["dataset"] for r in recs])
    est_ids = set(r["id"] for r in est)
    est_pairs = set(r.get("pair_id") or r["id"] for r in est)
    overlap = int(sum(r["id"] in est_ids for r in recs))
    pair_overlap = int(sum((r.get("pair_id") or r["id"]) in est_pairs for r in recs))
    print("E13 %s: est %d statements (%s), eval %d statements (%s) | id overlap %d | pair overlap %d" % (
        mcfg["name"], len(est), ",".join(e1_cfg["e_datasets"]), len(recs), ",".join(exp["eval_datasets"]), overlap, pair_overlap), flush=True)
    if overlap or pair_overlap:
        raise SystemExit("estimation and evaluation items overlap; refusing to continue")

    t0 = time.time()
    layers = list(args.layers)
    acts_est, _ = capture_last_token(bundle, [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est], layers, components=("resid",), batch_size=bs)
    texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in recs]
    acts_ev, _ = capture_last_token(bundle, texts, layers, components=("resid",), batch_size=bs)
    lab_map = build_label_token_map(bundle.tokenizer, texts[0], prompts["statement"]["label_variants"])
    enc_batches = [bundle.encode(texts[i:i + bs]) for i in range(0, len(texts), bs)]
    base_p, base_pred = run(bundle, enc_batches, lab_map, None)
    base_pred = np.asarray(base_pred)
    out_auc = float(auroc(base_p, labels))
    conf = confusion_tf(labels, list(base_pred))
    print("  captured in %.0fs | output AUROC %.3f | answer rate %.3f | acc all %.3f" % (time.time() - t0, out_auc, conf["answer_rate"], conf["acc_all"]), flush=True)

    e1_npz = os.path.join(RESULTS_DIR, "E1", mcfg["name"], "directions_meandiff.npz")
    z = np.load(e1_npz) if os.path.exists(e1_npz) else None
    rows = {}
    probe_store = np.zeros((len(layers), len(recs)), dtype=np.float32)
    md_store = np.zeros((len(layers), len(recs)), dtype=np.float32)
    for i, l in enumerate(layers):
        H = acts_est[("resid", l)].numpy().astype(np.float64)
        Hv = acts_ev[("resid", l)].numpy().astype(np.float64)
        fit = fit_logreg(H, y_est, l2=1.0, standardize=True)
        s = Hv @ fit["w"] + fit["b"]
        w, _ = mean_diff_direction(H[y_est == 1], H[y_est == 0])
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        proj = (Hv - H.mean(0)) @ w
        row = {"probe_auc": float(auroc(s, labels)), "probe_train_auc": float(auroc(H @ fit["w"] + fit["b"], y_est == 1)),
               "meandiff_auc": float(auroc(proj, labels)), "per_dataset": {}}
        if z is not None and ("E_L%d" % l) in z.files:
            row["e1dir_auc"] = float(auroc((Hv - z["E_center_L%d" % l]) @ z["E_L%d" % l], labels))
        for d in sorted(set(dsets)):
            m = dsets == d
            row["per_dataset"][d] = {"n": int(m.sum()), "probe_auc": float(auroc(s[m], labels[m])), "meandiff_auc": float(auroc(proj[m], labels[m])),
                                     "out_auc": float(auroc(base_p[m], labels[m]))}
        rows[l] = row
        probe_store[i], md_store[i] = s, proj
        print("  L%2d probe %.3f (train %.3f)  meandiff %.3f  e1dir %s  | output %.3f" % (
            l, row["probe_auc"], row["probe_train_auc"], row["meandiff_auc"], ("%.3f" % row["e1dir_auc"]) if "e1dir_auc" in row else "  -  ", out_auc), flush=True)

    room_layers = [l for l in args.room_layers if l in rows]
    best_probe_l = max(room_layers, key=lambda l: rows[l]["probe_auc"])
    best_md_l = max(room_layers, key=lambda l: rows[l]["meandiff_auc"])
    up_room_probe = rows[best_probe_l]["probe_auc"] - out_auc
    up_room_md = rows[best_md_l]["meandiff_auc"] - out_auc
    down_room = out_auc - 0.5
    if up_room_md > 0.03:
        pred_amp = {"sign": "+", "d_auroc_range_alpha3_L12-14": [round(0.3 * up_room_md, 3), round(0.7 * up_room_md, 3)]}
    elif up_room_md < -0.03:
        pred_amp = {"sign": "-", "d_auroc_range_alpha3_L12-14": [round(up_room_md, 3), 0.0]}
    else:
        pred_amp = {"sign": "~0", "d_auroc_range_alpha3_L12-14": [-0.02, 0.02]}
    pred_erase = {"drop": "none" if down_room < 0.05 else ("small" if down_room < 0.2 else "large"), "down_room": round(down_room, 3)}

    per_ds_summary = {}
    for d in sorted(set(dsets)):
        pd_rows = {l: rows[l]["per_dataset"][d] for l in room_layers}
        bl = max(room_layers, key=lambda l: pd_rows[l]["meandiff_auc"])
        per_ds_summary[d] = {"n": pd_rows[bl]["n"], "out_auc": pd_rows[bl]["out_auc"], "best_meandiff_auc": pd_rows[bl]["meandiff_auc"], "best_meandiff_layer": bl,
                             "best_probe_auc": max(pd_rows[l]["probe_auc"] for l in room_layers),
                             "up_room_md": pd_rows[bl]["meandiff_auc"] - pd_rows[bl]["out_auc"], "down_room": pd_rows[bl]["out_auc"] - 0.5}

    result = {"model": mcfg["name"], "exp": args.exp, "e1_exp": args.e1_exp, "n_est": len(est), "n_eval": len(recs), "layers": layers, "room_layers": room_layers,
              "output": {"auroc": out_auc, "answer_rate": conf["answer_rate"], "acc_all": conf["acc_all"], "J": conf["J"], "B": conf["B"]},
              "per_layer": {str(l): rows[l] for l in layers},
              "best_probe_layer": best_probe_l, "best_meandiff_layer": best_md_l, "up_room_probe": up_room_probe, "up_room_meandiff": up_room_md, "down_room": down_room,
              "prediction_before_steering": {"amplify": pred_amp, "erase": pred_erase}, "per_dataset": per_ds_summary, "elapsed_s": time.time() - t0}
    out_dir = os.path.join(RESULTS_DIR, "E13", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "clean_gap" + (("_" + args.tag) if args.tag else ""))
    atomic_json_dump(result, stem + ".json")
    np.savez(stem + ".npz", ids=np.array([r["id"] for r in recs]), labels=labels, datasets=dsets, layers=np.array(layers), probe_scores=probe_store, meandiff_proj=md_store,
             base_p=base_p, base_pred=base_pred)
    lines = ["# E13 clean knowledge-prediction gap: %s (eval n=%d from %s; estimation n=%d from %s calib+dev; no item/pair overlap)" % (
        mcfg["name"], len(recs), ",".join(exp["eval_datasets"]), len(est), ",".join(e1_cfg["e_datasets"])), "",
        "Output (unsteered): AUROC %.3f | answer rate %.3f | acc all %.3f | J %.3f | B %.3f" % (out_auc, conf["answer_rate"], conf["acc_all"], conf["J"], conf["B"]), "",
        "| layer | clean probe AUROC (train) | clean mean-diff AUROC | E1 direction AUROC | output AUROC | probe - output | mean-diff - output |", "|---|---|---|---|---|---|---|"]
    for l in layers:
        r = rows[l]
        lines.append("| %d | %.3f (%.2f) | %.3f | %s | %.3f | %+.3f | %+.3f |" % (l, r["probe_auc"], r["probe_train_auc"], r["meandiff_auc"],
                                                                                 ("%.3f" % r["e1dir_auc"]) if "e1dir_auc" in r else "-", out_auc, r["probe_auc"] - out_auc, r["meandiff_auc"] - out_auc))
    lines += ["", "Rooms over layers %s: up_room(probe) %+.3f (L%d), up_room(mean-diff) %+.3f (L%d), down_room %+.3f" % (
        room_layers, up_room_probe, best_probe_l, up_room_md, best_md_l, down_room),
        "Pre-registered prediction before steering: amplify alpha=3 @L12-14 -> %s, dAUROC in %s; erase alpha=-1 -> %s drop" % (
            pred_amp["sign"], pred_amp["d_auroc_range_alpha3_L12-14"], pred_erase["drop"]), ""]
    if len(per_ds_summary) > 1:
        lines += ["| dataset | n | output AUROC | best clean mean-diff AUROC (layer) | best clean probe AUROC | up_room (mean-diff) | down_room |", "|---|---|---|---|---|---|---|"]
        for d, s in per_ds_summary.items():
            lines.append("| %s | %d | %.3f | %.3f (L%d) | %.3f | %+.3f | %+.3f |" % (d, s["n"], s["out_auc"], s["best_meandiff_auc"], s["best_meandiff_layer"], s["best_probe_auc"], s["up_room_md"], s["down_room"]))
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()

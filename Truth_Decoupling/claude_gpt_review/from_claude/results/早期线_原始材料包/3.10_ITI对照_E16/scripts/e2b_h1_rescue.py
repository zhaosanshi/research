#!/usr/bin/env python3
"""E2b: can ANY evidence-direction intervention raise discrimination? (H1 rescue attempts)

The pilot (E2) tested one estimator (mean-diff), one strength (alpha=1), one layer per model and
two datasets, applied uniformly. Before concluding that H1 fails, try everything the plan allows:

  * alpha sweep for E_amp (gain) and small-alpha E_add at the LA-Score write layer
  * E_amp at the E3 oracle layer and at extra layers
  * DAS-trained E directions (the plan's primary estimator), amplify mode
  * gated E (the plan's Sec. 3.5 idea): alpha_i = alpha_max * (1 - |2 p_i - 1|) using the baseline
    P(True) as the uncertainty signal, plus a hard middle-tercile gate
  * C_add alpha sweep (matched-answer-rate comparison with prompt-only)
  * per-dataset breakdown (CounterFact / GoT / FEVER)

Metrics per condition: AUROC(P(True)) over all items (criterion-free sensitivity) with paired
bootstrap CI vs baseline, J / B on answered items, J / B on items answered in both, answer rate.

Usage: python3 scripts/e2b_h1_rescue.py --exp E2b_h1_rescue --model qwen3-8b
"""
import argparse
import glob
import math
import os
import time

from common import RESULTS_DIR, load_split, tf_prompt

import numpy as np
import torch

from core.config import load_experiment, load_model_config
from core.intervene import SteerSpec, next_token_logits
from core.io_utils import atomic_json_dump, load_json, set_seed
from core.metrics import auroc, confusion_tf
from core.model_io import load_model
from core.readout import build_label_token_map, decide_label, label_logprobs_from_logits, p_true_given_answer


def run(bundle, enc_batches, lab_map, make_specs):
    """make_specs(batch_index, batch_slice) -> list of SteerSpec (or None). Returns (p_true[n], pred[n])."""
    p_all, pred_all = [], []
    for bi, enc in enumerate(enc_batches):
        lg = next_token_logits(bundle, enc, specs=make_specs(bi) if make_specs else None)
        for l in lg:
            lp = label_logprobs_from_logits(l, lab_map)
            p_all.append(p_true_given_answer(lp))
            pred_all.append(decide_label(lp))
    return np.array(p_all, dtype=float), pred_all


def metrics(labels, p, pred, base_p, base_pred, datasets, rng, n_boot):
    labels = np.asarray(labels, dtype=bool)
    ans = np.array([x != "Abstain" for x in pred])
    base_ans = np.array([x != "Abstain" for x in base_pred])
    both = ans & base_ans
    conf = confusion_tf(labels, pred)
    cp = confusion_tf(labels[both], [pred[i] for i in np.where(both)[0]])
    bp = confusion_tf(labels[both], [base_pred[i] for i in np.where(both)[0]])
    au = auroc(p, labels)
    n = len(labels)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(auroc(p[idx], labels[idx]) - auroc(base_p[idx], labels[idx]))
    per_ds = {}
    ds_arr = np.array(datasets)
    for d in sorted(set(datasets)):
        m = ds_arr == d
        c_d = confusion_tf(labels[m], [pred[i] for i in np.where(m)[0]])
        per_ds[d] = {"auroc": auroc(p[m], labels[m]), "J": c_d["J"], "B": c_d["B"], "answer_rate": c_d["answer_rate"], "n": int(m.sum())}
    return {"auroc": au, "d_auroc_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "J": conf["J"], "B": conf["B"], "answer_rate": conf["answer_rate"], "acc_all": conf["acc_all"],
            "J_paired": cp["J"], "J_paired_base": bp["J"], "B_paired": cp["B"], "B_paired_base": bp["B"], "n_paired": int(both.sum()),
            "per_dataset": per_ds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_h1_rescue")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
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
    wL, cL = e1["E_write_layer"], e1["C_write_layer"]
    e3_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "E3", mcfg["name"], "*.json")))
    oracle_L = load_json(e3_files[-1])["E"]["oracle_layer"] if e3_files else wL
    layers_E = sorted(set([wL, oracle_L] + [l for l in exp.get("extra_layers", []) if l < e1["rows"].__len__()]))

    bundle = load_model(mcfg)
    recs = []
    per_ds = exp.get("eval_limit_per_dataset")
    for ds in exp["eval_datasets"]:
        r = load_split(ds, exp.get("eval_split", "test"))
        recs += r[:per_ds] if per_ds else r
    if args.limit:
        recs = recs[: args.limit]
    labels = [r["label"] for r in recs]
    datasets = [r["dataset"] for r in recs]
    texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in recs]
    lab_map = build_label_token_map(bundle.tokenizer, texts[0], prompts["statement"]["label_variants"])
    enc_batches = [bundle.encode(texts[i:i + bs]) for i in range(0, len(texts), bs)]
    slices = [slice(i, min(i + bs, len(texts))) for i in range(0, len(texts), bs)]
    print("E2b %s: n=%d, E layers %s (write L%d, oracle L%d), C layer L%d" % (mcfg["name"], len(recs), layers_E, wL, oracle_L, cL), flush=True)
    t0 = time.time()

    base_p, base_pred = run(bundle, enc_batches, lab_map, None)
    base_unc = 1.0 - np.abs(2 * np.nan_to_num(base_p, nan=0.5) - 1.0)  # 1 = maximally uncertain
    results = {"model": mcfg["name"], "n": len(recs), "layers_E": layers_E, "c_layer": cL, "conditions": {}}
    results["conditions"]["baseline"] = metrics(labels, base_p, base_pred, base_p, base_pred, datasets, rng, 1)

    def wE(l):
        return z["E_L%d" % l], z["E_center_L%d" % l]

    conds = []
    for l in layers_E:
        for a in exp.get("alphas_amp", [0.5, 1, 2, 3, 5]):
            conds.append(("E_amp_L%d_a%g" % (l, a), lambda bi, l=l, a=a: [SteerSpec(wE(l)[0], l, mode="amplify", alpha=a, center=wE(l)[1])]))
    for a in exp.get("alphas_add", [0.1, 0.25, 0.5]):
        conds.append(("E_add_L%d_a%g" % (wL, a), lambda bi, a=a: [SteerSpec(wE(wL)[0], wL, mode="add", alpha=a, coef=[e1["rows"][str(wL)]["gap_E"]])]))
    # negative-gain probe: erasing the E component should destroy discrimination if E carries it
    conds.append(("E_erase_L%d" % wL, lambda bi: [SteerSpec(wE(wL)[0], wL, mode="amplify", alpha=-1.0, center=wE(wL)[1])]))
    # DAS-trained E directions
    for pt in sorted(glob.glob(os.path.join(e1_dir, "das_E_L*_r1_s*.pt"))):
        res = torch.load(pt, map_location="cpu", weights_only=False)
        l = res["layer"]
        w = res["W"][:, 0].numpy()
        for a in exp.get("alphas_das", [1, 3]):
            conds.append(("E_das_amp_L%d_a%g" % (l, a), lambda bi, w=w, l=l, a=a: [SteerSpec(w, l, mode="amplify", alpha=a, center=z["E_center_L%d" % l])]))
        conds.append(("E_das_add_L%d_a0.5" % l, lambda bi, w=w, l=l, res=res: [SteerSpec(w, l, mode="add", alpha=0.5, coef=[abs(res["mean_gap"][0])])]))
    # gated E (plan Sec. 3.5): strength proportional to baseline uncertainty; and hard middle-tercile gate
    for a in exp.get("alphas_gated", [1, 3]):
        conds.append(("E_gated_soft_L%d_a%g" % (wL, a), lambda bi, a=a: [SteerSpec(wE(wL)[0], wL, mode="amplify",
                      alpha=torch.tensor(a * base_unc[slices[bi]], dtype=torch.float32), center=wE(wL)[1])]))
    lo, hi = np.quantile(base_unc, [1 / 3, 2 / 3])
    hard = ((base_unc >= lo) & (base_unc <= hi)).astype(np.float32)
    for a in exp.get("alphas_gated", [1, 3]):
        conds.append(("E_gated_hard_L%d_a%g" % (wL, a), lambda bi, a=a: [SteerSpec(wE(wL)[0], wL, mode="amplify",
                      alpha=torch.tensor(a * hard[slices[bi]], dtype=torch.float32), center=wE(wL)[1])]))
    # C sweep for matched-answer-rate comparison
    gC = e1["rows"][str(cL)]["gap_C"]
    for a in exp.get("alphas_c", [0.5, 1, 1.5, 2]):
        conds.append(("C_add_L%d_a%g" % (cL, a), lambda bi, a=a: [SteerSpec(z["C_L%d" % cL], cL, mode="add", alpha=-a, coef=[gC])]))

    for name, mk in conds:
        p, pred = run(bundle, enc_batches, lab_map, mk)
        m = metrics(labels, p, pred, base_p, base_pred, datasets, rng, n_boot)
        results["conditions"][name] = m
        print("  %-26s AUROC=%.3f dCI=[%+.3f,%+.3f] J=%.3f Jp=%.3f(b%.3f) B=%.3f Bp=%.3f(b%.3f) Rans=%.3f (%.0fs)" % (
            name, m["auroc"], m["d_auroc_ci95"][0], m["d_auroc_ci95"][1], m["J"], m["J_paired"], m["J_paired_base"], m["B"],
            m["B_paired"], m["B_paired_base"], m["answer_rate"], time.time() - t0), flush=True)

    results["baseline_uncertainty_terciles"] = [float(lo), float(hi)]
    results["elapsed_s"] = time.time() - t0
    out_dir = os.path.join(RESULTS_DIR, "E2b", mcfg["name"])
    atomic_json_dump(results, os.path.join(out_dir, "e2b_%s.json" % mcfg["name"]))
    # markdown
    b = results["conditions"]["baseline"]
    lines = ["# E2b H1 rescue: %s (n=%d)" % (mcfg["name"], len(recs)), "",
             "baseline AUROC %.3f | J %.3f | B %.3f | answer rate %.3f | per-dataset AUROC: %s" % (
                 b["auroc"], b["J"], b["B"], b["answer_rate"], {d: round(v["auroc"], 3) for d, v in b["per_dataset"].items()}), "",
             "| condition | AUROC | ΔAUROC 95% CI | J | J paired (base) | B | B paired (base) | answer rate | per-dataset ΔAUROC |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in results["conditions"].items():
        if name == "baseline":
            continue
        pd_s = ", ".join("%s %+.3f" % (d[:4], m["per_dataset"][d]["auroc"] - b["per_dataset"][d]["auroc"]) for d in m["per_dataset"])
        lines.append("| %s | %.3f | [%+.3f, %+.3f] | %.3f | %.3f (%.3f) | %.3f | %.3f (%.3f) | %.3f | %s |" % (
            name, m["auroc"], m["d_auroc_ci95"][0], m["d_auroc_ci95"][1], m["J"], m["J_paired"], m["J_paired_base"], m["B"],
            m["B_paired"], m["B_paired_base"], m["answer_rate"], pd_s))
    with open(os.path.join(out_dir, "e2b_%s.md" % mcfg["name"]), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nE2b done -> %s (%.0fs)" % (out_dir, results["elapsed_s"]), flush=True)


if __name__ == "__main__":
    main()

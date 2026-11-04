#!/usr/bin/env python3
"""E10: per-input GATING of the evidence write (the plan's Sec. 3.5 / claim C3), on TruthfulQA statements.

The global evidence write fixes many wrong items but also damages known-correct ones (TQA: 215 fixed / 90 damaged).
Gating decides per item whether / how hard to push, using only label-free signals available at inference:
  u_i   output uncertainty of the unsteered pass, 1 - |2 P_i(True) - 1|           (0 = confident, 1 = coin flip)
  s_i   knowledge-probe score at layer l_r (logistic probe fitted on calib+dev statements only, standardised)
  d_i   disagreement: sign(s_i) != sign(logit P_i)  (the probe and the mouth lean opposite ways)

Conditions (evidence operator = amplify at layers 12+13+14 unless noted; alpha_i per item):
  global_a3                  ungated reference
  soft_unc_a{A}              alpha_i = A * u_i                      (push the uncertain, spare the confident)
  hard_unc_t{T}_a3           alpha_i = 3 if u_i > T else 0
  disagree_a{A}              alpha_i = A if d_i else 0              (push only where probe and mouth disagree)
  agree_only_a3              alpha_i = 3 if not d_i else 0          (control: should do little)
  probeconf_t{T}_a3          alpha_i = 3 if |s_i| > T else 0        (the plan's "push when the probe knows")
  transfer_global_g2         KAPPA-style clamp at L16 to g*half_gap*s_i on all items (reference)
  transfer_disagree_g2       the same clamp only on disagreeing items
  combo_disagree_E_unc_C     E transfer on disagreeing items + C push (toward abstain) on items where |s_i| is small
  prompt_only_cautious       system-prompt swap, no steering (the prompt-only baseline every gate must beat)
Metrics per condition: corrected (wrong->right), damaged (right->wrong), abstention transitions, known-item damage rate,
unknown-item correction rate, accuracy over all items, wrong rate, answer rate, AUROC, J, B, penalised score.

Usage: python3 scripts/e10_gating.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b
Output: results/E10/<model>/gating.{json,md,npz}
"""
import argparse
import os
import time

from common import RESULTS_DIR, load_split, tf_prompt
from e2b_h1_rescue import run

import numpy as np
import torch

from core.config import load_experiment, load_model_config
from core.directions import fit_logreg
from core.intervene import SteerSpec
from core.io_utils import atomic_json_dump, load_json, set_seed
from core.metrics import auroc, confusion_tf
from core.model_io import capture_last_token, load_model
from core.readout import build_label_token_map


def state(pred, labels):
    pred = np.asarray(pred)
    ans = pred != "Abstain"
    corr = ans & ((pred == "True") == labels)
    return np.where(~ans, "abstain", np.where(corr, "known", "wrong"))


def summarize(labels, p, pred, base_pred, base_p, rng, n_boot=300):
    labels = np.asarray(labels, dtype=bool)
    sb, sc = state(base_pred, labels), state(pred, labels)
    t = {"%s->%s" % (x, y): int(((sb == x) & (sc == y)).sum()) for x in ("known", "wrong", "abstain") for y in ("known", "wrong", "abstain")}
    n_known, n_wrong, n_abst = int((sb == "known").sum()), int((sb == "wrong").sum()), int((sb == "abstain").sum())
    conf = confusion_tf(labels, list(pred))
    n = len(labels)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(auroc(p[idx], labels[idx]) - auroc(base_p[idx], labels[idx]))
    return {"corrected": t["wrong->known"], "damaged": t["known->wrong"], "wrong_to_abstain": t["wrong->abstain"], "known_to_abstain": t["known->abstain"],
            "abstain_to_known": t["abstain->known"], "abstain_to_wrong": t["abstain->wrong"],
            "known_damage_rate": t["known->wrong"] / max(1, n_known), "known_abstain_rate": t["known->abstain"] / max(1, n_known),
            "unknown_correction_rate": t["wrong->known"] / max(1, n_wrong), "unknown_abstain_rate": t["wrong->abstain"] / max(1, n_wrong),
            "n_correct": int((sc == "known").sum()), "n_wrong": int((sc == "wrong").sum()), "n_abstain": int((sc == "abstain").sum()),
            "acc_all": float((sc == "known").mean()), "wrong_all": float((sc == "wrong").mean()), "answer_rate": conf["answer_rate"],
            "penalised": float(((sc == "known").sum() - (sc == "wrong").sum()) / n),
            "auroc": float(auroc(p, labels)), "d_auroc_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "J": conf["J"], "B": conf["B"], "transitions": t, "baseline_counts": {"known": n_known, "wrong": n_wrong, "abstain": n_abst}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa")
    ap.add_argument("--e1-exp", default="E1_directions_tqa")
    ap.add_argument("--model", default=None)
    ap.add_argument("--e-layers", nargs="+", type=int, default=[12, 13, 14])
    ap.add_argument("--probe-layer", type=int, default=14)
    ap.add_argument("--write-layer", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    exp = load_experiment(args.exp)
    if args.model:
        exp["model"] = load_model_config(args.model)
    mcfg, prompts = exp["model"], exp["prompts"]
    seed = exp.get("seed", 42)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    bs = exp.get("batch_size", 8)
    e1_dir = os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    e1 = load_json(os.path.join(e1_dir, "summary.json"))
    z = np.load(os.path.join(e1_dir, "directions_meandiff.npz"))
    cL = e1["C_write_layer"]
    bundle = load_model(mcfg)
    lr, lw = args.probe_layer, args.write_layer

    # ---- knowledge probe on the estimation statements (never eval items) ---------------------------- #
    e1_cfg = load_experiment(args.e1_exp)
    est = []
    for ds in e1_cfg["e_datasets"]:
        for sp in e1_cfg.get("direction_splits", ["calib", "dev"]):
            est += load_split(ds, sp)
    y_est = np.array([1 if r["label"] else 0 for r in est])
    acts_est, _ = capture_last_token(bundle, [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est], [lr, lw], components=("resid",), batch_size=bs)
    H = acts_est[("resid", lr)].numpy().astype(np.float64)
    fit = fit_logreg(H, y_est, l2=1.0, standardize=True)
    logit_est = H @ fit["w"] + fit["b"]
    mu, sd = float(logit_est.mean()), float(logit_est.std() + 1e-6)
    Hw = acts_est[("resid", lw)].numpy().astype(np.float64)
    proj = (Hw - z["E_center_L%d" % lw]) @ z["E_L%d" % lw]
    half_gap = float((proj[y_est == 1].mean() - proj[y_est == 0].mean()) / 2)

    # ---- evaluation items ------------------------------------------------------------------------- #
    recs = []
    per_ds = exp.get("eval_limit_per_dataset")
    for ds in exp["eval_datasets"]:
        r = load_split(ds, exp.get("eval_split", "test"))
        recs += r[:per_ds] if per_ds else r
    if args.limit:
        recs = recs[: args.limit]
    labels = np.array([bool(r["label"]) for r in recs])
    texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in recs]
    lab_map = build_label_token_map(bundle.tokenizer, texts[0], prompts["statement"]["label_variants"])
    enc_batches = [bundle.encode(texts[i:i + bs]) for i in range(0, len(texts), bs)]
    slices = [slice(i, min(i + bs, len(texts))) for i in range(0, len(texts), bs)]
    acts_ev, _ = capture_last_token(bundle, texts, [lr], components=("resid",), batch_size=bs)
    s = np.clip(((acts_ev[("resid", lr)].numpy().astype(np.float64) @ fit["w"] + fit["b"]) - mu) / sd, -3.0, 3.0)
    t0 = time.time()
    base_p, base_pred = run(bundle, enc_batches, lab_map, None)
    base_pred = np.asarray(base_pred)
    pb = np.clip(np.nan_to_num(base_p, nan=0.5), 1e-6, 1 - 1e-6)
    logit_out = np.log(pb / (1 - pb))
    u = 1.0 - np.abs(2 * pb - 1)                      # output uncertainty
    d = np.sign(s) != np.sign(logit_out)              # probe vs mouth disagree
    print("E10 %s: n=%d | probe L%d AUROC on eval %.3f | output AUROC %.3f | disagreeing items %d (%.1f%%) | half-gap L%d %.2f" % (
        mcfg["name"], len(recs), lr, auroc(s, labels), auroc(base_p, labels), int(d.sum()), 100 * d.mean(), lw, half_gap), flush=True)
    sb = state(base_pred, labels)
    print("  disagreement among baseline-known %.1f%%, baseline-wrong %.1f%%" % (100 * d[sb == "known"].mean(), 100 * d[sb == "wrong"].mean()), flush=True)

    def amp_specs(alpha_vec):
        def mk(bi):
            a = torch.tensor(alpha_vec[slices[bi]], dtype=torch.float32)
            return [SteerSpec(z["E_L%d" % l], l, mode="amplify", alpha=a, center=z["E_center_L%d" % l]) for l in args.e_layers]
        return mk

    def transfer_specs(gate_vec, g=2.0, extra=None):
        tgt = g * half_gap * s

        def mk(bi):
            a = torch.tensor(gate_vec[slices[bi]], dtype=torch.float32)
            specs = [SteerSpec(z["E_L%d" % lw], lw, mode="clamp", alpha=a, z_target=torch.tensor(tgt[slices[bi]], dtype=torch.float32))]
            if extra is not None:
                specs += extra(bi)
            return specs
        return mk

    def c_specs(alpha_vec):
        def mk(bi):
            a = torch.tensor(-alpha_vec[slices[bi]], dtype=torch.float32)   # negative = toward abstention
            return [SteerSpec(z["C_L%d" % cL], cL, mode="add", alpha=a, coef=[e1["rows"][str(cL)]["gap_C"]])]
        return mk

    ones = np.ones(len(recs))
    conds = [("global_a3", amp_specs(3.0 * ones))]
    for A in (3.0, 5.0, 8.0):
        conds.append(("soft_unc_a%g" % A, amp_specs(A * u)))
    for T in (0.5, 0.8):
        conds.append(("hard_unc_t%g_a3" % T, amp_specs(np.where(u > T, 3.0, 0.0))))
    for A in (3.0, 5.0):
        conds.append(("disagree_a%g" % A, amp_specs(np.where(d, A, 0.0))))
    conds.append(("agree_only_a3", amp_specs(np.where(~d, 3.0, 0.0))))
    for T in (1.0, 2.0):
        conds.append(("probeconf_t%g_a3" % T, amp_specs(np.where(np.abs(s) > T, 3.0, 0.0))))
    conds.append(("transfer_global_g2", transfer_specs(ones)))
    conds.append(("transfer_disagree_g2", transfer_specs(d.astype(float))))
    conds.append(("transfer_probeconf_t1_g2", transfer_specs((np.abs(s) > 1.0).astype(float))))
    conds.append(("combo_disagreeE_uncC_a1", transfer_specs(d.astype(float), extra=c_specs(np.where(np.abs(s) < 1.0, 1.0, 0.0)))))
    conds.append(("C_global_a1", c_specs(ones)))
    cautious_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True, system=prompts["commitment_contrast"]["cautious_system"]) for r in recs]
    cautious_batches = [bundle.encode(cautious_texts[i:i + bs]) for i in range(0, len(texts), bs)]

    results = {"model": mcfg["name"], "n": len(recs), "probe_layer": lr, "write_layer": lw, "e_layers": args.e_layers,
               "probe_eval_auc": float(auroc(s, labels)), "n_disagree": int(d.sum()), "conditions": {}}
    results["conditions"]["baseline"] = summarize(labels, base_p, base_pred, base_pred, base_p, rng, 1)
    per_item = {"baseline": (base_p, base_pred)}
    print("  baseline: correct %d wrong %d abstain %d | AUROC %.3f J %.3f" % (results["conditions"]["baseline"]["n_correct"], results["conditions"]["baseline"]["n_wrong"],
                                                                             results["conditions"]["baseline"]["n_abstain"], base_p.size and auroc(base_p, labels), results["conditions"]["baseline"]["J"]), flush=True)
    for i, (name, mk) in enumerate(conds + [("prompt_only_cautious", None)]):
        if name == "prompt_only_cautious":
            p, pred = run(bundle, cautious_batches, lab_map, None)
        else:
            p, pred = run(bundle, enc_batches, lab_map, mk)
        pred = np.asarray(pred)
        per_item[name] = (p, pred)
        m = summarize(labels, p, pred, base_pred, base_p, rng)
        results["conditions"][name] = m
        print("  [%2d] %-26s fixed %3d  damaged %3d (known dmg %.1f%%)  wrong->abst %3d  known->abst %3d | correct %4d wrong %3d abst %3d | AUROC %+.3f  J %.3f  B %.3f  ans %.3f  (%.0fs)" % (
            i + 1, name, m["corrected"], m["damaged"], 100 * m["known_damage_rate"], m["wrong_to_abstain"], m["known_to_abstain"], m["n_correct"], m["n_wrong"], m["n_abstain"],
            m["auroc"] - results["conditions"]["baseline"]["auroc"], m["J"], m["B"], m["answer_rate"], time.time() - t0), flush=True)

    out_dir = os.path.join(RESULTS_DIR, "E10", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    stem = "gating" + (("_" + args.tag) if args.tag else "")
    results["elapsed_s"] = time.time() - t0
    atomic_json_dump(results, os.path.join(out_dir, stem + ".json"))
    np.savez(os.path.join(out_dir, stem + ".npz"), labels=labels, ids=np.array([r["id"] for r in recs]), probe_s=s, unc=u, disagree=d,
             **{"%s__p" % k: v[0] for k, v in per_item.items()}, **{"%s__pred" % k: v[1] for k, v in per_item.items()})
    b = results["conditions"]["baseline"]
    lines = ["# E10 gating on %s (n=%d; baseline correct %d / wrong %d / abstain %d; probe L%d eval AUROC %.3f; %d disagreeing items)" % (
        mcfg["name"], len(recs), b["n_correct"], b["n_wrong"], b["n_abstain"], lr, results["probe_eval_auc"], int(d.sum())), "",
        "| condition | fixed (wrong->right) | damaged (right->wrong) | known damage % | wrong->abstain | known->abstain | correct / wrong / abstain | acc all | wrong all | AUROC d | J | B | answer rate | penalised |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in results["conditions"].items():
        lines.append("| %s | %d | %d | %.1f | %d | %d | %d / %d / %d | %.3f | %.3f | %+.3f | %.3f | %.3f | %.3f | %+.3f |" % (
            k, m["corrected"], m["damaged"], 100 * m["known_damage_rate"], m["wrong_to_abstain"], m["known_to_abstain"], m["n_correct"], m["n_wrong"], m["n_abstain"],
            m["acc_all"], m["wrong_all"], m["auroc"] - b["auroc"], m["J"], m["B"], m["answer_rate"], m["penalised"]))
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(out_dir, stem + ".md"))


if __name__ == "__main__":
    main()

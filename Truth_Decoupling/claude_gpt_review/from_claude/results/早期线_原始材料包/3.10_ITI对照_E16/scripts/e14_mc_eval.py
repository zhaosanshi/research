#!/usr/bin/env python3
"""E14: question-level accuracy on TruthfulQA (MC1 / MC2 style) under the same writes as E11, using the statement readout.

Every answer of every question is scored as a statement ("The answer to the question ... is: <answer>") by P(True | answered)
from the unsteered or steered pass. Per question:
  MC1        the best answer out-scores every incorrect answer (TruthfulQA / ITI convention)
  argmax acc the top-scoring candidate among all answers is a correct one
  MC2        normalised P(True) mass on the correct answers
Questions whose pairs were used for direction / probe estimation (truthfulqa_tf calib+dev) are dropped.
Conditions: baseline, E_a{alpha} (amplify at --e-layers), gate_disagree_a5, gate_disagree_soft_a5, C_abst_a{c}, prompt_only_cautious.

Usage: python3 scripts/e14_mc_eval.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --e-layers 12 13 14 --probe-layer 14
Output: results/E14/<model>/mc_eval[_tag].{json,md,npz}
"""
import argparse
import os
import re
import time

from common import RESULTS_DIR, load_split, tf_prompt
from e2b_h1_rescue import run
from e10_gating import state

import numpy as np
import torch

from core.config import load_experiment, load_model_config
from core.directions import fit_logreg
from core.intervene import SteerSpec
from core.io_utils import atomic_json_dump, load_json, set_seed
from core.metrics import auroc
from core.model_io import capture_last_token, load_model
from core.readout import build_label_token_map


def question_metrics(p, pred, labels, roles, groups):
    """groups: list of index arrays, one per question. Returns per-question MC1 / argmax / MC2 arrays."""
    mc1, amx, mc2, abst = [], [], [], []
    for g in groups:
        pg, lg, rg = p[g], labels[g], roles[g]
        best = g[rg == "best"]
        inc = pg[~lg]
        mc1.append(bool(len(best) and (len(inc) == 0 or p[best[0]] > inc.max())))
        amx.append(bool(lg[int(np.argmax(pg))]))
        tot = float(pg.sum())
        mc2.append(float(pg[lg].sum() / tot) if tot > 0 else float("nan"))
        abst.append(float((np.asarray(pred)[g] == "Abstain").mean()))
    return np.array(mc1), np.array(amx), np.array(mc2), np.array(abst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa")
    ap.add_argument("--e1-exp", default="E1_directions_tqa")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mc-dataset", default="truthfulqa_mc")
    ap.add_argument("--exclude-dataset", default="truthfulqa_tf")
    ap.add_argument("--e-layers", nargs="+", type=int, default=[12, 13, 14])
    ap.add_argument("--probe-layer", type=int, default=14)
    ap.add_argument("--alphas", nargs="+", type=float, default=[1, 2, 3])
    ap.add_argument("--c-alpha", type=float, default=1.0)
    ap.add_argument("--write-layer", type=int, default=None, help="transfer (KAPPA-style clamp) write layer; omitted = no transfer conditions")
    ap.add_argument("--transfer-g", type=float, default=2.0)
    ap.add_argument("--conditions", default=None, help="regex; only conditions whose name matches are run (baseline always)")
    ap.add_argument("--limit-questions", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    exp = load_experiment(args.exp)
    if args.model:
        exp["model"] = load_model_config(args.model)
    mcfg, prompts = exp["model"], exp["prompts"]
    seed = exp.get("seed", 42)
    set_seed(seed)
    bs = exp.get("batch_size", 8)
    e1_dir = os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    e1 = load_json(os.path.join(e1_dir, "summary.json"))
    z = np.load(os.path.join(e1_dir, "directions_meandiff.npz"))
    cL = e1["C_write_layer"]
    gap_C = e1["rows"][str(cL)]["gap_C"]
    bundle = load_model(mcfg)
    e_layers, lp = list(args.e_layers), args.probe_layer

    e1_cfg = load_experiment(args.e1_exp)
    est = []
    for ds in e1_cfg["e_datasets"]:
        for sp in e1_cfg.get("direction_splits", ["calib", "dev"]):
            est += load_split(ds, sp)
    y_est = np.array([1 if r["label"] else 0 for r in est])
    excl = set(r.get("pair_id") or r["id"] for sp in ("calib", "dev") for r in load_split(args.exclude_dataset, sp))
    lw = args.write_layer
    cap_layers = sorted(set([lp] + ([lw] if lw is not None else [])))
    acts_est, _ = capture_last_token(bundle, [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est], cap_layers, components=("resid",), batch_size=bs)
    fit = fit_logreg(acts_est[("resid", lp)].numpy().astype(np.float64), y_est, l2=1.0, standardize=True)
    logit_est = acts_est[("resid", lp)].numpy().astype(np.float64) @ fit["w"] + fit["b"]
    mu, sd = float(logit_est.mean()), float(logit_est.std() + 1e-6)
    half_gap = None
    if lw is not None:                                    # transfer target scale, as in E10 / E6
        Hw = acts_est[("resid", lw)].numpy().astype(np.float64)
        proj = (Hw - z["E_center_L%d" % lw]) @ z["E_L%d" % lw]
        half_gap = float((proj[y_est == 1].mean() - proj[y_est == 0].mean()) / 2)

    recs = [r for r in load_split(args.mc_dataset, "test") if r["pair_id"] not in excl]
    qids = []
    for r in recs:
        if r["pair_id"] not in qids:
            qids.append(r["pair_id"])
    if args.limit_questions:
        qids = qids[: args.limit_questions]
        recs = [r for r in recs if r["pair_id"] in set(qids)]
    labels = np.array([bool(r["label"]) for r in recs])
    roles = np.array([r["meta"]["role"] for r in recs])
    qarr = np.array([r["pair_id"] for r in recs])
    groups = [np.where(qarr == q)[0] for q in qids]
    texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in recs]
    lab_map = build_label_token_map(bundle.tokenizer, texts[0], prompts["statement"]["label_variants"])
    enc_batches = [bundle.encode(texts[i:i + bs]) for i in range(0, len(texts), bs)]
    slices = [slice(i, min(i + bs, len(texts))) for i in range(0, len(texts), bs)]
    acts_ev, _ = capture_last_token(bundle, texts, [lp], components=("resid",), batch_size=bs)
    s = np.clip(((acts_ev[("resid", lp)].numpy().astype(np.float64) @ fit["w"] + fit["b"]) - mu) / sd, -3.0, 3.0)
    t0 = time.time()
    base_p, base_pred = run(bundle, enc_batches, lab_map, None)
    base_pred = np.asarray(base_pred)
    pb = np.clip(np.nan_to_num(base_p, nan=0.5), 1e-6, 1 - 1e-6)
    d = np.sign(s) != np.sign(np.log(pb / (1 - pb)))
    ones = np.ones(len(recs))
    print("E14 %s: %d questions (%d excluded as estimation questions), %d candidate statements | probe L%d AUROC %.3f | statement AUROC %.3f" % (
        mcfg["name"], len(qids), len(excl), len(recs), lp, auroc(s, labels), auroc(base_p, labels)), flush=True)

    def amp(alpha_vec):
        def mk(bi):
            a = torch.tensor(np.asarray(alpha_vec)[slices[bi]], dtype=torch.float32)
            return [SteerSpec(z["E_L%d" % l], l, mode="amplify", alpha=a, center=z["E_center_L%d" % l]) for l in e_layers]
        return mk

    def c_add(alpha):
        return lambda bi: [SteerSpec(z["C_L%d" % cL], cL, mode="add", alpha=-float(alpha), coef=[gap_C])]

    conds = [("baseline", None)] + [("E_a%g" % a, amp(a * ones)) for a in args.alphas]
    conds += [("gate_disagree_a5", amp(np.where(d, 5.0, 0.0))), ("gate_disagree_soft_a5", amp(np.where(d, 5.0 * np.minimum(np.abs(s), 2.0) / 2.0, 0.0))),
              ("C_abst_a%g" % args.c_alpha, c_add(args.c_alpha))]
    if lw is not None:
        tgt = args.transfer_g * half_gap * s

        def transfer(gate_vec):
            def mk(bi):
                a = torch.tensor(np.asarray(gate_vec)[slices[bi]], dtype=torch.float32)
                return [SteerSpec(z["E_L%d" % lw], lw, mode="clamp", alpha=a, z_target=torch.tensor(tgt[slices[bi]], dtype=torch.float32))]
            return mk
        conds += [("transfer_global_g%g" % args.transfer_g, transfer(ones)), ("transfer_disagree_g%g" % args.transfer_g, transfer(d.astype(float)))]
    cautious_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True, system=prompts["commitment_contrast"]["cautious_system"]) for r in recs]
    cautious_batches = [bundle.encode(cautious_texts[i:i + bs]) for i in range(0, len(texts), bs)]
    conds.append(("prompt_only_cautious", "cautious"))
    if args.conditions:
        conds = [(n, m) for (n, m) in conds if n == "baseline" or re.search(args.conditions, n)]
        print("  conditions after filter: %s" % [n for n, _ in conds], flush=True)

    results = {"model": mcfg["name"], "n_questions": len(qids), "n_statements": len(recs), "e_layers": e_layers, "probe_layer": lp,
               "write_layer": lw, "half_gap": half_gap, "mc_dataset": args.mc_dataset, "conditions": {}}
    per_item, base_q = {}, None
    for i, (name, mk) in enumerate(conds):
        if mk is None:
            p, pred = base_p, base_pred
        elif mk == "cautious":
            p, pred = run(bundle, cautious_batches, lab_map, None)
        else:
            p, pred = run(bundle, enc_batches, lab_map, mk)
        pred = np.asarray(pred)
        p = np.nan_to_num(np.asarray(p, dtype=float), nan=0.5)
        per_item[name] = (p, pred)
        mc1, amx, mc2, abst = question_metrics(p, pred, labels, roles, groups)
        if base_q is None:
            base_q = (mc1, amx)
        st = state(pred, labels)
        m = {"mc1": float(mc1.mean()), "argmax_acc": float(amx.mean()), "mc2": float(np.nanmean(mc2)), "cand_abstain_rate": float(abst.mean()),
             "statement_auroc": float(auroc(p, labels)), "statement_acc_all": float((st == "known").mean()),
             "mc1_fixed": int((~base_q[0] & mc1).sum()), "mc1_damaged": int((base_q[0] & ~mc1).sum()),
             "argmax_fixed": int((~base_q[1] & amx).sum()), "argmax_damaged": int((base_q[1] & ~amx).sum())}
        results["conditions"][name] = m
        print("  [%d/%d] %-22s MC1 %.3f  argmax %.3f  MC2 %.3f | MC1 fixed %3d damaged %3d | stmt AUROC %.3f acc %.3f | cand abstain %.3f (%.0fs)" % (
            i + 1, len(conds), name, m["mc1"], m["argmax_acc"], m["mc2"], m["mc1_fixed"], m["mc1_damaged"], m["statement_auroc"], m["statement_acc_all"], m["cand_abstain_rate"], time.time() - t0), flush=True)
    results["elapsed_s"] = time.time() - t0
    out_dir = os.path.join(RESULTS_DIR, "E14", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "mc_eval" + (("_" + args.tag) if args.tag else ""))
    atomic_json_dump(results, stem + ".json")
    np.savez(stem + ".npz", ids=np.array([r["id"] for r in recs]), qids=qarr, labels=labels, roles=roles, probe_s=s, disagree=d,
             **{"%s__p" % k: v[0] for k, v in per_item.items()}, **{"%s__pred" % k: v[1] for k, v in per_item.items()})
    lines = ["# E14 %s question-level accuracy with the statement readout: %s (%d questions, %d candidates; estimation questions excluded; E layers %s, probe L%d%s)" % (
        args.mc_dataset, mcfg["name"], len(qids), len(recs), e_layers, lp, (", transfer write L%d" % lw) if lw is not None else ""), "",
             "| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |", "|---|---|---|---|---|---|---|---|---|"]
    for k, m in results["conditions"].items():
        lines.append("| %s | %.3f | %.3f | %.3f | %d | %d | %.3f | %.3f | %.3f |" % (k, m["mc1"], m["argmax_acc"], m["mc2"], m["mc1_fixed"], m["mc1_damaged"], m["statement_auroc"], m["statement_acc_all"], m["cand_abstain_rate"]))
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()

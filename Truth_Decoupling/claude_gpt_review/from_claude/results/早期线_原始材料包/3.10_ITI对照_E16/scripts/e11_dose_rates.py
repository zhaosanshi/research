#!/usr/bin/env python3
"""E11: the teacher's steps 2-5 (and the gap-guided variant of step 9) in one run per dataset, with per-item outputs.

Same model, fixed layers, one evaluation set. Every condition reports, over ALL items: correct / wrong / abstain rates,
the four transitions vs the unsteered pass (wrong->right, right->wrong, wrong->abstain, right->abstain), the known-item
damage rate, AUROC, J, B, answer rate and the penalised score (correct - wrong)/n. Items are tagged half A / half B by
pair hash (as in E8/E10) so a strength can be chosen on A and reported on B.

Conditions
  baseline
  E_a{alpha}            amplify along the mean-difference evidence direction at --e-layers, alpha from negative (erase) to positive
  C_abst_a{alpha}       commitment direction pushed toward abstention (add at the C write layer); C_answer_a1 = pushed the other way
  E_a3+C_abst_a{alpha}  the two writes combined (step 4), to be compared with the singles at matched answer rate
  random{k}_matched_a3  Haar-random directions at the same layers, alpha rescaled so the mean perturbation norm on the
                        estimation statements equals that of E_a3 (step 5: "same perturbation magnitude")
  prompt_only_cautious  system-prompt swap, no steering (step 5: "just tell it to say I don't know")
  gate_disagree_a5      alpha_i = 5 where the clean L--probe-layer probe and the output disagree (E10's best gate, step 9)
  gate_disagree_soft_a5 alpha_i = 5 * min(|s_i|, 2)/2 on disagreeing items: strength guided by the probe's confidence (step 9)

Usage: python3 scripts/e11_dose_rates.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --e-layers 12 13 14
Output: results/E11/<model>/dose_rates[_tag].{json,md,npz}
"""
import argparse
import hashlib
import os
import re
import time

from common import RESULTS_DIR, load_split, tf_prompt
from e2b_h1_rescue import run
from e10_gating import state

import numpy as np
import torch

from core.config import load_experiment, load_model_config, load_prompt_config
from core.directions import fit_logreg, mean_diff_direction, random_direction
from core.intervene import SteerSpec
from core.io_utils import atomic_json_dump, load_json, set_seed
from core.metrics import auroc, confusion_tf
from core.model_io import capture_last_token, load_model
from core.readout import build_label_token_map


def rates(labels, p, pred, base_pred, base_p, m):
    lab, pr, bp = labels[m], np.asarray(pred)[m], np.asarray(base_pred)[m]
    sb, sc = state(bp, lab), state(pr, lab)
    conf = confusion_tf(lab, list(pr))
    n = len(lab)
    n_known, n_wrong = max(1, int((sb == "known").sum())), max(1, int((sb == "wrong").sum()))
    t = lambda a, b: int(((sb == a) & (sc == b)).sum())
    return {"n": n, "correct": float((sc == "known").mean()), "wrong": float((sc == "wrong").mean()), "abstain": float((sc == "abstain").mean()),
            "penalised": float(((sc == "known").sum() - (sc == "wrong").sum()) / n),
            "fixed": t("wrong", "known"), "damaged": t("known", "wrong"), "wrong_to_abstain": t("wrong", "abstain"), "known_to_abstain": t("known", "abstain"),
            "abstain_to_known": t("abstain", "known"), "abstain_to_wrong": t("abstain", "wrong"),
            "known_damage_rate": t("known", "wrong") / n_known, "fix_rate": t("wrong", "known") / n_wrong,
            "auroc": float(auroc(p[m], lab)), "d_auroc": float(auroc(p[m], lab) - auroc(base_p[m], lab)),
            "J": conf["J"], "B": conf["B"], "answer_rate": conf["answer_rate"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--e1-exp", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--e-layers", nargs="+", type=int, required=True)
    ap.add_argument("--c-layer", type=int, default=None, help="default: E1 summary C_write_layer")
    ap.add_argument("--alphas", nargs="+", type=float, default=[-1.5, -1, -0.5, 0.5, 1, 2, 3, 4, 5])
    ap.add_argument("--c-alphas", nargs="+", type=float, default=[0.5, 1, 1.5, 2])
    ap.add_argument("--combo-c-alphas", nargs="+", type=float, default=[0.5, 1])
    ap.add_argument("--combo-e-alpha", type=float, default=3.0)
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--probe-layer", type=int, default=14)
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompts", default=None, help="prompt config name overriding the experiment's (teacher step 8: wording robustness)")
    ap.add_argument("--conditions", default=None, help="regex; only conditions whose name matches are run (baseline always)")
    ap.add_argument("--reestimate-seed", type=int, default=None, help="re-estimate the E direction on a random subset of estimation pairs")
    ap.add_argument("--reestimate-frac", type=float, default=0.67)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    exp = load_experiment(args.exp)
    if args.model:
        exp["model"] = load_model_config(args.model)
    if args.prompts:
        exp["prompts"] = load_prompt_config(args.prompts)
    mcfg, prompts = exp["model"], exp["prompts"]
    seed = exp.get("seed", 42)
    set_seed(seed)
    bs = exp.get("batch_size", 8)
    e1_dir = os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    e1 = load_json(os.path.join(e1_dir, "summary.json"))
    z = np.load(os.path.join(e1_dir, "directions_meandiff.npz"))
    cL = args.c_layer if args.c_layer is not None else e1["C_write_layer"]
    gap_C = e1["rows"][str(cL)]["gap_C"]
    bundle = load_model(mcfg)
    e_layers = list(args.e_layers)
    lp = args.probe_layer

    # estimation statements: probe for the gates + scale matching for the random controls (never evaluation items)
    e1_cfg = load_experiment(args.e1_exp)
    est = []
    for ds in e1_cfg["e_datasets"]:
        for sp in e1_cfg.get("direction_splits", ["calib", "dev"]):
            est += load_split(ds, sp)
    y_est = np.array([1 if r["label"] else 0 for r in est])
    cap_layers = sorted(set(e_layers + [lp]))
    acts_est, _ = capture_last_token(bundle, [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est], cap_layers, components=("resid",), batch_size=bs)
    fit = fit_logreg(acts_est[("resid", lp)].numpy().astype(np.float64), y_est, l2=1.0, standardize=True)
    logit_est = acts_est[("resid", lp)].numpy().astype(np.float64) @ fit["w"] + fit["b"]
    mu, sd = float(logit_est.mean()), float(logit_est.std() + 1e-6)
    E_dir = {l: np.asarray(z["E_L%d" % l], dtype=np.float32).reshape(-1) for l in e_layers}
    E_ctr = {l: np.asarray(z["E_center_L%d" % l], dtype=np.float32).reshape(-1) for l in e_layers}
    if args.reestimate_seed is not None:                 # direction-estimation seed (teacher step 8)
        rng_r = np.random.default_rng(args.reestimate_seed)
        pairs = {}
        for i, r in enumerate(est):
            pairs.setdefault(r.get("pair_id") or r["id"], []).append(i)
        keys = sorted(pairs)
        rng_r.shuffle(keys)
        keep = keys[: max(2, int(round(args.reestimate_frac * len(keys))))]
        idx = np.array(sorted(i for k in keep for i in pairs[k]))
        for l in e_layers:
            H = acts_est[("resid", l)].numpy().astype(np.float64)[idx]
            yy = y_est[idx]
            w, _ = mean_diff_direction(H[yy == 1], H[yy == 0])
            E_dir[l] = np.asarray(w, dtype=np.float32).reshape(-1)
            E_ctr[l] = H.mean(0).astype(np.float32)
        print("  E re-estimated on %d/%d estimation pairs (seed %d); cos to full-set E: %s" % (
            len(keep), len(keys), args.reestimate_seed, {l: round(float(E_dir[l] @ np.asarray(z["E_L%d" % l]).reshape(-1)), 3) for l in e_layers}), flush=True)

    # evaluation items
    recs = []
    per_ds = exp.get("eval_limit_per_dataset")
    for ds in exp["eval_datasets"]:
        r = load_split(ds, exp.get("eval_split", "test"))
        recs += r[:per_ds] if per_ds else r
    if args.limit:
        recs = recs[: args.limit]
    labels = np.array([bool(r["label"]) for r in recs])
    pair_ids = [r.get("pair_id") or r["id"] for r in recs]
    half_a = np.array([int(hashlib.sha1(str(p).encode()).hexdigest(), 16) % 2 == 0 for p in pair_ids])
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
    print("E11 %s: n=%d (half A %d / half B %d) | E layers %s | C layer %d (gap_C %.3f) | probe L%d AUROC %.3f | output AUROC %.3f | disagreeing %d" % (
        mcfg["name"], len(recs), int(half_a.sum()), int((~half_a).sum()), e_layers, cL, gap_C, lp, auroc(s, labels), auroc(base_p, labels), int(d.sum())), flush=True)

    def amp(alpha_vec):
        def mk(bi):
            a = torch.tensor(np.asarray(alpha_vec)[slices[bi]], dtype=torch.float32)
            return [SteerSpec(E_dir[l], l, mode="amplify", alpha=a, center=E_ctr[l]) for l in e_layers]
        return mk

    def c_add(alpha):                                   # positive alpha = toward abstention (E10 convention)
        def mk(bi):
            return [SteerSpec(z["C_L%d" % cL], cL, mode="add", alpha=-float(alpha), coef=[gap_C])]
        return mk

    def combo(e_alpha, c_alpha):
        fe, fc = amp(e_alpha * ones), c_add(c_alpha)
        return lambda bi: fe(bi) + fc(bi)

    random_scales = {}

    def rand(k, alpha):
        static = []
        for l in e_layers:
            r = random_direction(bundle.hidden_size, 1, seed=seed + 100 * k + l).squeeze(1).numpy().astype(np.float64)
            H = acts_est[("resid", l)].numpy().astype(np.float64)
            c = E_ctr[l].astype(np.float64)
            pe = np.abs((H - c) @ E_dir[l].astype(np.float64)).mean()
            pr = np.abs((H - c) @ r).mean()
            static.append((r.astype(np.float32), l, float(alpha * pe / max(pr, 1e-8)), c.astype(np.float32)))
        random_scales["random%d" % k] = {("L%d" % l): a for (_, l, a, _) in static}
        return lambda bi: [SteerSpec(r, l, mode="amplify", alpha=a, center=c) for (r, l, a, c) in static]

    conds = [("baseline", None)]
    for a in args.alphas:
        conds.append(("E_a%g" % a, amp(a * ones)))
    for a in args.c_alphas:
        conds.append(("C_abst_a%g" % a, c_add(a)))
    conds.append(("C_answer_a1", c_add(-1.0)))
    for a in args.combo_c_alphas:
        conds.append(("E_a%g+C_abst_a%g" % (args.combo_e_alpha, a), combo(args.combo_e_alpha, a)))
    for k in range(args.random_seeds):
        conds.append(("random%d_matched_a3" % k, rand(k, 3.0)))
    if not args.no_gates:
        conds.append(("gate_disagree_a5", amp(np.where(d, 5.0, 0.0))))
        conds.append(("gate_disagree_soft_a5", amp(np.where(d, 5.0 * np.minimum(np.abs(s), 2.0) / 2.0, 0.0))))
    cautious_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True, system=prompts["commitment_contrast"]["cautious_system"]) for r in recs]
    cautious_batches = [bundle.encode(cautious_texts[i:i + bs]) for i in range(0, len(texts), bs)]
    conds.append(("prompt_only_cautious", "cautious"))
    if args.conditions:
        conds = [(n, m) for (n, m) in conds if n == "baseline" or re.search(args.conditions, n)]
        print("  conditions after filter: %s" % [n for n, _ in conds], flush=True)

    masks = {"ALL": np.ones(len(recs), dtype=bool), "halfA": half_a, "halfB": ~half_a}
    results = {"model": mcfg["name"], "exp": args.exp, "n": len(recs), "e_layers": e_layers, "c_layer": cL, "probe_layer": lp, "probe_eval_auc": float(auroc(s, labels)),
               "n_disagree": int(d.sum()), "random_alpha_scales": random_scales, "conditions": {}}
    per_item = {}
    for i, (name, mk) in enumerate(conds):
        if mk is None:
            p, pred = base_p, base_pred
        elif mk == "cautious":
            p, pred = run(bundle, cautious_batches, lab_map, None)
        else:
            p, pred = run(bundle, enc_batches, lab_map, mk)
        pred = np.asarray(pred)
        per_item[name] = (p, pred)
        results["conditions"][name] = {g: rates(labels, p, pred, base_pred, base_p, m) for g, m in masks.items()}
        r = results["conditions"][name]["ALL"]
        print("  [%2d/%d] %-24s correct %.3f wrong %.3f abstain %.3f | fixed %3d damaged %3d (known dmg %4.1f%%) w->abst %3d k->abst %3d | AUROC %+.3f J %.3f B %.3f ans %.3f (%.0fs)" % (
            i + 1, len(conds), name, r["correct"], r["wrong"], r["abstain"], r["fixed"], r["damaged"], 100 * r["known_damage_rate"], r["wrong_to_abstain"], r["known_to_abstain"],
            r["d_auroc"], r["J"], r["B"], r["answer_rate"], time.time() - t0), flush=True)

    # strength chosen on half A (penalised score), reported on half B
    e_names = [n for n, _ in conds if n.startswith("E_a")] or ["baseline"]
    best_a = max(e_names, key=lambda n: results["conditions"][n]["halfA"]["penalised"])
    results["selection"] = {"criterion": "penalised on half A", "chosen": best_a, "halfA": results["conditions"][best_a]["halfA"]["penalised"],
                            "halfB": results["conditions"][best_a]["halfB"]["penalised"], "halfB_baseline": results["conditions"]["baseline"]["halfB"]["penalised"]}
    results["prompts"] = args.prompts or exp["prompts"].get("name", "default")
    results["reestimate_seed"] = args.reestimate_seed
    results["elapsed_s"] = time.time() - t0

    out_dir = os.path.join(RESULTS_DIR, "E11", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "dose_rates" + (("_" + args.tag) if args.tag else ""))
    atomic_json_dump(results, stem + ".json")
    np.savez(stem + ".npz", ids=np.array([r["id"] for r in recs]), labels=labels, pair_ids=np.array(pair_ids), half_a=half_a, probe_s=s, disagree=d,
             **{"%s__p" % k: v[0] for k, v in per_item.items()}, **{"%s__pred" % k: v[1] for k, v in per_item.items()})
    hdr = "| condition | correct | wrong | abstain | fixed | damaged | known dmg % | wrong->abst | known->abst | penalised | dAUROC | J | B | answer rate |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"

    def table(g):
        out = [hdr, sep]
        for name, _ in conds:
            r = results["conditions"][name][g]
            out.append("| %s | %.3f | %.3f | %.3f | %d | %d | %.1f | %d | %d | %+.3f | %+.3f | %.3f | %.3f | %.3f |" % (
                name, r["correct"], r["wrong"], r["abstain"], r["fixed"], r["damaged"], 100 * r["known_damage_rate"], r["wrong_to_abstain"], r["known_to_abstain"],
                r["penalised"], r["d_auroc"], r["J"], r["B"], r["answer_rate"]))
        return out
    lines = ["# E11 dose / rates / controls: %s (n=%d; E layers %s; C layer %d; probe L%d clean AUROC %.3f; %d disagreeing items)" % (
        mcfg["name"], len(recs), e_layers, cL, lp, results["probe_eval_auc"], int(d.sum())), "",
        "Random-direction alpha rescaling (perturbation norm matched to E_a3 on the estimation statements): %s" % random_scales, "",
        "## All items", ""] + table("ALL") + ["", "## Half B only (strength chosen on half A by penalised score: %s; B penalised %+.3f vs baseline %+.3f)" % (
            best_a, results["selection"]["halfB"], results["selection"]["halfB_baseline"]), ""] + table("halfB")
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()

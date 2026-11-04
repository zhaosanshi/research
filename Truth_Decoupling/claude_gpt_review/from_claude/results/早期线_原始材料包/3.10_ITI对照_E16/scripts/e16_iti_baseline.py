#!/usr/bin/env python3
"""E16: ITI (Li et al., NeurIPS 2023) under OUR protocol, as the published-method baseline the teacher's step 5 asks for.

Same estimation statements as the evidence direction (calib+dev of the regime's TF set), same evaluation items, same
readout (True/False/Unknown first-token log-probs), same all-item metrics as E11 (correct / wrong / abstain, fixed /
damaged, penalised, AUROC) with the same pair-hash half A / half B. Optionally the TruthfulQA MC1 / MC2 evaluation of E14.

  heads     K heads with the highest grouped-CV probe accuracy on the estimation statements (ITI's selection rule)
  shift     alpha * sigma_h * unit mass-mean-shift at the last position of every forward (ITI's intervention)
  --train-on-half-a   also use the test half-A statements for selection and directions (ITI's own data budget is far
                      larger than our 96 pairs); then only half B is a clean report.

Usage: python3 scripts/e16_iti_baseline.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --k 48 --alphas 5 10 15 20 --mc
Output: results/E16/<model>/iti[_tag].{json,md,npz}, iti_heads[_tag].npz (reused by e4b for generation)
"""
import argparse
import hashlib
import os
import time

from common import RESULTS_DIR, load_split, tf_prompt
from e2b_h1_rescue import run
from e11_dose_rates import rates
from e14_mc_eval import question_metrics
from e10_gating import state

import numpy as np

from core.config import load_experiment, load_model_config
from core.io_utils import atomic_json_dump, set_seed
from core.iti import ITISpec, capture_heads, head_directions, head_geometry, iti_steer, select_heads
from core.metrics import auroc
from core.model_io import load_model
from core.readout import build_label_token_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--e1-exp", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--alphas", nargs="+", type=float, default=[5, 10, 15, 20])
    ap.add_argument("--train-on-half-a", action="store_true")
    ap.add_argument("--mc", action="store_true", help="also run the TruthfulQA MC1/MC2 evaluation (truthfulqa_mc)")
    ap.add_argument("--mc-dataset", default="truthfulqa_mc")
    ap.add_argument("--exclude-dataset", default="truthfulqa_tf")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    exp = load_experiment(args.exp)
    if args.model:
        exp["model"] = load_model_config(args.model)
    mcfg, prompts = exp["model"], exp["prompts"]
    seed = exp.get("seed", 42)
    set_seed(seed)
    bs = exp.get("batch_size", 8)
    bundle = load_model(mcfg)
    n_heads, head_dim = head_geometry(bundle)
    layers = list(range(bundle.num_layers))
    t0 = time.time()

    # evaluation items (E11 convention)
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

    # estimation statements (+ test half A if asked)
    e1_cfg = load_experiment(args.e1_exp)
    est = []
    for ds in e1_cfg["e_datasets"]:
        for sp in e1_cfg.get("direction_splits", ["calib", "dev"]):
            est += load_split(ds, sp)
    est_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est]
    y_est = [1 if r["label"] else 0 for r in est]
    g_est = [str(r.get("pair_id") or r["id"]) for r in est]
    if args.train_on_half_a:
        idx = np.where(half_a)[0]
        est_texts += [texts[i] for i in idx]
        y_est += [int(labels[i]) for i in idx]
        g_est += [str(pair_ids[i]) for i in idx]
    y_est = np.array(y_est)
    acts = capture_heads(bundle, est_texts, layers, batch_size=bs)
    heads, table = select_heads(acts, y_est, g_est, args.k, n_folds=4, l2=1.0, seed=seed)
    dirs, sigmas = head_directions(acts, y_est, heads)
    spec = ITISpec(heads, dirs, sigmas)
    print("E16 %s: %d estimation statements, %d heads x %d layers, head_dim %d | top-%d heads val acc %.3f..%.3f (median over all heads %.3f) | layers of chosen heads: %s (%.0fs)" % (
        mcfg["name"], len(y_est), n_heads, len(layers), head_dim, args.k, heads[-1][2], heads[0][2], float(np.median([a for v in table.values() for a in v])),
        sorted(set(l for l, _, _ in heads)), time.time() - t0), flush=True)
    out_dir = os.path.join(RESULTS_DIR, "E16", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + args.tag) if args.tag else ""
    np.savez(os.path.join(out_dir, "iti_heads%s.npz" % tag), heads=np.array([(l, h) for l, h, _ in heads]), val_acc=np.array([a for _, _, a in heads]),
             directions=dirs, sigmas=sigmas, k=args.k, train_on_half_a=args.train_on_half_a)

    # statement-level evaluation
    lab_map = build_label_token_map(bundle.tokenizer, texts[0], prompts["statement"]["label_variants"])
    enc_batches = [bundle.encode(texts[i:i + bs]) for i in range(0, len(texts), bs)]
    base_p, base_pred = run(bundle, enc_batches, lab_map, None)
    base_pred = np.asarray(base_pred)
    masks = {"ALL": np.ones(len(recs), dtype=bool), "halfA": half_a, "halfB": ~half_a}
    results = {"model": mcfg["name"], "exp": args.exp, "n": len(recs), "k": args.k, "train_on_half_a": args.train_on_half_a,
               "heads": [(int(l), int(h), float(a)) for l, h, a in heads], "conditions": {}}
    per_item = {"baseline": (base_p, base_pred)}
    results["conditions"]["baseline"] = {g: rates(labels, base_p, base_pred, base_pred, base_p, m) for g, m in masks.items()}
    names = ["baseline"]
    for a in args.alphas:
        name = "iti_k%d_a%g" % (args.k, a)
        with iti_steer(bundle, spec, alpha=a):
            p, pred = run(bundle, enc_batches, lab_map, None)
        pred = np.asarray(pred)
        per_item[name] = (p, pred)
        names.append(name)
        results["conditions"][name] = {g: rates(labels, p, pred, base_pred, base_p, m) for g, m in masks.items()}
        r = results["conditions"][name]["ALL"]
        print("  %-16s correct %.3f wrong %.3f abstain %.3f | fixed %3d damaged %3d (known dmg %4.1f%%) | AUROC %+.3f J %.3f B %.3f ans %.3f (%.0fs)" % (
            name, r["correct"], r["wrong"], r["abstain"], r["fixed"], r["damaged"], 100 * r["known_damage_rate"], r["d_auroc"], r["J"], r["B"], r["answer_rate"], time.time() - t0), flush=True)
    rep = "halfB" if args.train_on_half_a else "ALL"
    best = max(names[1:], key=lambda n: results["conditions"][n]["halfA"]["penalised"]) if len(names) > 1 else "baseline"
    results["selection"] = {"criterion": "penalised on half A", "chosen": best, "halfB": results["conditions"][best]["halfB"]["penalised"],
                            "halfB_baseline": results["conditions"]["baseline"]["halfB"]["penalised"]}

    hdr = "| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"

    def table_md(g):
        out = [hdr, sep]
        for n in names:
            r = results["conditions"][n][g]
            out.append("| %s | %.3f | %.3f | %.3f | %d | %d | %.1f | %+.3f | %+.3f | %.3f | %.3f | %.3f |" % (
                n, r["correct"], r["wrong"], r["abstain"], r["fixed"], r["damaged"], 100 * r["known_damage_rate"], r["penalised"], r["d_auroc"], r["J"], r["B"], r["answer_rate"]))
        return out
    lines = ["# E16 ITI baseline, same protocol: %s (n=%d; K=%d heads; %d estimation statements%s; chosen heads' layers %s; val acc %.3f-%.3f)" % (
        mcfg["name"], len(recs), args.k, len(y_est), " incl. test half A" if args.train_on_half_a else "", sorted(set(l for l, _, _ in heads)), heads[-1][2], heads[0][2]), "",
        "## All items%s" % (" (half A used for training: NOT a clean report)" if args.train_on_half_a else ""), ""] + table_md("ALL") + [
        "", "## Half B only (strength chosen on half A by penalised score: %s; B penalised %+.3f vs baseline %+.3f)" % (best, results["selection"]["halfB"], results["selection"]["halfB_baseline"]), ""] + table_md("halfB")

    # question-level (MC) evaluation
    if args.mc:
        excl = set(r.get("pair_id") or r["id"] for sp in ("calib", "dev") for r in load_split(args.exclude_dataset, sp))
        if args.train_on_half_a:
            excl |= set(str(p) for p, a in zip(pair_ids, half_a) if a)
        mrecs = [r for r in load_split(args.mc_dataset, "test") if r["pair_id"] not in excl]
        qids = []
        for r in mrecs:
            if r["pair_id"] not in qids:
                qids.append(r["pair_id"])
        if args.limit:
            qids = qids[: max(10, args.limit // 5)]
            mrecs = [r for r in mrecs if r["pair_id"] in set(qids)]
        mlabels = np.array([bool(r["label"]) for r in mrecs])
        roles = np.array([r["meta"]["role"] for r in mrecs])
        qarr = np.array([r["pair_id"] for r in mrecs])
        groups = [np.where(qarr == q)[0] for q in qids]
        mtexts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in mrecs]
        menc = [bundle.encode(mtexts[i:i + bs]) for i in range(0, len(mtexts), bs)]
        results["mc"] = {"n_questions": len(qids), "n_statements": len(mrecs), "conditions": {}}
        base_q = None
        lines += ["", "## %s question level (%d questions%s)" % (args.mc_dataset, len(qids), "; half-A questions excluded" if args.train_on_half_a else ""), "",
                  "| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) |", "|---|---|---|---|---|---|---|---|"]
        for name, a in [("baseline", None)] + [("iti_k%d_a%g" % (args.k, a), a) for a in args.alphas]:
            with iti_steer(bundle, spec if a is not None else None, alpha=a):
                p, pred = run(bundle, menc, lab_map, None)
            pred = np.asarray(pred)
            p = np.nan_to_num(np.asarray(p, dtype=float), nan=0.5)
            mc1, amx, mc2, _ = question_metrics(p, pred, mlabels, roles, groups)
            if base_q is None:
                base_q = (mc1, amx)
            st = state(pred, mlabels)
            m = {"mc1": float(mc1.mean()), "argmax_acc": float(amx.mean()), "mc2": float(np.nanmean(mc2)), "statement_auroc": float(auroc(p, mlabels)),
                 "statement_acc_all": float((st == "known").mean()), "mc1_fixed": int((~base_q[0] & mc1).sum()), "mc1_damaged": int((base_q[0] & ~mc1).sum())}
            results["mc"]["conditions"][name] = m
            lines.append("| %s | %.3f | %.3f | %.3f | %d | %d | %.3f | %.3f |" % (name, m["mc1"], m["argmax_acc"], m["mc2"], m["mc1_fixed"], m["mc1_damaged"], m["statement_auroc"], m["statement_acc_all"]))
            print("  MC %-16s MC1 %.3f argmax %.3f MC2 %.3f | fixed %d damaged %d (%.0fs)" % (name, m["mc1"], m["argmax_acc"], m["mc2"], m["mc1_fixed"], m["mc1_damaged"], time.time() - t0), flush=True)

    results["elapsed_s"] = time.time() - t0
    stem = os.path.join(out_dir, "iti" + tag)
    atomic_json_dump(results, stem + ".json")
    np.savez(stem + ".npz", ids=np.array([r["id"] for r in recs]), labels=labels, pair_ids=np.array(pair_ids), half_a=half_a,
             **{"%s__p" % k: v[0] for k, v in per_item.items()}, **{"%s__pred" % k: v[1] for k, v in per_item.items()})
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()

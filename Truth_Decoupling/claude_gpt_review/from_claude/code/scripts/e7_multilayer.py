#!/usr/bin/env python3
"""E7: stronger evidence writes where a knowledge gap exists -- multi-layer amplify, all-position amplify,
and larger single-layer gains. Same items / metrics as E2b; directions from E1 (mean-diff, centred).

Conditions (all mode="amplify", i.e. h += alpha * w w^T (h - center), the model's own evidence scaled):
  amp_L{l}_a{alpha}             single layer, last position, alphas beyond the E2b grid
  amp_multi_{l1+l2+..}_a{alpha} the same gain applied at several layers at once (LayerNavigator-style K>1)
  amp_all_L{l}_a{alpha}         single layer but at every token position (statement tokens included)
plus the matching C_add alpha for the 2x3 row.

Usage: python3 scripts/e7_multilayer.py --exp E2b_tqa --model llama3.1-8b --layers 12 13 14 --alphas 3 5 8 --multi 12,13,14 13,14
Output: results/E7/<model>/multilayer.json (+ .md)
"""
import argparse
import os
import time

from common import RESULTS_DIR, load_split, tf_prompt
from e2b_h1_rescue import metrics, run

import numpy as np

from core.config import load_experiment, load_model_config
from core.intervene import SteerSpec
from core.io_utils import atomic_json_dump, load_json, set_seed
from core.model_io import load_model
from core.readout import build_label_token_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa")
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=[13])
    ap.add_argument("--alphas", nargs="+", type=float, default=[5.0, 8.0, 12.0])
    ap.add_argument("--multi", nargs="*", default=["12,13,14"], help="comma-joined layer groups for joint amplify")
    ap.add_argument("--multi-alphas", nargs="+", type=float, default=[1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--all-pos-alphas", nargs="*", type=float, default=[0.5, 1.0, 2.0],
                    help="pass the flag with no values to skip the all-position conditions (layer sweeps)")
    ap.add_argument("--c-alphas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--random-seeds", type=int, default=0,
                    help="N Haar-random unit directions amplified with the same layers/alpha as the first --multi group (null control)")
    ap.add_argument("--only-random", action="store_true", help="run baseline + random-direction controls only")
    ap.add_argument("--direction", choices=["meandiff", "das"], default="meandiff",
                    help="which E estimate to amplify: mean-difference (E1 npz) or the rank-1 DAS direction (E1 das_E_L*.pt)")
    ap.add_argument("--no-idk", action="store_true", help="evaluate under the forced True/False prompt (no Unknown option); pair with directions estimated the same way")
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
    n_boot = exp.get("n_boot", 300)
    e1_dir = os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    e1 = load_json(os.path.join(e1_dir, "summary.json"))
    z = np.load(os.path.join(e1_dir, "directions_meandiff.npz"))
    cL = e1["C_write_layer"]

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
    texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=not args.no_idk) for r in recs]
    lab_map = build_label_token_map(bundle.tokenizer, texts[0], prompts["statement"]["label_variants"])
    enc_batches = [bundle.encode(texts[i:i + bs]) for i in range(0, len(texts), bs)]
    print("E7 %s: n=%d layers %s alphas %s multi %s prompt=%s" % (mcfg["name"], len(recs), args.layers, args.alphas, args.multi,
                                                                "forced True/False" if args.no_idk else "with Unknown"), flush=True)

    dirs = {}
    if args.direction == "das":
        import glob
        import torch
        for l in set(args.layers) | set(int(x) for g in (args.multi or []) for x in g.split(",") if x):
            files = glob.glob(os.path.join(e1_dir, "das_E_L%d_r1_s*.pt" % l))
            if not files:
                raise SystemExit("no DAS direction for layer %d in %s (run e1_directions.py --das-layers ...)" % (l, e1_dir))
            res = torch.load(files[0], map_location="cpu")
            dirs[l] = res["W"][:, 0].numpy()
        print("using DAS directions at layers %s" % sorted(dirs), flush=True)

    def amp(l, a, position="last"):
        w = dirs[l] if args.direction == "das" else z["E_L%d" % l]
        return SteerSpec(w, l, mode="amplify", alpha=a, center=z["E_center_L%d" % l], position=position)

    conds = []
    if not args.only_random:
        for l in args.layers:
            for a in args.alphas:
                conds.append(("amp_L%d_a%g" % (l, a), lambda bi, l=l, a=a: [amp(l, a)]))
        for grp in args.multi or []:
            ls = [int(x) for x in grp.split(",") if x]
            for a in args.multi_alphas:
                conds.append(("amp_multi_%s_a%g" % ("+".join(map(str, ls)), a), lambda bi, ls=ls, a=a: [amp(l, a) for l in ls]))
        for l in args.layers:
            for a in args.all_pos_alphas:
                conds.append(("amp_all_L%d_a%g" % (l, a), lambda bi, l=l, a=a: [amp(l, a, position="all")]))
        for a in args.c_alphas:
            conds.append(("C_add_L%d_a%g" % (cL, a), lambda bi, a=a: [SteerSpec(z["C_L%d" % cL], cL, mode="add", alpha=-a, coef=[e1["rows"][str(cL)]["gap_C"]])]))
    if args.random_seeds > 0 and args.multi:
        # null control: same operator, same layers, same alphas, but Haar-random unit directions (E centres kept)
        from core.directions import random_direction
        ls = [int(x) for x in args.multi[0].split(",") if x]
        for s in range(args.random_seeds):
            for a in args.multi_alphas:
                conds.append(("rand_multi_%s_a%g_s%d" % ("+".join(map(str, ls)), a, s),
                              lambda bi, ls=ls, a=a, s=s: [SteerSpec(random_direction(bundle.hidden_size, 1, seed=seed + 100 * s + l).squeeze(1).numpy(), l,
                                                                      mode="amplify", alpha=a, center=z["E_center_L%d" % l]) for l in ls]))

    t0 = time.time()
    base_p, base_pred = run(bundle, enc_batches, lab_map, None)
    results = {"model": mcfg["name"], "n": len(recs), "conditions": {"baseline": metrics(labels, base_p, base_pred, base_p, base_pred, datasets, rng, 1)}}
    per_item = {"baseline": (base_p, base_pred)}       # kept for the post-hoc select-on-A / report-on-B layer choice (e7_select_layer.py)
    b = results["conditions"]["baseline"]
    print("baseline AUROC %.3f J %.3f B %.3f ans %.3f" % (b["auroc"], b["J"], b["B"], b["answer_rate"]), flush=True)
    for i, (name, mk) in enumerate(conds):
        p, pred = run(bundle, enc_batches, lab_map, mk)
        m = metrics(labels, p, pred, base_p, base_pred, datasets, rng, n_boot)
        results["conditions"][name] = m
        per_item[name] = (p, pred)
        print("  [%2d/%d] %-26s AUROC %.3f (d %+.3f [%+.3f,%+.3f])  J %.3f  Jp %.3f/%.3f  B %.3f  Bp %.3f/%.3f  ans %.3f  (%.0fs)" % (
            i + 1, len(conds), name, m["auroc"], m["auroc"] - b["auroc"], m["d_auroc_ci95"][0], m["d_auroc_ci95"][1], m["J"],
            m["J_paired"], m["J_paired_base"], m["B"], m["B_paired"], m["B_paired_base"], m["answer_rate"], time.time() - t0), flush=True)

    out_dir = os.path.join(RESULTS_DIR, "E7", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    stem = "multilayer" + (("_" + args.tag) if args.tag else "")
    results["elapsed_s"] = time.time() - t0
    atomic_json_dump(results, os.path.join(out_dir, stem + ".json"))
    # per-item outputs (same pair-hash halves as E11) so the best layer can be chosen on half A and reported on half B
    import hashlib
    pair_ids = [r.get("pair_id") or r["id"] for r in recs]
    half_a = np.array([int(hashlib.sha1(str(pid).encode()).hexdigest(), 16) % 2 == 0 for pid in pair_ids])
    names = list(per_item.keys())
    np.savez(os.path.join(out_dir, stem + ".npz"), ids=np.array([r["id"] for r in recs]), labels=np.array(labels, dtype=bool),
             pair_ids=np.array(pair_ids), datasets=np.array(datasets), half_a=half_a, cond_names=np.array(names),
             **{"p__" + k: np.asarray(per_item[k][0], dtype=float) for k in names},
             **{"pred__" + k: np.array(per_item[k][1]) for k in names})
    lines = ["# E7 %s (n=%d)" % (mcfg["name"], len(recs)), "", "| condition | AUROC (d [95% CI]) | J | J_paired cond/base | B | B_paired cond/base | answer rate |", "|---|---|---|---|---|---|---|"]
    for name, m in results["conditions"].items():
        lines.append("| %s | %.3f (%+.3f [%+.3f,%+.3f]) | %.3f | %.3f / %.3f | %.3f | %.3f / %.3f | %.3f |" % (
            name, m["auroc"], m["auroc"] - b["auroc"], m["d_auroc_ci95"][0], m["d_auroc_ci95"][1], m["J"], m["J_paired"], m["J_paired_base"],
            m["B"], m["B_paired"], m["B_paired_base"], m["answer_rate"]))
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(out_dir, stem + ".md"))


if __name__ == "__main__":
    main()

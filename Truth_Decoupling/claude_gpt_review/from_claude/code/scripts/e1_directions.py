#!/usr/bin/env python3
"""E1: do the two directions exist? Layer profiles, LA-Score, cosines.

For one model:
  * E contrast  : true vs false statements under the neutral fact-check prompt
  * C contrast  : confident vs cautious system prompt, identical user content
  * capture last-token resid (+ attn) at every layer in one pass per prompt set
  * per layer: mean-diff directions wE, wC; probe CV-AUC; cos(wE, wC); LA-Score rows
  * optional: rank-1 DAS at the LA-Score top-k layers (--das-topk K)

Outputs results/E1/<model>/: directions_meandiff.npz, layer_scores.json, summary.json

Usage:
  python3 scripts/e1_directions.py --exp E1_directions --model qwen3-8b [--das-topk 3]
"""
import argparse
import os
import time

from common import RESULTS_DIR, commitment_prompts, load_split, tf_prompt, to_numpy

import numpy as np
import torch

from core.config import load_experiment
from core.directions import balanced_pair_weights, cosine, cv_auc, das_file_name, length_bins, mean_diff_direction, train_das
from core.io_utils import atomic_json_dump, exp_name, now_iso, set_seed
from core.layer_score import layer_scores, top_k_layers
from core.model_io import capture_last_token, load_model


def collect(bundle, texts, layers, batch_size):
    acts, _ = capture_last_token(bundle, texts, layers, components=("resid", "attn"), batch_size=batch_size)
    resid = {l: to_numpy(acts[("resid", l)]) for l in layers}
    attn = {l: to_numpy(acts[("attn", l)]) for l in layers}
    return resid, attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E1_directions")
    ap.add_argument("--model", default=None, help="override experiment model")
    layer_choice = ap.add_mutually_exclusive_group()
    layer_choice.add_argument("--das-topk", type=int, default=0)
    ap.add_argument("--das-tags", default="C,E", help="which directions to fit with DAS (comma list of C,E); E alone halves the cost")
    ap.add_argument("--das-pairs", type=int, default=None, help="number of estimation pairs for DAS (default: experiment das.n_train_pairs); "
                    "set to the full calib+dev pair count so DAS and mean-diff see the same data")
    # Estimation-set size experiment (赵磊 2026-11-02: is DAS data-starved under the 1:2:7 split?). The extra pairs join the
    # pool for BOTH estimators (mean-diff directions and the DAS pairs); e23_knob_sweep.py must get the same --est-extra-pairs.
    ap.add_argument("--est-extra-pairs", type=int, default=0,
                    help="add this many pairs, drawn from --est-extra-split with the SAME seeded draw as e23_knob_sweep.py --tune-pairs "
                         "(sorted unique pair ids, default_rng(seed).choice), to the estimation pool; 192 = exactly the old tuning pairs")
    ap.add_argument("--est-extra-split", default="test", help="split the extra estimation pairs come from (the evaluation split)")
    ap.add_argument("--est-extra-seed", type=int, default=None,
                    help="seed of the extra-pair draw (default: the experiment seed); must equal the evaluation experiment's seed")
    layer_choice.add_argument("--das-layers", nargs="+", type=int, default=None,
                    help="explicit layers for DAS (overrides the LA-Score top-k choice)")
    layer_choice.add_argument("--das-all-layers", action="store_true",
                    help="fit a separate DAS direction at EVERY loaded decoder layer, including first and last; do not preselect by readout score")
    ap.add_argument("--limit", type=int, default=None, help="limit statements per contrast (debug)")
    # DAS variants (2026-10-31). Defaults reproduce the previous behaviour and file names exactly.
    ap.add_argument("--das-seeds", nargs="*", type=int, default=None, help="DAS seeds (default: the experiment seed only); the plan's setting is 42 43 44")
    ap.add_argument("--das-loss", choices=["kl", "jsd"], default="kl", help="divergence: forward KL (v12 default) or Jensen-Shannon over the vocabulary (CDAS)")
    ap.add_argument("--das-bidir", action="store_true", help="bidirectional objective: also clamp the source with the base's projection and match the base's distribution (CDAS D+ and D-)")
    ap.add_argument("--das-positions", choices=["last", "all"], default="last", help="clamp the last position only (one-to-one) or every base position with the source's last-token projection (CDAS one-to-many)")
    ap.add_argument("--das-balance", action="store_true", help="inverse-frequency pair weights so every (category, length tertile) cell carries equal weight")
    ap.add_argument("--das-trim", type=float, default=0.0, help="trimmed-mean fraction for the DAS mean projection gap (0.1 = drop 10%% of each tail)")
    ap.add_argument("--das-anchor", choices=["prompt", "current"], default="prompt",
                    help="where the interchange reads the source and writes the base at every teacher-forced step: 'prompt' = the original "
                         "prompts' last token, fixed while the answer prefix grows (= the evaluation's intervention position; GPT review "
                         "2026-11-01 round 2); 'current' = the growing sequence's last token (v12 behaviour, swaps answer tokens from step 2 on)")
    ap.add_argument("--das-variant", default=None, help="file-name suffix for non-default DAS variants (default: derived from the flags, "
                    "e.g. tf-jsd-bidir-allpos; 'tf' marks the forced True/False estimation prompt, 'cur' the v12 anchor)")
    ap.add_argument("--das-rank", type=int, default=None, help="DAS subspace rank (default: the experiment's das.rank, 1). The rank is part of the "
                    "file name (das_E_L<l>_r<rank>_s<seed>[_<variant>].pt); rank > 1 is a pilot of whether a low-rank subspace swaps where no single direction does")
    ap.add_argument("--trim", type=float, default=0.0, help="trimmed-mean fraction for the mean-difference directions and centres (default 0 = plain mean)")
    ap.add_argument("--direction-prompt", choices=["config", "tf", "idk"], default="config",
                    help="statement prompt under which E (mean-diff AND the DAS pairs) is estimated: 'tf' = forced True/False, i.e. the final "
                         "protocol's pass-B prompt; 'idk' = True/False/Unknown; 'config' = the experiment's direction_prompt_allow_idk. "
                         "Estimation prompt must equal the evaluation prompt (GPT review 2026-11-01)")
    args = ap.parse_args()
    if args.est_extra_pairs < 0:
        raise SystemExit("--est-extra-pairs must be >= 0")

    exp = load_experiment(args.exp)
    if args.model:
        from core.config import load_model_config
        exp["model"] = load_model_config(args.model)
    mcfg, prompts = exp["model"], exp["prompts"]
    seed = exp.get("seed", 42)
    set_seed(seed)
    out_dir = os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    bundle = load_model(mcfg)
    layers = list(range(bundle.num_layers))
    if args.das_layers is not None and (len(set(args.das_layers)) != len(args.das_layers)
                                       or not set(args.das_layers).issubset(layers)):
        raise ValueError("--das-layers must contain unique indices in [0, %d)" % bundle.num_layers)
    bs = exp.get("batch_size", 8)

    # ---- E contrast ------------------------------------------------------- #
    e_recs = []
    for ds in exp["e_datasets"]:
        for split in exp.get("direction_splits", ["calib", "dev"]):
            e_recs += load_split(ds, split)
    est_extra = None
    if args.est_extra_pairs > 0:
        # Same draw as e23_knob_sweep.py's tuning set (its FIRST draw from default_rng(seed)): with N = the old --tune-pairs the
        # added pairs are exactly the old tuning pairs, so the old test pairs are untouched. e23 --est-extra-pairs N removes the
        # same pairs from its tuning/test pool and cross-checks the ids recorded in summary.json["est_extra"].
        ex_seed = args.est_extra_seed if args.est_extra_seed is not None else seed
        extra_recs = []
        for ds in exp["e_datasets"]:
            extra_recs += load_split(ds, args.est_extra_split)
        pool_ids = {str(r.get("pair_id") or r["id"]) for r in e_recs}
        ex_pids = [str(r.get("pair_id") or r["id"]) for r in extra_recs]
        uniq = sorted(set(ex_pids))
        if args.est_extra_pairs > len(uniq):
            raise ValueError("--est-extra-pairs %d exceeds the %d pairs of split %s" % (args.est_extra_pairs, len(uniq), args.est_extra_split))
        chosen = set(np.random.default_rng(ex_seed).choice(uniq, args.est_extra_pairs, replace=False).tolist())
        if chosen & pool_ids:
            raise ValueError("extra pairs overlap the calib/dev estimation pool, e.g. %s" % sorted(chosen & pool_ids)[:5])
        added = [r for r, p in zip(extra_recs, ex_pids) if p in chosen]
        n_t = sum(1 for r in added if r["label"] is True)
        n_f = sum(1 for r in added if r["label"] is False)
        if not all(r.get("pair_id") for r in added) or n_t != args.est_extra_pairs or n_f != args.est_extra_pairs:
            raise ValueError("extra pairs must be complete true/false pairs with a pair_id: %d true / %d false for %d pairs" % (n_t, n_f, args.est_extra_pairs))
        e_recs += added
        est_extra = {"n_pairs": args.est_extra_pairs, "split": args.est_extra_split, "seed": ex_seed, "pair_ids": sorted(chosen)}
        print("[E] estimation pool extended by %d pairs from split %s (seed %d): %d statements" % (
            args.est_extra_pairs, args.est_extra_split, ex_seed, len(e_recs)), flush=True)
    if args.limit:
        e_recs = e_recs[: args.limit]
    if args.das_pairs is not None:                              # fail before any GPU work, not after the contrasts are captured
        bp = {}
        for r in e_recs:
            if r.get("pair_id"):
                bp.setdefault(r["pair_id"], set()).add(r["label"])
        n_complete = sum(1 for s in bp.values() if {True, False} <= s)
        if n_complete < args.das_pairs:
            raise ValueError("--das-pairs %d requested but the estimation pool holds only %d complete pairs" % (args.das_pairs, n_complete))
    e_pos = [r for r in e_recs if r["label"] is True]
    e_neg = [r for r in e_recs if r["label"] is False]
    # Estimate E under the SAME statement prompt the evaluations use (with the Unknown option) unless the
    # experiment config says otherwise; a prompt mismatch shifts the last-token mean and turns the
    # amplify operator's (h - center) into a constant offset along w, i.e. a hidden additive bias.
    e_idk = bool(exp.get("direction_prompt_allow_idk", True)) if args.direction_prompt == "config" else (args.direction_prompt == "idk")
    print("[E] %d true / %d false statements (prompt allow_idk=%s, source=%s)" % (len(e_pos), len(e_neg), e_idk, args.direction_prompt), flush=True)
    E_pos_r, E_pos_a = collect(bundle, [tf_prompt(bundle, prompts, r["text"], allow_idk=e_idk) for r in e_pos], layers, bs)
    E_neg_r, E_neg_a = collect(bundle, [tf_prompt(bundle, prompts, r["text"], allow_idk=e_idk) for r in e_neg], layers, bs)

    # ---- C contrast (same statements, confident vs cautious system) -------- #
    c_recs = e_recs if not exp.get("c_datasets") else sum((load_split(d, s) for d in exp["c_datasets"] for s in exp.get("direction_splits", ["calib", "dev"])), [])
    if args.limit:
        c_recs = c_recs[: args.limit]
    conf_texts, caut_texts = [], []
    for r in c_recs:
        a, b = commitment_prompts(bundle, prompts, r["text"], is_statement=(r["task"] == "tf"))
        conf_texts.append(a)
        caut_texts.append(b)
    print("[C] %d confident/cautious prompt pairs" % len(conf_texts), flush=True)
    C_pos_r, C_pos_a = collect(bundle, conf_texts, layers, bs)
    C_neg_r, C_neg_a = collect(bundle, caut_texts, layers, bs)

    # ---- per-layer analysis ------------------------------------------------ #
    ls_cfg = exp.get("layer_score", {})
    scores_E = layer_scores(E_pos_r, E_neg_r, E_pos_a, E_neg_a, eps_sigma=ls_cfg.get("eps_sigma", 0.5), lam=ls_cfg.get("lam", 1.0),
                            k_folds=ls_cfg.get("k_folds", 5), l2=ls_cfg.get("l2", 1.0), seed=seed)
    scores_C = layer_scores(C_pos_r, C_neg_r, C_pos_a, C_neg_a, eps_sigma=ls_cfg.get("eps_sigma", 0.5), lam=ls_cfg.get("lam", 1.0),
                            k_folds=ls_cfg.get("k_folds", 5), l2=ls_cfg.get("l2", 1.0), seed=seed)
    # token-unembedding reference directions (is a direction just a shallow "say Unknown" / "say True" lever?)
    from core.readout import build_label_token_map
    lab_map = build_label_token_map(bundle.tokenizer, tf_prompt(bundle, prompts, e_recs[0]["text"], allow_idk=True),
                                    prompts["statement"]["label_variants"])
    U = bundle.model.get_output_embeddings().weight.detach().float().cpu().numpy()
    u_abst = U[lab_map["Abstain"]].mean(0) if lab_map.get("Abstain") else np.zeros(U.shape[1])
    u_tf = U[lab_map["True"]].mean(0) - U[lab_map["False"]].mean(0)
    # random-direction null for cosines in this dimension (|cos| of two random unit vectors)
    rng_null = np.random.default_rng(seed)
    null = np.abs(np.sum(rng_null.normal(size=(2000, bundle.hidden_size)) * rng_null.normal(size=(2000, bundle.hidden_size)), axis=1)
                  / bundle.hidden_size) * np.sqrt(bundle.hidden_size) / np.sqrt(bundle.hidden_size)
    null_cos_p99 = float(np.quantile(np.abs([cosine(a, b) for a, b in zip(rng_null.normal(size=(500, bundle.hidden_size)),
                                                                           rng_null.normal(size=(500, bundle.hidden_size)))]), 0.99))
    del null

    wE, wC, rows = {}, {}, {}
    cE, cC = {}, {}
    for l in layers:
        wE[l], mE = mean_diff_direction(E_pos_r[l], E_neg_r[l], trim=args.trim)
        wC[l], mC = mean_diff_direction(C_pos_r[l], C_neg_r[l], trim=args.trim)
        cE[l], cC[l] = mE["center"], mC["center"]          # (mu_pos + mu_neg) / 2, trimmed if --trim > 0
        # Mahalanobis-style cosine: whiten by the pooled within-class std per coordinate (divide, i.e. Sigma^-1/2)
        pooled = np.vstack([E_pos_r[l] - E_pos_r[l].mean(0), E_neg_r[l] - E_neg_r[l].mean(0),
                            C_pos_r[l] - C_pos_r[l].mean(0), C_neg_r[l] - C_neg_r[l].mean(0)])
        sd = pooled.std(0) + 1e-6
        rows[l] = {
            "cos_EC": cosine(wE[l], wC[l]),
            "cos_EC_whitened": cosine(wE[l] / sd, wC[l] / sd),
            "cos_C_unembed_abstain": cosine(wC[l], u_abst) if np.any(u_abst) else float("nan"),
            "cos_E_unembed_true_minus_false": cosine(wE[l], u_tf),
            "cos_C_unembed_true_minus_false": cosine(wC[l], u_tf),
            "gap_E": mE["gap"], "gap_C": mC["gap"],
            "sep_E": scores_E["rows"][l]["sep"], "sep_C": scores_C["rows"][l]["sep"],
            "S_E": scores_E["rows"][l]["S"], "S_C": scores_C["rows"][l]["S"],
            "mass_E": scores_E["rows"][l]["mass"], "mass_C": scores_C["rows"][l]["mass"],
        }
        print("  L%02d sepE=%.3f sepC=%.3f S_E=%+.2f S_C=%+.2f cos=%.3f cosW=%.3f cos(C,unk)=%.3f cos(E,T-F)=%.3f" % (
            l, rows[l]["sep_E"], rows[l]["sep_C"], rows[l]["S_E"], rows[l]["S_C"], rows[l]["cos_EC"], rows[l]["cos_EC_whitened"],
            rows[l]["cos_C_unembed_abstain"], rows[l]["cos_E_unembed_true_minus_false"]), flush=True)

    np.savez(os.path.join(out_dir, "directions_meandiff.npz"),
             **{"E_L%d" % l: wE[l] for l in layers}, **{"C_L%d" % l: wC[l] for l in layers},
             **{"E_center_L%d" % l: cE[l] for l in layers},
             **{"C_center_L%d" % l: cC[l] for l in layers})
    atomic_json_dump({"E": scores_E, "C": scores_C}, os.path.join(out_dir, "layer_scores.json"))

    summary = {
        "experiment": exp_name("E1", model=mcfg["name"], s=seed), "model": mcfg["name"], "started": now_iso(),
        "num_layers": bundle.num_layers, "direction_prompt_allow_idk": e_idk,
        "layer_roles": {"read": "probe separability on estimation data",
                        "score_proposed_write": "LA-Score proposal only; not measured intervention benefit",
                        "measured_write": "selected later by E23 tuning-set intervention results"},
        "n_E": [len(e_pos), len(e_neg)], "n_C": len(conf_texts), "est_extra": est_extra, "rows": rows, "null_cos_p99": null_cos_p99,
        "label_token_map": lab_map,
        "E_read_layer": scores_E["read_layer"], "E_write_layer": scores_E["write_layer"], "E_delta": scores_E["delta_read_write"],
        "C_read_layer": scores_C["read_layer"], "C_write_layer": scores_C["write_layer"], "C_delta": scores_C["delta_read_write"],
        "E_topk": top_k_layers(scores_E, 3), "C_topk": top_k_layers(scores_C, 3),
    }

    # ---- optional DAS at top-k layers ------------------------------------ #
    if args.das_all_layers or args.das_topk > 0 or args.das_layers:
        das_cfg = exp.get("das", {})
        n_pairs = args.das_pairs or das_cfg.get("n_train_pairs", 64)
        def n_tok(text):
            return len(bundle.tokenizer(text, add_special_tokens=False)["input_ids"])

        def category(r):
            m = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            return str(m.get("category") or r.get("dataset") or "?")

        pairs_C = [{"base": bundle.encode(a), "source": bundle.encode(b)} for a, b in zip(conf_texts[:n_pairs], caut_texts[:n_pairs])]
        keys_C = [(category(r), n_tok(r["text"])) for r in c_recs[:n_pairs]]
        # E pairs: base = false statement, source = true statement of the same pair (needs pair_id)
        by_pair = {}
        for r in e_recs:
            if r.get("pair_id"):
                by_pair.setdefault(r["pair_id"], {})[r["label"]] = r
        pair_recs_E = [d for d in by_pair.values() if True in d and False in d][:n_pairs]
        pairs_E = [{"base": bundle.encode(tf_prompt(bundle, prompts, d[False]["text"], allow_idk=e_idk)),
                    "source": bundle.encode(tf_prompt(bundle, prompts, d[True]["text"], allow_idk=e_idk))}
                   for d in pair_recs_E]
        keys_E = [(category(d[True]), n_tok(d[True]["text"]) + n_tok(d[False]["text"])) for d in pair_recs_E]

        def cell_weights(keys):
            # balanced weighting by (category, length tertile): every cell carries equal total loss weight
            bins = length_bins([k[1] for k in keys], 3)
            return balanced_pair_weights([(k[0], b) for k, b in zip(keys, bins)])

        seeds = args.das_seeds or [seed]
        variant = args.das_variant if args.das_variant is not None else "-".join(
            (["tf"] if not e_idk else []) + (["cur"] if args.das_anchor == "current" else [])
            + ([args.das_loss] if args.das_loss != "kl" else []) + (["bidir"] if args.das_bidir else [])
            + (["allpos"] if args.das_positions == "all" else []) + (["bal"] if args.das_balance else [])
            + (["trim%g" % args.das_trim] if args.das_trim > 0 else []))
        summary["das"] = {"variant": variant, "seeds": seeds, "loss": args.das_loss, "bidirectional": bool(args.das_bidir),
                          "positions": args.das_positions, "anchor": args.das_anchor, "prompt_allow_idk": e_idk,
                          "balanced": bool(args.das_balance), "trim": args.das_trim, "n_pairs": {}}
        das_rank = args.das_rank if args.das_rank is not None else das_cfg.get("rank", 1)
        if das_rank < 1 or das_rank > bundle.hidden_size:
            raise ValueError("--das-rank must be in [1, %d]" % bundle.hidden_size)
        summary["das"]["rank"] = das_rank
        print("[DAS] variant=%r rank=%d seeds=%s loss=%s bidir=%s positions=%s anchor=%s prompt_allow_idk=%s balanced=%s trim=%g" % (
            variant, das_rank, seeds, args.das_loss, args.das_bidir, args.das_positions, args.das_anchor, e_idk, args.das_balance, args.das_trim), flush=True)
        das_tags = [t.strip() for t in args.das_tags.split(",") if t.strip()]
        for tag, pairs, keys, sc in (("C", pairs_C, keys_C, scores_C), ("E", pairs_E, keys_E, scores_E)):
            if tag not in das_tags:
                continue
            if not pairs:
                print("  [DAS %s] no pairs available, skipped" % tag, flush=True)
                continue
            weights = cell_weights(keys) if args.das_balance else None
            summary["das"]["n_pairs"][tag] = len(pairs)
            if weights is not None:
                print("  [DAS %s] balanced weights over %d cells: min %.3f max %.3f" % (
                    tag, len(set((k[0], b) for k, b in zip(keys, length_bins([k[1] for k in keys], 3)))), min(weights), max(weights)), flush=True)
            train_layers = layers if args.das_all_layers else (args.das_layers if args.das_layers else top_k_layers(sc, args.das_topk))
            summary["das"]["%s_trained_layers" % tag] = list(train_layers)
            for l in train_layers:
                ws = []
                for s in seeds:
                    res = train_das(bundle, pairs, l, rank=das_rank, T=das_cfg.get("T", 8),
                                    epochs=das_cfg.get("epochs", 10), lr=das_cfg.get("lr", 1e-3), seed=s,
                                    loss=args.das_loss, bidirectional=args.das_bidir, positions=args.das_positions,
                                    pair_weights=weights, trim=args.das_trim, anchor=args.das_anchor)
                    w = res["W"][:, 0].numpy()
                    ws.append(w)
                    torch.save(res, os.path.join(out_dir, das_file_name(tag, l, res["rank"], s, variant)))
                    key = "%s_L%d_s%d" % (tag, l, s)
                    own, other = (wC if tag == "C" else wE)[l], (wE if tag == "C" else wC)[l]
                    if das_rank == 1:
                        cos_own, cos_other = cosine(w, own), cosine(w, other)
                    else:   # rank > 1: how much of the mean-diff direction lies in the subspace (largest principal cosine)
                        from core.directions import principal_cosines
                        cos_own = float(max(principal_cosines(res["W"].numpy(), own.reshape(-1, 1))))
                        cos_other = float(max(principal_cosines(res["W"].numpy(), other.reshape(-1, 1))))
                    summary["das"][key] = {"final_loss": res["losses"][-1], "mean_gap": res["mean_gap"], "rank": das_rank,
                                           "cos_with_meandiff": cos_own, "cos_with_other_meandiff": cos_other}
                    print("  [DAS %s L%d s%d r%d] loss=%.5f cos(meandiff)=%.3f cos(other)=%.3f" % (
                        tag, l, s, das_rank, res["losses"][-1], summary["das"][key]["cos_with_meandiff"],
                        summary["das"][key]["cos_with_other_meandiff"]), flush=True)
                if len(ws) > 1:
                    pc = [abs(cosine(ws[i], ws[j])) for i in range(len(ws)) for j in range(i + 1, len(ws))]
                    summary["das"]["%s_L%d_seed_abscos" % (tag, l)] = {"min": float(min(pc)), "mean": float(np.mean(pc))}
                    print("  [DAS %s L%d] |cos| between seeds: min %.3f mean %.3f" % (tag, l, min(pc), float(np.mean(pc))), flush=True)

    summary["elapsed_s"] = time.time() - t0
    atomic_json_dump(summary, os.path.join(out_dir, "summary.json"))
    print("\nE1 done -> %s (%.0fs)" % (out_dir, summary["elapsed_s"]), flush=True)


if __name__ == "__main__":
    main()

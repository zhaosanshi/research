#!/usr/bin/env python3
"""E23: choose the operating point of BOTH knobs for the two-context protocol by a pre-declared rule, per model.
(赵磊: Qwen's 2x2 was measured at layers chosen for the old protocol; apply the DAS rule -- try every layer, pick on half A,
report on half B.)

Protocol (same as e22 split/pik): pass A = question-level "Do you know the answer to this question? Yes/No" (answer rate),
pass B = forced True/False on all items (J / AUROC / B).  For every layer in --layers:
  E_l   : mean-diff evidence direction (estimated under the pass-B prompt), amplified by alpha in --e-alphas
  Cp_l  : commitment direction from the confident/cautious system-prompt contrast at the pass-A position,
          orthogonalised against E_l; added with +/- alpha in --c-alphas (negative = toward abstention)
  Cs_l  : self-report direction = mean(pass-A activation | model says Yes) - mean(... | says No) on the estimation set
          (label-free, Ferrando-style "known/unknown" direction), orthogonalised against E_l; same alphas
Selection rule (declared here, applied on the pair-hash half A, reported on half B):
  E : maximise dJ_all subject to |d answer_rate| <= 0.03
  C : maximise |d answer_rate| subject to |dJ_all| <= 0.02 and |dB_all| <= 0.05, separately for each direction
usage: python3 scripts/e23_knob_sweep.py --exp E2b_tqa --model qwen3-8b --layers 14 16 18 20 22 24 26 28
output: results/E23/<model>/knob_sweep.{json,md,npz}
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e22_two_stage as e22                                      # noqa: E402
from common import RESULTS_DIR, load_split, tf_prompt            # noqa: E402
from core.config import load_experiment, load_model_config       # noqa: E402
from core.directions import load_das_direction, mean_diff_direction  # noqa: E402
from core.intervene import SteerSpec, next_token_logits          # noqa: E402
from core.io_utils import atomic_json_dump, load_json, set_seed  # noqa: E402
from core.metrics import auroc, sel_key                          # noqa: E402
from core.model_io import capture_last_token, load_model         # noqa: E402
from core.readout import build_label_token_map, label_logprobs_from_logits  # noqa: E402


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa")
    ap.add_argument("--model", required=True)
    layer_choice = ap.add_mutually_exclusive_group(required=True)
    layer_choice.add_argument("--layers", nargs="+", type=int)
    layer_choice.add_argument("--all-layers", action="store_true",
                             help="evaluate a separate intervention at every loaded decoder layer, including first and last")
    ap.add_argument("--e-alphas", nargs="+", type=float, default=[2.0, 3.0])
    ap.add_argument("--c-alphas", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--wording", default="pik")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plan-split", dest="teacher_split", action="store_true",
                    help="the original plan's split: directions from calib (32 pairs) only; selection on dev (64 pairs); report on test")
    ap.add_argument("--tune-pairs", type=int, default=0, help="1:2:7 split: draw this many test pairs (seeded) as the tuning set; select there, report on the remaining test pairs")
    ap.add_argument("--est-extra-pairs", type=int, default=0,
                    help="estimation-set size experiment: move this many pairs (the FIRST seeded draw, i.e. the old --tune-pairs set when equal) "
                         "from the evaluation pool into the estimation set (mean-diff directions, centres, C directions); the tuning set is "
                         "then drawn from the remaining pairs. e1_directions.py --est-extra-pairs N trains DAS on the same pairs (ids cross-checked)")
    ap.add_argument("--e-spans", nargs="*", default=[], help="multi-layer E candidates, e.g. 12,13,14 13,14,15 (each layer uses its own direction/centre)")
    ap.add_argument("--no-cs", action="store_true", help="skip the self-report C estimator (ablation only)")
    ap.add_argument("--skip-e", action="store_true", help="no E conditions (C-only sweep)")
    ap.add_argument("--e-estimator", choices=["meandiff", "probe", "das"], default="meandiff",
                    help="E direction: class-mean difference (default), the logistic-probe weight vector fitted on the same statements, "
                         "or the rank-1 DAS direction trained at the layer (all seeds in --das-dir averaged after sign alignment)")
    ap.add_argument("--das-dir", default=None, help="directory with das_E_L<l>_r1_s<seed>[_<variant>].pt files (e1_directions.py --das-layers ...); required for --e-estimator das")
    ap.add_argument("--das-variant", default="", help="DAS file-name suffix to use (default: the plain v12 objective, files without suffix)")
    ap.add_argument("--skip-cp", action="store_true", help="no prompt-contrast C conditions (self-report C only)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.e_estimator == "das" and not args.das_dir:
        raise SystemExit("--e-estimator das needs --das-dir")
    if args.est_extra_pairs < 0:
        raise SystemExit("--est-extra-pairs must be >= 0")
    e22.SPLIT_A = dict(e22.SPLIT_WORDINGS[args.wording])

    exp = load_experiment(args.exp)
    mcfg = load_model_config(args.model)
    prompts = exp["prompts"]; st = prompts["statement"]; cc = prompts["commitment_contrast"]
    set_seed(exp.get("seed", 42)); bs = exp.get("batch_size", 8)
    t0 = time.time()

    recs = []
    for ds in exp["eval_datasets"]:
        recs += load_split(ds, exp.get("eval_split", "test"))
    if args.limit:
        recs = recs[: args.limit]
    rng = np.random.default_rng(exp.get("seed", 42))          # one stream: the extra-estimation draw first (if any), then the tuning draw
    extra_recs, extra_ids = [], set()
    if args.est_extra_pairs > 0:
        # Estimation-set size experiment (赵磊 2026-11-02). The first draw of the stream is identical to the old --tune-pairs draw
        # (fresh default_rng(seed).choice on the sorted unique pair ids), so --est-extra-pairs 192 moves exactly the old tuning
        # pairs into the estimation set; the tuning set is then redrawn from the remaining pairs and the test set shrinks.
        if args.teacher_split:
            raise SystemExit("--est-extra-pairs cannot be combined with --plan-split")
        uniq0 = sorted(set(str(r.get("pair_id") or r["id"]) for r in recs))
        if args.est_extra_pairs + max(args.tune_pairs, 0) >= len(uniq0):
            raise SystemExit("--est-extra-pairs %d + --tune-pairs %d leave no test pairs out of %d" % (args.est_extra_pairs, args.tune_pairs, len(uniq0)))
        extra_ids = set(rng.choice(uniq0, args.est_extra_pairs, replace=False).tolist())
        extra_recs = [r for r in recs if str(r.get("pair_id") or r["id"]) in extra_ids]
        recs = [r for r in recs if str(r.get("pair_id") or r["id"]) not in extra_ids]
        if len(extra_recs) != 2 * args.est_extra_pairs or sum(bool(r["label"]) for r in extra_recs) != args.est_extra_pairs:
            raise SystemExit("the moved pairs are not complete true/false pairs (%d statements for %d pairs)" % (len(extra_recs), args.est_extra_pairs))
        print("estimation pool: +%d pairs (%d statements) moved out of the evaluation pool; %d statements remain for tuning/test" % (
            args.est_extra_pairs, len(extra_recs), len(recs)), flush=True)
    labels = np.array([bool(r["label"]) for r in recs])
    pair_ids = [r.get("pair_id") or r["id"] for r in recs]
    if args.tune_pairs > 0:                                   # 赵磊's 1:2:7: tune set drawn once from the test pool, fixed seed
        uniq = sorted(set(map(str, pair_ids)))
        tune = set(rng.choice(uniq, args.tune_pairs, replace=False).tolist())
        half_a = np.array([str(p) in tune for p in pair_ids])
        print("tune set: %d pairs (%d statements); remaining test: %d statements" % (args.tune_pairs, int(half_a.sum()), int((~half_a).sum())), flush=True)
    else:
        half_a = np.array([int(hashlib.sha1(str(p).encode()).hexdigest(), 16) % 2 == 0 for p in pair_ids])
    spans = [[int(x) for x in sp.split(",")] for sp in args.e_spans]
    est, dev = [], []
    for ds in exp["eval_datasets"]:
        est += load_split(ds, "calib") + ([] if args.teacher_split else load_split(ds, "dev"))
        dev += load_split(ds, "dev")
    est += extra_recs                                         # order: calib, dev, extra -- the same order e1_directions.py builds its pool
    if args.e_estimator == "das":
        # The DAS directions must not have been trained on pairs that are still in this sweep's tuning/test pool. The pair ids
        # come from e1's summary.json (its own record of the draw); the per-layer pair COUNT is checked again from the .pt files below.
        sm_path = os.path.join(args.das_dir, "summary.json")
        if not os.path.exists(sm_path):
            raise SystemExit("no summary.json in %s: cannot verify which pairs the DAS directions were trained on" % args.das_dir)
        das_ids = set((load_json(sm_path).get("est_extra") or {}).get("pair_ids") or [])
        if not das_ids <= extra_ids:
            raise SystemExit("DAS directions in %s were trained on %d extra pairs that this sweep keeps in its tuning/test pool "
                             "(run both with the same --est-extra-pairs and seed)" % (args.das_dir, len(das_ids - extra_ids)))
        if das_ids != extra_ids:
            print("note: DAS trained with %d extra estimation pairs, this sweep moves %d; no leak, but the estimators do not see the same data" % (
                len(das_ids), len(extra_ids)), flush=True)
    y_est = np.array([bool(r["label"]) for r in est])
    y_dev = np.array([bool(r["label"]) for r in dev])
    print("directions from %d statements (%s); selection set for --plan-split = dev %d statements" % (len(est), "calib only" if args.teacher_split else "calib+dev", len(dev)), flush=True)
    bundle = load_model(mcfg)                                 # after every data/leak check, so a mismatch fails before the model loads

    # ---- texts and label maps -------------------------------------------- #
    tA = [bundle.build_chat(st["system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in recs]
    tB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in recs]
    mA = build_label_token_map(bundle.tokenizer, tA[0], e22.SPLIT_A["variants"])
    mB = build_label_token_map(bundle.tokenizer, tB[0], {"True": st["label_variants"]["True"], "False": st["label_variants"]["False"]})
    eA = [bundle.build_chat(st["system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    eB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
    dA = [bundle.build_chat(st["system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in dev]
    dB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in dev]
    conf = [bundle.build_chat(cc["confident_system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    caut = [bundle.build_chat(cc["cautious_system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    L = list(range(bundle.num_layers)) if args.all_layers else args.layers
    Lall = sorted(set(L) | {l for sp in spans for l in sp})      # directions needed at every layer any candidate touches
    if len(set(L)) != len(L) or any(l < 0 or l >= bundle.num_layers for l in Lall):
        raise ValueError("requested layers must be unique valid decoder indices")

    # ---- activations on the estimation set ----------------------------- #
    aB, _ = capture_last_token(bundle, eB, Lall, components=("resid",), batch_size=bs)
    aA, lgA = capture_last_token(bundle, eA, Lall, components=("resid",), batch_size=bs, return_logits=True)
    aConf, _ = capture_last_token(bundle, conf, Lall, components=("resid",), batch_size=bs)
    aCaut, _ = capture_last_token(bundle, caut, Lall, components=("resid",), batch_size=bs)
    self_yes = np.array([label_logprobs_from_logits(lg.numpy(), mA)["Answer"] > label_logprobs_from_logits(lg.numpy(), mA)["Abstain"] for lg in lgA])
    print("estimation set: %d statements; model says Yes on %d of them (self-report direction)" % (len(est), int(self_yes.sum())), flush=True)
    dirs = {}
    missing = set()
    for l in Lall:
        HB = aB[("resid", l)].numpy().astype(np.float64); HA = aA[("resid", l)].numpy().astype(np.float64)
        wE, _ = mean_diff_direction(HB[y_est], HB[~y_est])
        das_meta = None
        if args.e_estimator == "probe":                       # probe-weight direction (same 192 statements), unit norm, true = positive
            from sklearn.linear_model import LogisticRegression
            mu, sd = HB.mean(0), HB.std(0) + 1e-6
            clf = LogisticRegression(C=1.0, max_iter=2000).fit((HB - mu) / sd, y_est.astype(int))
            wE = unit(clf.coef_[0] / sd)                      # back to raw activation space
        elif args.e_estimator == "das":                       # DAS direction trained at this layer; seeds averaged, sign-aligned to mean-diff (true = positive)
            wD, das_meta = load_das_direction(args.das_dir, "E", l, variant=args.das_variant, align_to=wE)
            if wD is None:
                raise FileNotFoundError("missing DAS direction at layer %d in %s (variant %r); refusing an incomplete layer comparison"
                                        % (l, args.das_dir, args.das_variant))
            wE = wD
            if any(n != len(est) // 2 for n in das_meta["n_pairs"]):
                msg = "L%02d: DAS trained on %s pairs but the estimation set holds %d pairs" % (l, das_meta["n_pairs"], len(est) // 2)
                if args.est_extra_pairs > 0:
                    raise SystemExit(msg + " (both estimators must see the same data in the estimation-set size experiment)")
                if args.teacher_split:   # review round 14 (6.2): --plan-split selects on the dev pairs, so the directions must come from calib only
                    raise SystemExit(msg + " (--plan-split selects on the dev pairs; load DAS directions trained on the calib pairs only)")
                print("note: " + msg, flush=True)
            print("L%02d  DAS: %d seed(s), final loss %s, cos(seed, mean) %s, cos(DAS, meandiff) %+.3f" % (
                l, das_meta["n_seeds"], ["%.4f" % x for x in das_meta["final_losses"]], ["%.3f" % x for x in das_meta["cos_seed_vs_mean"]], das_meta["cos_with_align_to"]), flush=True)
        cB = (HB[y_est].mean(0) + HB[~y_est].mean(0)) / 2; cA = HA.mean(0)
        wCp, mCp = mean_diff_direction(aConf[("resid", l)].numpy().astype(np.float64), aCaut[("resid", l)].numpy().astype(np.float64))
        cosCp = float(np.dot(wCp, wE)); wCp = unit(wCp - cosCp * wE)
        if self_yes.any() and (~self_yes).any() and not args.no_cs:
            wCs, mCs = mean_diff_direction(HA[self_yes], HA[~self_yes]); cosCs = float(np.dot(wCs, wE)); wCs = unit(wCs - cosCs * wE); gCs = float(mCs["gap"])
        else:
            wCs, cosCs, gCs = None, float("nan"), float("nan")
        dirs[l] = {"wE": wE, "cA": cA, "cB": cB, "wCp": wCp, "gCp": float(mCp["gap"]), "cosCp": cosCp, "wCs": wCs, "gCs": gCs, "cosCs": cosCs, "das_meta": das_meta}
        print("L%02d  gap_Cprompt %.2f cos(Cp,E) %+.3f | gap_Cself %.2f cos(Cs,E) %+.3f" % (l, mCp["gap"], cosCp, gCs, cosCs), flush=True)
    # ---- conditions -------------------------------------------------------- #
    conds = [("baseline", None, None)]
    for l in L:
        d = dirs[l]
        for a in ([] if args.skip_e else args.e_alphas):
            conds.append(("E_L%d_a%g" % (l, a), [SteerSpec(d["wE"], l, mode="amplify", alpha=a, center=d["cA"])], [SteerSpec(d["wE"], l, mode="amplify", alpha=a, center=d["cB"])]))
        for a in args.c_alphas:
            for sgn, nm in ((-1, "abst"), (1, "ans")):
                if not args.skip_cp:
                    sp = [SteerSpec(d["wCp"], l, mode="add", alpha=sgn * a, coef=[d["gCp"]])]
                    conds.append(("Cp_L%d_%s_a%g" % (l, nm, a), sp, sp))
                if d["wCs"] is not None:
                    ss = [SteerSpec(d["wCs"], l, mode="add", alpha=sgn * a, coef=[d["gCs"]])]
                    conds.append(("Cs_L%d_%s_a%g" % (l, nm, a), ss, ss))
    for sp in ([] if args.skip_e else spans):                 # joint multi-layer E (the chain's main E configuration)
        for a in args.e_alphas:
            conds.append(("E_L%s_a%g" % ("-".join(map(str, sp)), a),
                          [SteerSpec(dirs[l]["wE"], l, mode="amplify", alpha=a, center=dirs[l]["cA"]) for l in sp],
                          [SteerSpec(dirs[l]["wE"], l, mode="amplify", alpha=a, center=dirs[l]["cB"]) for l in sp]))
    print("%d conditions" % len(conds), flush=True)

    def run(texts, specs, lab_map, stage):
        out = []
        for i in range(0, len(texts), bs):
            out.append(next_token_logits(bundle, bundle.encode(texts[i:i + bs]), specs).numpy())
        lg = np.concatenate(out)
        if stage == "A":
            return np.array([label_logprobs_from_logits(x, lab_map)["Answer"] > label_logprobs_from_logits(x, lab_map)["Abstain"] for x in lg])
        # Stage B returns p_true (renormalised over True/False) AND the log-odds margin log p_True - log p_False (float64).
        # p_true saturates to exactly 1.0 / 0.0 once |margin| exceeds ~36 nat (Gemma-3-12B: 93% of statements), which ties
        # the AUROC; the margin keeps the ranking among confident statements. J (threshold 0.5 <=> margin 0) is unchanged.
        # Added after the Mistral/Gemma results (review round 7 candidate 2); stored as <cond>__margin and AUROC_margin.
        lp = [label_logprobs_from_logits(x, lab_map, use_abstain=False) for x in lg]
        return e22.tf_scores(lg, lab_map), np.array([d["True"] - d["False"] for d in lp], dtype=np.float64)

    res = {"model": mcfg["name"], "n": len(recs), "n_est": len(est), "layers": L, "conds": {},
           "num_layers": bundle.num_layers, "all_layers_requested": args.all_layers,
           "tune_pairs": args.tune_pairs, "teacher_split": args.teacher_split,
           "est_extra_pairs": args.est_extra_pairs, "est_extra_ids": sorted(extra_ids),
           "wording": args.wording, "direction_prompt_allow_idk": False,
           "e_alphas": args.e_alphas,
           "e_estimator": args.e_estimator,
           "das": ({"dir": args.das_dir, "variant": args.das_variant, "dropped_layers": sorted(missing),
                    "per_layer": {str(l): dirs[l]["das_meta"] for l in dirs}} if args.e_estimator == "das" else None)}
    per = {}
    for name, sA, sB in conds:
        ans = run(tA, sA, mA, "A"); p, mg = run(tB, sB, mB, "B")
        ans_e = run(eA, sA, mA, "A"); p_e, mg_e = run(eB, sB, mB, "B")    # the 96 estimation pairs (directions were fitted here)
        per[name] = (ans, p, mg)
        m = {}
        for hname, mask in (("A", half_a), ("B", ~half_a), ("all", np.ones(len(recs), dtype=bool))):
            pred = p[mask] > 0.5; lab = labels[mask]
            m[hname] = {"ans": float(ans[mask].mean()), "J": e22.j_stat(pred, lab), "AUROC": float(auroc(p[mask], lab)), "B": float(pred.mean()),
                        "AUROC_margin": float(auroc(mg[mask], lab))}
        pred_e = p_e > 0.5
        m["est"] = {"ans": float(ans_e.mean()), "J": e22.j_stat(pred_e, y_est), "AUROC": float(auroc(p_e, y_est)), "B": float(pred_e.mean()),
                    "AUROC_margin": float(auroc(mg_e, y_est))}
        if args.teacher_split:
            ans_d = run(dA, sA, mA, "A"); p_d, mg_d = run(dB, sB, mB, "B"); pred_d = p_d > 0.5
            m["dev"] = {"ans": float(ans_d.mean()), "J": e22.j_stat(pred_d, y_dev), "AUROC": float(auroc(p_d, y_dev)), "B": float(pred_d.mean()),
                        "AUROC_margin": float(auroc(mg_d, y_dev))}
        res["conds"][name] = m
        b = res["conds"]["baseline"]["all"]
        print("  %-18s ans %.3f (%+.3f) | J %+.3f (%+.3f) | AUROC %.3f (%+.3f) | B %.3f (%+.3f)  (%.0fs)" % (
            name, m["all"]["ans"], m["all"]["ans"] - b["ans"], m["all"]["J"], m["all"]["J"] - b["J"], m["all"]["AUROC"], m["all"]["AUROC"] - b["AUROC"], m["all"]["B"], m["all"]["B"] - b["B"], time.time() - t0), flush=True)

    # ---- pre-declared selection on half A, report on half B ------------ #
    def delta(name, h, key):
        return res["conds"][name][h][key] - res["conds"]["baseline"][h][key]
    # Review round 13/14 (2026-11-04): (i) ties are compared on exact integer counts (core.metrics.sel_key), so two
    # conditions with the same number of correct tuning statements are equal and the FIRST in sweep order wins, as the
    # rule states -- the raw float dJ let a 1.1e-16 rounding difference pick Qwen3-8B's L35 a2 over L34 a3; (ii) when no
    # condition satisfies the answer-rate constraint the selection is None ("none feasible"), never a violating fallback.
    ys = {"A": labels[half_a], "B": labels[~half_a], "all": labels, "est": y_est}
    if args.teacher_split:
        ys["dev"] = y_dev

    def scale_J(h):          # dJ on split h is an exact multiple of 1/(n_pos*n_neg)
        return int(ys[h].sum()) * int((~ys[h]).sum())

    def scale_rate(h):       # d answer rate on split h is an exact multiple of 1/n_items
        return int(len(ys[h]))

    e_c = [n for n in res["conds"] if n.startswith("E_")] or ["baseline"]

    def pick_E(h):
        """E rule on split h: feasible = |d ans| <= 0.03; best = max dJ; ties -> first in sweep order. (None, 0) if none feasible."""
        ok = [n for n in e_c if abs(delta(n, h, "ans")) <= 0.03 + 1e-12]
        return (max(ok, key=lambda n: sel_key(delta(n, h, "J"), scale_J(h))) if ok else None), len(ok)

    def pick_C(h, cands, sign):
        """C rule on split h: feasible = |dJ| <= 0.02, |dB| <= 0.05, moves the answer rate the requested way; best = max |d ans|."""
        ok = [n for n in cands if abs(delta(n, h, "J")) <= 0.02 + 1e-12 and abs(delta(n, h, "B")) <= 0.05 + 1e-12 and sign * delta(n, h, "ans") > 0]
        return (max(ok, key=lambda n: sel_key(sign * delta(n, h, "ans"), scale_rate(h))) if ok else None), len(ok)

    def report(name, h):
        return {k: delta(name, h, k) for k in ("ans", "J", "AUROC", "B")} if name else None

    def c_groups():
        for kind in ("Cp", "Cs"):
            for dname, sign in (("abst", -1), ("ans", 1)):
                cands = [n for n in res["conds"] if n.startswith(kind + "_") and ("_%s_" % dname) in n]
                if cands:
                    yield "%s_%s" % (kind, dname), cands, sign

    sel = {}
    bestE, n_ok = pick_E("A")
    sel["E"] = {"chosen": bestE, "feasible_on_A": n_ok, "B_half": report(bestE, "B"), "A_half": report(bestE, "A")}
    for key, cands, sign in c_groups():
        best, n_ok = pick_C("A", cands, sign)
        sel[key] = {"chosen": best, "feasible_on_A": n_ok, "B_half": report(best, "B"), "A_half": report(best, "A")}
    res["selection"] = sel
    sel_est = {}
    bestE, n_ok = pick_E("est")
    sel_est["E"] = {"chosen": bestE, "feasible_on_est": n_ok, "test_all": report(bestE, "all")}
    for key, cands, sign in c_groups():
        best, n_ok = pick_C("est", cands, sign)
        sel_est[key] = {"chosen": best, "feasible_on_est": n_ok, "test_all": report(best, "all")}
    res["selection_on_est"] = sel_est
    if args.teacher_split:
        sel_dev = {}
        bestE, n_ok = pick_E("dev"); oracleE, _ = pick_E("all")          # oracle obeys the same leak constraint
        sel_dev["E"] = {"chosen": bestE, "oracle_on_test": oracleE, "feasible_on_dev": n_ok, "test_all": report(bestE, "all")}
        for key, cands, sign in c_groups():
            best, n_ok = pick_C("dev", cands, sign); oracle, _ = pick_C("all", cands, sign)
            sel_dev[key] = {"chosen": best, "oracle_on_test": oracle, "feasible_on_dev": n_ok, "test_all": report(best, "all")}
        res["selection_on_dev"] = sel_dev
    out_dir = os.path.join(RESULTS_DIR, "E23", mcfg["name"]); os.makedirs(out_dir, exist_ok=True)
    stem = "knob_sweep" + (("_" + args.tag) if args.tag else "")
    res["elapsed_s"] = time.time() - t0
    atomic_json_dump(res, os.path.join(out_dir, stem + ".json"))
    np.savez(os.path.join(out_dir, stem + ".npz"), ids=np.array([r["id"] for r in recs]), labels=labels, half_a=half_a,
             **{n + "__answer": v[0] for n, v in per.items()}, **{n + "__ptrue": v[1] for n, v in per.items()},
             **{n + "__margin": v[2] for n, v in per.items()})
    lines = ["# E23 knob sweep, %s (n=%d; layers %s; wording %s; E estimator %s%s; estimation set %d statements%s)" % (
                 mcfg["name"], len(recs), L, args.wording, args.e_estimator,
                 (" variant=%s dir=%s" % (args.das_variant or "default", args.das_dir)) if args.e_estimator == "das" else "",
                 len(est), (" incl. %d pairs moved from the evaluation pool" % args.est_extra_pairs) if args.est_extra_pairs else ""), "",
             "Selection on half A (E: max dJ s.t. |d ans| <= 0.03; C: max |d ans| s.t. |dJ| <= 0.02, |dB| <= 0.05), reported on half B.", "",
             "| knob | chosen | feasible on A | half B: d ans | d J | d AUROC | d B |", "|---|---|---|---|---|---|---|"]
    for k, v in sel.items():
        if v["chosen"]:
            h = v["B_half"]; lines.append("| %s | %s | %d | %+.3f | %+.3f | %+.3f | %+.3f |" % (k, v["chosen"], v["feasible_on_A"], h["ans"], h["J"], h["AUROC"], h["B"]))
        else:
            lines.append("| %s | none feasible | 0 | | | | |" % k)
    lines += ["", "## Selection on the %d estimation pairs instead (reported on ALL %d evaluation pairs)" % (len(est) // 2, len(set(map(str, pair_ids)))), "",
              "| knob | chosen on est | feasible on est | test: d ans | d J | d AUROC | d B |", "|---|---|---|---|---|---|---|"]
    for k, v in sel_est.items():
        if v["chosen"]:
            h = v["test_all"]; lines.append("| %s | %s | %d | %+.3f | %+.3f | %+.3f | %+.3f |" % (k, v["chosen"], v["feasible_on_est"], h["ans"], h["J"], h["AUROC"], h["B"]))
        else:
            lines.append("| %s | none feasible | 0 | | | | |" % k)
    if args.teacher_split:
        lines += ["", "## Original plan's split: directions from calib (32 pairs), selection on dev (64 pairs), reported on ALL test pairs; oracle = best on test", "", "| knob | chosen on dev | oracle on test | feasible on dev | test: d ans | d J | d AUROC | d B |", "|---|---|---|---|---|---|---|---|"]
        for k, v in res["selection_on_dev"].items():
            if v["chosen"]:
                h = v["test_all"]; lines.append("| %s | %s | %s | %d | %+.3f | %+.3f | %+.3f | %+.3f |" % (k, v["chosen"], v["oracle_on_test"] or "none feasible", v["feasible_on_dev"], h["ans"], h["J"], h["AUROC"], h["B"]))
            else:
                lines.append("| %s | none feasible | %s | 0 | | | | |" % (k, v["oracle_on_test"] or "none feasible"))
    lines += ["", "## All conditions (all items; d = change vs baseline)", "", "| condition | ans (d) | J (d) | AUROC (d) | B (d) |", "|---|---|---|---|---|"]
    b = res["conds"]["baseline"]["all"]
    for name in res["conds"]:
        m = res["conds"][name]["all"]
        lines.append("| %s | %.3f (%+.3f) | %+.3f (%+.3f) | %.3f (%+.3f) | %.3f (%+.3f) |" % (name, m["ans"], m["ans"] - b["ans"], m["J"], m["J"] - b["J"], m["AUROC"], m["AUROC"] - b["AUROC"], m["B"], m["B"] - b["B"]))
    if args.tune_pairs > 0:
        n_rest = len(set(map(str, pair_ids))) - args.tune_pairs
        lines = [ln.replace("half A", "tune set (%d pairs)" % args.tune_pairs).replace("half B", "remaining test (%d pairs)" % n_rest) for ln in lines]
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()

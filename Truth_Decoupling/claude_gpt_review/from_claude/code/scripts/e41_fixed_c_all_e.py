#!/usr/bin/env python3
"""E41: 赵磊's protocol for E+C on true/false judgement, run as stated (Claude, 2026-11-07).

Fix the commitment direction C at the layer / estimator / sign / strength that the formal 2x2 selected on the tuning set
(report 11 section 8, results_das_tfce/E23/<model>/report_2x2_das_tfce.md), then sweep the evidence direction E over EVERY
decoder layer (alphas 2 and 3) and evaluate baseline, C alone, E alone and E+C under the two-pass protocol of
e23_knob_sweep.py: pass A = "Do you know the answer to this question? Yes/No" (answer rate), pass B = forced True/False
(J, AUROC, bias). Directions are estimated with the same code path as e23_knob_sweep.py (same estimation statements, same
DAS files, C orthogonalised against the same-layer DAS-E of the 2x2), so baseline, E-only and C-only conditions must
reproduce the two 2x2 sweeps; --ref-e-npz / --ref-c-npz check this per item and the baseline mismatch is fatal.
New per-item output: the pass-A gate margin log P(Yes) - log P(No), which review_checks/e41_analysis.py needs to compare
arms at the same number of self-selected answers. Nothing here reads a label to steer or to select.

usage: e41_fixed_c_all_e.py --exp E2b_tqa --model llama3.1-8b --tune-pairs 192 --e-estimator das \
         --das-dir <results_das_tfce>/E1/llama3.1-8b --das-variant tf-ce-bidir \
         --c-abst Cp_L25_abst_a2 --c-ans Cp_L26_ans_a2 --tag das [--limit N] [--ref-e-npz F] [--ref-c-npz F]
output: $TD_RESULTS_DIR/E41/<model>/e41_<tag>.{json,md,npz}
"""
import argparse
import hashlib
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e22_two_stage as e22                                      # noqa: E402
from common import RESULTS_DIR, load_split, tf_prompt            # noqa: E402
from core.config import load_experiment, load_model_config       # noqa: E402
from core.directions import load_das_direction, mean_diff_direction  # noqa: E402
from core.intervene import SteerSpec, next_token_logits          # noqa: E402
from core.io_utils import atomic_json_dump, set_seed             # noqa: E402
from core.metrics import auroc                                   # noqa: E402
from core.model_io import capture_last_token, load_model         # noqa: E402
from core.readout import build_label_token_map, label_logprobs_from_logits  # noqa: E402

C_NAME = re.compile(r"^(Cp|Cs)_L(\d+)_(abst|ans)_a([0-9.]+)$")


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parse_c(name):
    m = C_NAME.match(name)
    if not m:
        raise SystemExit("C condition must look like Cp_L25_abst_a2 or Cs_L33_ans_a2, got %r" % name)
    return {"name": name, "kind": m.group(1), "layer": int(m.group(2)), "sign": -1 if m.group(3) == "abst" else 1, "alpha": float(m.group(4))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa")
    ap.add_argument("--model", required=True)
    ap.add_argument("--e-alphas", nargs="+", type=float, default=[2.0, 3.0])
    ap.add_argument("--wording", default="pik")
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N evaluation statements only")
    ap.add_argument("--tune-pairs", type=int, default=192, help="same seeded 1:2:7 tuning draw as e23_knob_sweep.py")
    ap.add_argument("--e-estimator", choices=["meandiff", "das"], default="das")
    ap.add_argument("--das-dir", required=True, help="DAS directions (also used to orthogonalise C, see --c-orth)")
    ap.add_argument("--das-variant", default="tf-ce-bidir")
    ap.add_argument("--c-orth", choices=["das", "e"], default="das",
                    help="C is orthogonalised against the DAS-E of its layer (the 2x2's definition, default) or against the E estimator used here")
    ap.add_argument("--c-abst", required=True, help="2x2-chosen C toward abstention, e.g. Cp_L25_abst_a2")
    ap.add_argument("--c-ans", required=True, help="2x2-chosen C toward answering, e.g. Cp_L26_ans_a2")
    ap.add_argument("--ref-e-npz", default=None, help="knob_sweep_das_tfce_full.npz of the 2x2: baseline and E-only conditions must match")
    ap.add_argument("--ref-c-npz", default=None, help="knob_sweep_das_tfce_C.npz of the 2x2: the two C-only conditions must match")
    ap.add_argument("--tag", default="das")
    args = ap.parse_args()
    e22.SPLIT_A = dict(e22.SPLIT_WORDINGS[args.wording])
    cfix = [parse_c(args.c_abst), parse_c(args.c_ans)]
    if cfix[0]["sign"] != -1 or cfix[1]["sign"] != 1:
        raise SystemExit("--c-abst must be an _abst_ condition and --c-ans an _ans_ condition")

    exp = load_experiment(args.exp)
    mcfg = load_model_config(args.model)
    prompts = exp["prompts"]; st = prompts["statement"]; cc = prompts["commitment_contrast"]
    set_seed(exp.get("seed", 42)); bs = exp.get("batch_size", 8)
    t0 = time.time()

    # ---- data: same loading, same seeded tuning draw as e23_knob_sweep.py --tune-pairs ------------------------------ #
    recs = []
    for ds in exp["eval_datasets"]:
        recs += load_split(ds, exp.get("eval_split", "test"))
    if args.limit:
        recs = recs[: args.limit]
    rng = np.random.default_rng(exp.get("seed", 42))
    labels = np.array([bool(r["label"]) for r in recs])
    pair_ids = [r.get("pair_id") or r["id"] for r in recs]
    uniq = sorted(set(map(str, pair_ids)))
    tune = set(rng.choice(uniq, min(args.tune_pairs, len(uniq) - 1), replace=False).tolist())
    half_a = np.array([str(p) in tune for p in pair_ids])
    print("evaluation: %d statements; tune %d statements; test %d statements" % (len(recs), int(half_a.sum()), int((~half_a).sum())), flush=True)
    est = []
    for ds in exp["eval_datasets"]:
        est += load_split(ds, "calib") + load_split(ds, "dev")
    y_est = np.array([bool(r["label"]) for r in est])
    bundle = load_model(mcfg)
    L = list(range(bundle.num_layers))
    for c in cfix:
        if c["layer"] not in L:
            raise SystemExit("C layer %d outside this model's %d layers" % (c["layer"], bundle.num_layers))

    # ---- texts, label maps (as the sweep) ---------------------------------------------------------------------------- #
    tA = [bundle.build_chat(st["system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in recs]
    tB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in recs]
    mA = build_label_token_map(bundle.tokenizer, tA[0], e22.SPLIT_A["variants"])
    mB = build_label_token_map(bundle.tokenizer, tB[0], {"True": st["label_variants"]["True"], "False": st["label_variants"]["False"]})
    eA = [bundle.build_chat(st["system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    eB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
    conf = [bundle.build_chat(cc["confident_system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    caut = [bundle.build_chat(cc["cautious_system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]

    # ---- directions at every layer, exactly as e23_knob_sweep.py builds them ---------------------------------------- #
    aB, _ = capture_last_token(bundle, eB, L, components=("resid",), batch_size=bs)
    aA, lgA = capture_last_token(bundle, eA, L, components=("resid",), batch_size=bs, return_logits=True)
    aConf, _ = capture_last_token(bundle, conf, L, components=("resid",), batch_size=bs)
    aCaut, _ = capture_last_token(bundle, caut, L, components=("resid",), batch_size=bs)
    self_yes = np.array([label_logprobs_from_logits(lg.numpy(), mA)["Answer"] > label_logprobs_from_logits(lg.numpy(), mA)["Abstain"] for lg in lgA])
    print("estimation set: %d statements; model says Yes on %d (self-report direction)" % (len(est), int(self_yes.sum())), flush=True)
    dirs, das_meta_all = {}, {}
    for l in L:
        HB = aB[("resid", l)].numpy().astype(np.float64); HA = aA[("resid", l)].numpy().astype(np.float64)
        wMD, _ = mean_diff_direction(HB[y_est], HB[~y_est])
        wDAS, das_meta = load_das_direction(args.das_dir, "E", l, variant=args.das_variant, align_to=wMD)
        if wDAS is None:
            raise FileNotFoundError("missing DAS direction at layer %d in %s (variant %r)" % (l, args.das_dir, args.das_variant))
        das_meta_all[str(l)] = das_meta
        wE = wDAS if args.e_estimator == "das" else wMD
        cB = (HB[y_est].mean(0) + HB[~y_est].mean(0)) / 2; cA = HA.mean(0)
        wOrth = wDAS if args.c_orth == "das" else wE                  # the 2x2's C was orthogonalised against the DAS-E of its layer
        wCp, mCp = mean_diff_direction(aConf[("resid", l)].numpy().astype(np.float64), aCaut[("resid", l)].numpy().astype(np.float64))
        cosCp = float(np.dot(wCp, wOrth)); wCp = unit(wCp - cosCp * wOrth)
        if self_yes.any() and (~self_yes).any():
            wCs, mCs = mean_diff_direction(HA[self_yes], HA[~self_yes]); cosCs = float(np.dot(wCs, wOrth)); wCs = unit(wCs - cosCs * wOrth); gCs = float(mCs["gap"])
        else:
            wCs, cosCs, gCs = None, float("nan"), float("nan")
        dirs[l] = {"wE": wE, "cA": cA, "cB": cB, "wCp": wCp, "gCp": float(mCp["gap"]), "cosCp": cosCp, "wCs": wCs, "gCs": gCs, "cosCs": cosCs,
                   "cos_E_vs_md": float(np.dot(wE, wMD))}
    c_specs, c_meta = {}, {}
    for c in cfix:
        d = dirs[c["layer"]]
        w, gap, cos = (d["wCp"], d["gCp"], d["cosCp"]) if c["kind"] == "Cp" else (d["wCs"], d["gCs"], d["cosCs"])
        if w is None:
            raise SystemExit("self-report C unavailable at layer %d" % c["layer"])
        c_specs[c["name"]] = SteerSpec(w, c["layer"], mode="add", alpha=c["sign"] * c["alpha"], coef=[gap])
        c_meta[c["name"]] = dict(c, gap=gap, cos_with_orth_E_before=cos, direction_sha256=hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest(),
                                 orthogonalised_against="DAS %s E at the same layer" % args.das_variant if args.c_orth == "das" else "the %s E at the same layer" % args.e_estimator)
        print("fixed C %s: gap %.2f, cos(C, E) before orthogonalisation %+.3f" % (c["name"], gap, cos), flush=True)

    # ---- conditions: baseline, C alone, then per layer E alone and E + each C ------------------------------------------ #
    conds = [("baseline", [], [])]
    for c in cfix:
        conds.append((c["name"], [c_specs[c["name"]]], [c_specs[c["name"]]]))
    for l in L:
        d = dirs[l]
        for a in args.e_alphas:
            eA_spec = SteerSpec(d["wE"], l, mode="amplify", alpha=a, center=d["cA"]); eB_spec = SteerSpec(d["wE"], l, mode="amplify", alpha=a, center=d["cB"])
            conds.append(("E_L%d_a%g" % (l, a), [eA_spec], [eB_spec]))
            for c in cfix:
                conds.append(("E_L%d_a%g+%s" % (l, a, c["name"]), [eA_spec, c_specs[c["name"]]], [eB_spec, c_specs[c["name"]]]))
    print("%d conditions" % len(conds), flush=True)

    def run(texts, specs, lab_map, stage):
        out = []
        for i in range(0, len(texts), bs):
            out.append(next_token_logits(bundle, bundle.encode(texts[i:i + bs]), specs or None).numpy())
        lg = np.concatenate(out)
        if stage == "A":
            lp = [label_logprobs_from_logits(x, lab_map) for x in lg]
            gate = np.array([d["Answer"] - d["Abstain"] for d in lp], dtype=np.float64)
            return np.array([d["Answer"] > d["Abstain"] for d in lp]), gate
        lp = [label_logprobs_from_logits(x, lab_map, use_abstain=False) for x in lg]
        return e22.tf_scores(lg, lab_map), np.array([d["True"] - d["False"] for d in lp], dtype=np.float64)

    # ---- reference sweeps of the 2x2 (per-item arrays aligned by statement id) ------------------------------------- #
    refs = {}
    for key, path in (("E", args.ref_e_npz), ("C", args.ref_c_npz)):
        if path:
            z = np.load(path, allow_pickle=True)
            pos = {str(i): k for k, i in enumerate(z["ids"])}
            idx = np.array([pos.get(str(r["id"]), -1) for r in recs])
            if (idx < 0).any():
                raise SystemExit("%d evaluation statements missing from %s" % (int((idx < 0).sum()), path))
            refs[key] = (z, idx)
    ref_check = {}

    def check(name, ans, p, mg):
        key = "C" if name.startswith("C") else "E"
        if key not in refs or name not in {k.rsplit("__", 1)[0] for k in refs[key][0].files}:
            return
        z, idx = refs[key]
        same_ans = bool(np.array_equal(z[name + "__answer"][idx], ans))
        dp = float(np.abs(z[name + "__ptrue"][idx] - p).max()); dm = float(np.abs(z[name + "__margin"][idx] - mg).max())
        ref_check[name] = {"answer_identical": same_ans, "max_abs_diff_ptrue": dp, "max_abs_diff_margin": dm}
        if not (same_ans and dp == 0.0 and dm == 0.0):
            print("  REFERENCE MISMATCH %s: answer identical %s, ptrue diff %.3g, margin diff %.3g" % (name, same_ans, dp, dm), flush=True)
            if name == "baseline":
                raise SystemExit("baseline does not reproduce the 2x2 sweep: different code path or data; refusing to continue")

    def metrics(ans, p, mg, mask):
        pred = p[mask] > 0.5; lab = labels[mask]; a = ans[mask]; correct = pred == lab
        return {"n": int(mask.sum()), "ans": float(a.mean()), "J": e22.j_stat(pred, lab), "AUROC": float(auroc(p[mask], lab)), "B": float(pred.mean()),
                "AUROC_margin": float(auroc(mg[mask], lab)), "acc_all": float(correct.mean()),
                "n_answered": int(a.sum()), "correct_answered": int(correct[a].sum()), "acc_answered": float(correct[a].mean()) if a.any() else float("nan")}

    res = {"model": mcfg["name"], "protocol": "e41_fixed_c_all_e_v1", "n": len(recs), "n_est": len(est), "layers": L, "num_layers": bundle.num_layers,
           "tune_pairs": args.tune_pairs, "limit": args.limit, "wording": args.wording, "e_alphas": args.e_alphas, "e_estimator": args.e_estimator,
           "das": {"dir": args.das_dir, "variant": args.das_variant, "per_layer": das_meta_all}, "c_orth": args.c_orth, "fixed_C": c_meta,
           "cos_E_vs_meandiff_per_layer": {str(l): dirs[l]["cos_E_vs_md"] for l in L},
           "code_sha256": {f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in
                           ("e41_fixed_c_all_e.py", "e22_two_stage.py", "common.py", "../core/intervene.py", "../core/directions.py", "../core/readout.py", "../core/model_io.py", "../core/metrics.py")},
           "conditions": [c[0] for c in conds], "conds": {}, "reference_check": ref_check, "completed": False}
    out_dir = os.path.join(RESULTS_DIR, "E41", mcfg["name"]); os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "e41_" + args.tag)
    per = {}
    for k, (name, sA, sB) in enumerate(conds):
        ans, gate = run(tA, sA, mA, "A"); p, mg = run(tB, sB, mB, "B")
        per[name] = (ans, gate, p, mg)
        check(name, ans, p, mg)
        res["conds"][name] = {"all": metrics(ans, p, mg, np.ones(len(recs), dtype=bool)), "tune": metrics(ans, p, mg, half_a), "test": metrics(ans, p, mg, ~half_a)}
        b = res["conds"]["baseline"]["test"]; m = res["conds"][name]["test"]
        print("  [%3d/%d] %-26s test: ans %.3f (%+.3f) | J %+.3f (%+.3f) | AUROC %.3f (%+.3f) | answered %d, correct among them %d  (%.0fs)" % (
            k + 1, len(conds), name, m["ans"], m["ans"] - b["ans"], m["J"], m["J"] - b["J"], m["AUROC"], m["AUROC"] - b["AUROC"],
            m["n_answered"], m["correct_answered"], time.time() - t0), flush=True)
        if (k + 1) % 20 == 0 or k + 1 == len(conds):     # checkpoint: a crash keeps what was computed
            res["elapsed_s"] = time.time() - t0
            atomic_json_dump(res, stem + ".json")
            np.savez(stem + ".npz", ids=np.array([r["id"] for r in recs]), labels=labels, half_a=half_a,
                     **{n + "__answer": v[0] for n, v in per.items()}, **{n + "__gate": v[1] for n, v in per.items()},
                     **{n + "__ptrue": v[2] for n, v in per.items()}, **{n + "__margin": v[3] for n, v in per.items()})
    res["completed"] = True; res["elapsed_s"] = time.time() - t0
    atomic_json_dump(res, stem + ".json")
    lines = ["# E41 %s: fixed 2x2 C (%s, %s) x E at every layer (%s, alphas %s); n=%d, tune %d / test %d statements" % (
        mcfg["name"], args.c_abst, args.c_ans, args.e_estimator, args.e_alphas, len(recs), int(half_a.sum()), int((~half_a).sum())), "",
        "Reference check against the 2x2 sweeps: %d conditions compared, %d identical" % (
            len(ref_check), sum(1 for v in ref_check.values() if v["answer_identical"] and v["max_abs_diff_ptrue"] == 0 and v["max_abs_diff_margin"] == 0)), "",
        "## All conditions (test statements; d = change vs baseline; 'answered' = pass-A Yes, 'acc' = pass-B accuracy among answered)", "",
        "| condition | ans (d) | J (d) | AUROC (d) | B (d) | answered | correct among answered | acc among answered |", "|---|---|---|---|---|---|---|---|"]
    b = res["conds"]["baseline"]["test"]
    for name in res["conditions"]:
        m = res["conds"][name]["test"]
        lines.append("| %s | %.3f (%+.3f) | %+.3f (%+.3f) | %.3f (%+.3f) | %.3f (%+.3f) | %d | %d | %.3f |" % (
            name, m["ans"], m["ans"] - b["ans"], m["J"], m["J"] - b["J"], m["AUROC"], m["AUROC"] - b["AUROC"], m["B"], m["B"] - b["B"],
            m["n_answered"], m["correct_answered"], m["acc_answered"]))
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("E41 DONE %s %s: %d conditions in %.0f min -> %s" % (mcfg["name"], args.tag, len(conds), (time.time() - t0) / 60, stem), flush=True)


if __name__ == "__main__":
    main()

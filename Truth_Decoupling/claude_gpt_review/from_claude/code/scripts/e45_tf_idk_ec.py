#!/usr/bin/env python3
"""E45: true/false judgement WITH an Unknown option in the same pass - does E change the answer rate, does C bring it back,
and does E+C beat "one direction + the model's own confidence" at the same number of answers (Claude, 2026-11-12;
plan from_claude/40, written before any result; 赵磊 2026-11-12 02:54).

Directions are the main 2x2's, estimated with exactly E41's code path (e41_fixed_c_all_e.py lines 117-162):
  E = DAS tf-ce-bidir at the 2x2-chosen layer, amplify (h <- h + a * w w^T (h - centre));
  C = the 2x2-chosen Cp / Cs (estimated from pass-A states, orthogonalised against the same-layer DAS-E), add
      (h <- h + s * a * gap * u); verified against E41's recorded direction hashes when --ref-e41-json is given.
NEW here:
  * ONE pass with the prompt that allows Unknown (default_v1.json statement.user_tf_idk), read as a three-way arg-max over
    the True / False / Unknown label variants (core.readout.decide_label); Unknown = abstain.
  * E's amplify centre is recomputed under that prompt: c_idk = (mu_true + mu_false) / 2 of the estimation statements at the
    E layer (the forced-prompt centre would add the constant a * w^T (c_idk - c_forced) w - the old bug, notes line 75); the
    forced centre is kept for one diagnostic condition only.
  * A secondary C estimated under the same prompt (CpI = confident minus cautious system prompt with the Unknown-allowed
    statement prompt, common.commitment_prompts, orthogonalised against the same-layer DAS-E, same layers and signs as the
    2x2's C), in case the 2x2's C (estimated at pass A) does not transfer to this prompt.
Per statement and condition: answer (not Unknown), gate = max(logp True, logp False) - logp Unknown, margin = logp True -
logp False (log-probs, no saturation), ptrue = P(True | True or False). Nothing reads a label to steer or to select.

usage: e45_tf_idk_ec.py --model llama3.1-8b --das-dir <results_das_tfce>/E1/llama3.1-8b --e-layer 11 \
         --c-abst Cp_L25_abst_a2 --c-ans Cp_L26_ans_a2 --tag v1 [--ref-e41-json F] [--limit N]
output: $TD_RESULTS_DIR/E45/<model>/e45_<tag>.{json,md,npz}
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e22_two_stage as e22                                      # noqa: E402
from common import RESULTS_DIR, commitment_prompts, load_split, tf_prompt  # noqa: E402
from core.config import load_experiment, load_model_config       # noqa: E402
from core.directions import load_das_direction, mean_diff_direction  # noqa: E402
from core.intervene import SteerSpec, next_token_logits          # noqa: E402
from core.io_utils import atomic_json_dump, set_seed             # noqa: E402
from core.metrics import auroc                                   # noqa: E402
from core.model_io import capture_last_token, load_model         # noqa: E402
from core.readout import build_label_token_map, decide_label, label_logprobs_from_logits, p_true_given_answer  # noqa: E402

C_NAME = re.compile(r"^(Cp|Cs)_L(\d+)_(abst|ans)_a([0-9.]+)$")
ALPHAS_E = [1.0, 2.0, 3.0]
ALPHAS_C = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def vsha(w):
    return hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()


def parse_c(name):
    m = C_NAME.match(name)
    if not m:
        raise SystemExit("C condition must look like Cp_L25_abst_a2 or Cs_L33_ans_a2, got %r" % name)
    return {"name": name, "kind": m.group(1), "layer": int(m.group(2)), "sign": -1 if m.group(3) == "abst" else 1,
            "side": m.group(3), "alpha_2x2": float(m.group(4))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa")
    ap.add_argument("--model", required=True)
    ap.add_argument("--wording", default="pik")
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N evaluation statements only")
    ap.add_argument("--tune-pairs", type=int, default=192, help="same seeded tuning draw as e23_knob_sweep.py / E41")
    ap.add_argument("--das-dir", required=True)
    ap.add_argument("--das-variant", default="tf-ce-bidir")
    ap.add_argument("--e-layer", type=int, required=True, help="the 2x2-chosen E layer")
    ap.add_argument("--e-alpha", type=float, default=3.0, help="the 2x2-chosen E strength, used in every E+C condition")
    ap.add_argument("--c-abst", required=True, help="the 2x2-chosen C toward abstention, e.g. Cp_L25_abst_a2 (kind, layer, sign)")
    ap.add_argument("--c-ans", required=True, help="the 2x2-chosen C toward answering, e.g. Cp_L26_ans_a2")
    ap.add_argument("--ref-e41-json", default=None, help="E41 json of this model: the 2x2 C must reproduce its recorded direction hashes")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    e22.SPLIT_A = dict(e22.SPLIT_WORDINGS[args.wording])
    cfix = [parse_c(args.c_abst), parse_c(args.c_ans)]
    if cfix[0]["sign"] != -1 or cfix[1]["sign"] != 1:
        raise SystemExit("--c-abst must be an _abst_ condition and --c-ans an _ans_ condition")
    if args.e_alpha not in ALPHAS_E:
        raise SystemExit("--e-alpha must be one of %s" % ALPHAS_E)

    exp = load_experiment(args.exp)
    mcfg = load_model_config(args.model)
    prompts = exp["prompts"]; st = prompts["statement"]; cc = prompts["commitment_contrast"]
    set_seed(exp.get("seed", 42)); bs = exp.get("batch_size", 8)
    t0 = time.time()

    # ---- data: E41's loading and seeded tuning draw ------------------------------------------------------------------ #
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
    for l in [args.e_layer] + [c["layer"] for c in cfix]:
        if not 0 <= l < bundle.num_layers:
            raise SystemExit("layer %d outside this model's %d layers" % (l, bundle.num_layers))
    L = sorted({args.e_layer, cfix[0]["layer"], cfix[1]["layer"]})

    # ---- texts: E41's estimation texts, plus the Unknown-allowed prompt ------------------------------------------------ #
    tA0 = bundle.build_chat(st["system"], e22.split_user(recs[0], e22.SPLIT_A["instr"]))
    mA = build_label_token_map(bundle.tokenizer, tA0, e22.SPLIT_A["variants"])
    eA = [bundle.build_chat(st["system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    eB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
    conf = [bundle.build_chat(cc["confident_system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    caut = [bundle.build_chat(cc["cautious_system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in est]
    tI = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in recs]
    eI = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est]
    confI, cautI = zip(*[commitment_prompts(bundle, prompts, r["text"], is_statement=True) for r in est])
    mI = build_label_token_map(bundle.tokenizer, tI[0], st["label_variants"])
    if set(mI) != {"True", "False", "Abstain"}:
        raise SystemExit("the Unknown-allowed readout needs True / False / Abstain label variants, got %s" % sorted(mI))

    # ---- directions (E41's estimates at the needed layers) + the Unknown-prompt centre and CpI ------------------------- #
    aB, _ = capture_last_token(bundle, eB, L, components=("resid",), batch_size=bs)
    aA, lgA = capture_last_token(bundle, eA, L, components=("resid",), batch_size=bs, return_logits=True)
    aConf, _ = capture_last_token(bundle, conf, L, components=("resid",), batch_size=bs)
    aCaut, _ = capture_last_token(bundle, caut, L, components=("resid",), batch_size=bs)
    aI, _ = capture_last_token(bundle, eI, L, components=("resid",), batch_size=bs)
    aConfI, _ = capture_last_token(bundle, list(confI), L, components=("resid",), batch_size=bs)
    aCautI, _ = capture_last_token(bundle, list(cautI), L, components=("resid",), batch_size=bs)
    self_yes = np.array([label_logprobs_from_logits(lg.numpy(), mA)["Answer"] > label_logprobs_from_logits(lg.numpy(), mA)["Abstain"] for lg in lgA])
    print("estimation set: %d statements; model says Yes on %d (self-report direction)" % (len(est), int(self_yes.sum())), flush=True)
    dirs, das_meta_all = {}, {}
    for l in L:
        HB = aB[("resid", l)].numpy().astype(np.float64); HA = aA[("resid", l)].numpy().astype(np.float64)
        HI = aI[("resid", l)].numpy().astype(np.float64)
        wMD, _ = mean_diff_direction(HB[y_est], HB[~y_est])
        wDAS, das_meta = load_das_direction(args.das_dir, "E", l, variant=args.das_variant, align_to=wMD)
        if wDAS is None:
            raise FileNotFoundError("missing DAS direction at layer %d in %s (variant %r)" % (l, args.das_dir, args.das_variant))
        das_meta_all[str(l)] = das_meta
        cB = (HB[y_est].mean(0) + HB[~y_est].mean(0)) / 2
        cI = (HI[y_est].mean(0) + HI[~y_est].mean(0)) / 2
        wCp, mCp = mean_diff_direction(aConf[("resid", l)].numpy().astype(np.float64), aCaut[("resid", l)].numpy().astype(np.float64))
        cosCp = float(np.dot(wCp, wDAS)); wCp = unit(wCp - cosCp * wDAS)
        if self_yes.any() and (~self_yes).any():
            wCs, mCs = mean_diff_direction(HA[self_yes], HA[~self_yes]); cosCs = float(np.dot(wCs, wDAS)); wCs = unit(wCs - cosCs * wDAS); gCs = float(mCs["gap"])
        else:
            wCs, cosCs, gCs = None, float("nan"), float("nan")
        wCpI, mCpI = mean_diff_direction(aConfI[("resid", l)].numpy().astype(np.float64), aCautI[("resid", l)].numpy().astype(np.float64))
        cosCpI = float(np.dot(wCpI, wDAS)); wCpI = unit(wCpI - cosCpI * wDAS)
        wMDI, _ = mean_diff_direction(HI[y_est], HI[~y_est])
        proj = HI @ wDAS
        dirs[l] = {"wE": wDAS, "cB": cB, "cI": cI, "wCp": wCp, "gCp": float(mCp["gap"]), "cosCp": cosCp, "wCs": wCs, "gCs": gCs, "cosCs": cosCs,
                   "wCpI": wCpI, "gCpI": float(mCpI["gap"]), "cosCpI": cosCpI,
                   "centre_shift_along_E_in_sd": float(np.dot(cI - cB, wDAS) / (proj.std() + 1e-12)),
                   "cos_E_vs_idk_meandiff": float(np.dot(wDAS, wMDI)), "cos_Cp_vs_CpI": float(np.dot(wCp, wCpI)),
                   "cos_Cs_vs_CpI": float(np.dot(wCs, wCpI)) if wCs is not None else float("nan")}
    dE = dirs[args.e_layer]
    print("E L%d: centre shift (Unknown prompt minus forced prompt) along E = %+.3f sd; cos(E, Unknown-prompt true-false mean difference) %+.3f" % (
        args.e_layer, dE["centre_shift_along_E_in_sd"], dE["cos_E_vs_idk_meandiff"]), flush=True)

    # ---- C specs: the 2x2's C (hash-checked) and CpI, each at every alpha of the grid ---------------------------------- #
    ref = json.load(open(args.ref_e41_json)) if args.ref_e41_json else None
    c_meta, c_specs = {}, {}
    for c in cfix:
        d = dirs[c["layer"]]
        w, gap, cos = (d["wCp"], d["gCp"], d["cosCp"]) if c["kind"] == "Cp" else (d["wCs"], d["gCs"], d["cosCs"])
        if w is None:
            raise SystemExit("self-report C unavailable at layer %d" % c["layer"])
        if ref is not None:
            rm = ref["fixed_C"].get(c["name"])
            if rm is None or rm.get("direction_sha256") != vsha(w) or abs(float(rm["gap"]) - gap) > 1e-6 * max(1.0, abs(gap)):
                raise SystemExit("the 2x2 C %s does not reproduce E41's direction / gap" % c["name"])
        for variant, (wv, gv, cv) in (("C", (w, gap, cos)), ("CpI", (d["wCpI"], d["gCpI"], d["cosCpI"]))):
            label = "%s_L%d_%s" % (c["kind"] if variant == "C" else "CpI", c["layer"], c["side"])
            c_meta[label] = {"variant": variant, "kind": c["kind"] if variant == "C" else "CpI", "layer": c["layer"], "sign": c["sign"], "gap": gv,
                             "cos_with_E_before_orthogonalisation": cv, "direction_sha256": vsha(wv),
                             "estimated_from": "pass A (2x2 / E41 code path)" if variant == "C" else "Unknown-allowed statement prompt, confident minus cautious system"}
            for a in ALPHAS_C:
                c_specs["%s_a%g" % (label, a)] = SteerSpec(wv, c["layer"], mode="add", alpha=c["sign"] * a, coef=[gv])
        print("C %s: gap %.3f, cos(C, E) before orthogonalisation %+.3f; CpI at L%d: gap %.3f, cos(Cp-or-Cs, CpI) %+.3f" % (
            c["name"], gap, cos, c["layer"], d["gCpI"], d["cos_Cp_vs_CpI"] if c["kind"] == "Cp" else d["cos_Cs_vs_CpI"]), flush=True)

    # ---- conditions ------------------------------------------------------------------------------------------------------ #
    e_specs = {a: SteerSpec(dE["wE"], args.e_layer, mode="amplify", alpha=a, center=dE["cI"]) for a in ALPHAS_E}
    conds = [("baseline", [])] + [("E_a%g" % a, [e_specs[a]]) for a in ALPHAS_E]
    conds.append(("E_a%g_forcedcentre" % args.e_alpha, [SteerSpec(dE["wE"], args.e_layer, mode="amplify", alpha=args.e_alpha, center=dE["cB"])]))
    for name, spec in c_specs.items():
        conds.append((name, [spec]))
        conds.append(("E_a%g+%s" % (args.e_alpha, name), [e_specs[args.e_alpha], spec]))
    print("%d conditions" % len(conds), flush=True)

    def run(specs):
        out = []
        for i in range(0, len(tI), bs):
            out.append(next_token_logits(bundle, bundle.encode(tI[i:i + bs]), specs or None).numpy())
        lg = np.concatenate(out)
        lp = [label_logprobs_from_logits(x, mI, use_abstain=True) for x in lg]
        pred = [decide_label(d) for d in lp]
        ans = np.array([p != "Abstain" for p in pred])
        lpT = np.array([d["True"] for d in lp], dtype=np.float64)
        lpF = np.array([d["False"] for d in lp], dtype=np.float64)
        lpU = np.array([d["Abstain"] for d in lp], dtype=np.float64)
        gate = np.maximum(lpT, lpF) - lpU
        margin = lpT - lpF
        ptrue = 1.0 / (1.0 + np.exp(-np.clip(margin, -700.0, 700.0)))      # P(True | True or False), overflow-safe
        if not np.array_equal(ans, gate >= 0):
            raise RuntimeError("three-way arg-max and the answer-vs-Unknown gate disagree")
        return ans, gate, margin, ptrue, lpT, lpF, lpU

    def metrics(ans, p, mg, mask):
        pred = p[mask] > 0.5; lab = labels[mask]; a = ans[mask]; correct = pred == lab
        return {"n": int(mask.sum()), "ans": float(a.mean()), "J": e22.j_stat(pred, lab), "AUROC": float(auroc(p[mask], lab)),
                "AUROC_margin": float(auroc(mg[mask], lab)), "B": float(pred.mean()), "n_answered": int(a.sum()),
                "correct_answered": int(correct[a].sum()), "acc_answered": float(correct[a].mean()) if a.any() else float("nan")}

    res = {"model": mcfg["name"], "protocol": "e45_tf_idk_ec_v1", "n": len(recs), "n_est": len(est), "num_layers": bundle.num_layers,
           "tune_pairs": args.tune_pairs, "limit": args.limit, "wording": args.wording, "e_layer": args.e_layer, "e_alpha": args.e_alpha,
           "alphas_E": ALPHAS_E, "alphas_C": ALPHAS_C, "das": {"dir": args.das_dir, "variant": args.das_variant, "per_layer": das_meta_all},
           "fixed_C_2x2": {c["name"]: c for c in cfix}, "C": c_meta, "ref_e41_json": args.ref_e41_json,
           "E_diagnostics": {k: dE[k] for k in ("centre_shift_along_E_in_sd", "cos_E_vs_idk_meandiff")},
           "code_sha256": {f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in
                           ("e45_tf_idk_ec.py", "e22_two_stage.py", "common.py", "../core/intervene.py", "../core/directions.py",
                            "../core/readout.py", "../core/model_io.py", "../core/metrics.py")},
           "conditions": [c[0] for c in conds], "conds": {}, "completed": False}
    out_dir = os.path.join(RESULTS_DIR, "E45", mcfg["name"]); os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "e45_" + args.tag)
    if os.path.exists(stem + ".json"):
        raise SystemExit("%s.json exists" % stem)
    per = {}
    for k, (name, specs) in enumerate(conds):
        ans, gate, mg, p, lpT, lpF, lpU = run(specs)
        per[name] = (ans, gate, p, mg, lpT, lpF, lpU)
        res["conds"][name] = {"all": metrics(ans, p, mg, np.ones(len(recs), dtype=bool)), "tune": metrics(ans, p, mg, half_a), "test": metrics(ans, p, mg, ~half_a)}
        b = res["conds"]["baseline"]["test"]; m = res["conds"][name]["test"]
        print("  [%3d/%d] %-34s test: ans %.3f (%+.3f) | J %+.3f (%+.3f) | AUROC %.3f (%+.3f) | answered %d, correct among them %d  (%.0fs)" % (
            k + 1, len(conds), name, m["ans"], m["ans"] - b["ans"], m["J"], m["J"] - b["J"], m["AUROC"], m["AUROC"] - b["AUROC"],
            m["n_answered"], m["correct_answered"], time.time() - t0), flush=True)
        if (k + 1) % 20 == 0 or k + 1 == len(conds):
            res["elapsed_s"] = time.time() - t0
            atomic_json_dump(res, stem + ".json")
            np.savez(stem + ".npz", ids=np.array([r["id"] for r in recs]), labels=labels, half_a=half_a,
                     **{n + "__answer": v[0] for n, v in per.items()}, **{n + "__gate": v[1] for n, v in per.items()},
                     **{n + "__ptrue": v[2] for n, v in per.items()}, **{n + "__margin": v[3] for n, v in per.items()},
                     **{n + "__lpT": v[4] for n, v in per.items()}, **{n + "__lpF": v[5] for n, v in per.items()},
                     **{n + "__lpU": v[6] for n, v in per.items()})
    res["completed"] = True; res["elapsed_s"] = time.time() - t0
    atomic_json_dump(res, stem + ".json")
    lines = ["# E45 %s: Unknown-allowed true/false pass; E = DAS L%d (amplify, Unknown-prompt centre), C = 2x2 %s / %s and CpI; n=%d, tune %d / test %d" % (
        mcfg["name"], args.e_layer, args.c_abst, args.c_ans, len(recs), int(half_a.sum()), int((~half_a).sum())), "",
        "| condition | ans (d) | J (d) | AUROC (d) | answered | correct among answered | acc among answered |", "|---|---|---|---|---|---|---|"]
    b = res["conds"]["baseline"]["test"]
    for name in res["conditions"]:
        m = res["conds"][name]["test"]
        lines.append("| %s | %.3f (%+.3f) | %+.3f (%+.3f) | %.3f (%+.3f) | %d | %d | %.3f |" % (
            name, m["ans"], m["ans"] - b["ans"], m["J"], m["J"] - b["J"], m["AUROC"], m["AUROC"] - b["AUROC"], m["n_answered"], m["correct_answered"], m["acc_answered"]))
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("E45 DONE %s %s: %d conditions in %.0f min -> %s" % (mcfg["name"], args.tag, len(conds), (time.time() - t0) / 60, stem), flush=True)


if __name__ == "__main__":
    main()

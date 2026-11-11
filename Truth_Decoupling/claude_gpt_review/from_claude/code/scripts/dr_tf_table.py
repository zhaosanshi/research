#!/usr/bin/env python3
"""The Dr's true/false comparison table (plan of 2026-11-10): original model vs mean-difference E vs DAS E, per model.

For every model, the two E23 sweeps that differ only in the E estimator (DAS `--das-tag`, mean difference `--md-tag`; same
items, same tuning split, same prompts, alphas and rule) are read; each estimator's condition is chosen on the tuning pairs
by the sweep's own pre-declared rule (max dJ s.t. |d answer rate| <= 0.03, ties to the first condition in sweep order;
e23_compare_estimators.rule_pick, single-layer conditions on the layers both sweeps cover). Nothing here looks at test items
before the choice. The chosen conditions are then reported on two pre-declared report sets:
  * "100 pairs": the Dr's 100 pre-fixed pairs (--pairs-file, drawn before any new-model run), the same for every model
  * "529 pairs": all test pairs of the split (the 100 are a subset of these; the two sets are not independent)
Per set: AUROC (on p_true, as in all earlier reports; AUROC on the log-odds margin alongside, which cannot saturate), J,
self-report Yes rate of the separate "Do you know the answer?" pass (NOT the actual answer rate: every item is answered
True/False in the scored pass), and B (share of True answers; 0.5 = no bias on these balanced pairs). Differences
DAS - original, mean-difference - original and DAS - mean-difference with pair-bootstrap 95% CIs (whole pairs resampled,
the same resamples for all arms of a model).

usage: dr_tf_table.py --root $RES --models qwen3.5-9b qwen3.8-27b --pairs-file configs/splits/dr_100pairs_20261110.json --out $RES/dr_tf_table
output: <out>.md, <out>.json (refuses to overwrite)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from core.metrics import auroc                          # noqa: E402  same tie-aware AUROC as the sweeps
import e23_compare_estimators as cmp                     # noqa: E402  the sweep's rule and tie-break (rule_pick, layers_of)

ARMS = ("original", "meandiff", "DAS")
KEYS = ("AUROC", "AUROC_margin", "J", "yes_rate", "B")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def metrics(ans, p, mg, lab):
    pred = p > 0.5
    am = float(auroc(mg, lab)) if np.all(np.isfinite(mg)) else float("nan")      # sweeps older than review round 7 stored no margin
    return {"AUROC": float(auroc(p, lab)), "AUROC_margin": am, "J": float(pred[lab].mean() - pred[~lab].mean()),
            "yes_rate": float(ans.mean()), "B": float(pred.mean())}


def margin_of(z, name):
    key = name + "__margin"
    return z[key].astype(float) if key in z.files else np.full(len(z["ids"]), np.nan)


def load(root, model, tag):
    d = os.path.join(root, "E23", model)
    jp, zp = os.path.join(d, "knob_sweep_%s.json" % tag), os.path.join(d, "knob_sweep_%s.npz" % tag)
    return json.load(open(jp)), np.load(zp, allow_pickle=True), {"json": jp, "npz": zp, "json_sha256": sha(jp), "npz_sha256": sha(zp)}


def ci(arr):
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--das-tag", default="das_tfce_full")
    ap.add_argument("--md-tag", default="meandiff_full")
    ap.add_argument("--md-root", default=None, help="results root of the mean-difference sweep when it lives elsewhere (default: --root)")
    ap.add_argument("--das-variant", default="tf-ce-bidir", help="the DAS sweep must use this DAS variant")
    ap.add_argument("--pairs-file", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output stem (.md and .json); refuses to overwrite")
    args = ap.parse_args()
    for ext in (".md", ".json"):
        if os.path.exists(args.out + ext):
            raise SystemExit("refusing to overwrite %s" % (args.out + ext))
    pf = json.load(open(args.pairs_file))
    dr_pairs = set(pf["pair_ids"])
    if len(dr_pairs) != 100:
        raise SystemExit("the pairs file must hold 100 distinct pair ids")
    rep = {"pairs_file": args.pairs_file, "pairs_file_sha256": sha(args.pairs_file), "das_tag": args.das_tag, "md_tag": args.md_tag,
           "n_boot": args.n_boot, "seed": args.seed, "rule": "E: max dJ on the tuning pairs s.t. |d answer rate| <= %.2f; ties -> first in sweep order" % cmp.E_LEAK,
           "models": {}}
    for model in args.models:
        jd, zd, fd = load(args.root, model, args.das_tag)
        jm, zm, fm = load(args.md_root or args.root, model, args.md_tag)
        if jd.get("e_estimator") != "das" or jm.get("e_estimator") != "meandiff":
            raise SystemExit("%s: --das-tag must be a DAS sweep and --md-tag a mean-difference sweep" % model)
        if (jd.get("das") or {}).get("variant") != args.das_variant:
            raise SystemExit("%s: DAS sweep variant %r, expected %r" % (model, (jd.get("das") or {}).get("variant"), args.das_variant))
        for k in ("ids", "half_a", "labels"):
            if not np.array_equal(zd[k], zm[k]):
                raise SystemExit("%s: the two sweeps differ in %s" % (model, k))
        for k in ("model", "wording", "e_alphas", "n", "n_est", "tune_pairs"):
            if jd.get(k) != jm.get(k):
                raise SystemExit("%s: the two sweeps differ in %s (%r vs %r)" % (model, k, jd.get(k), jm.get(k)))
        both_margin = "baseline__margin" in zd.files and "baseline__margin" in zm.files
        same_base = all(np.array_equal(zd["baseline__" + k], zm["baseline__" + k]) for k in ("answer", "ptrue") + (("margin",) if both_margin else ()))
        if not same_base:
            raise SystemExit("%s: baseline arrays differ between the sweeps (not the same run conditions)" % model)
        lab = zd["labels"].astype(bool)
        tune = zd["half_a"].astype(bool)
        pair = np.array([str(i).rsplit(":", 1)[0] for i in zd["ids"]])
        missing = dr_pairs - set(pair[~tune])
        if missing:
            raise SystemExit("%s: %d of the 100 pairs are not in the test half (e.g. %s)" % (model, len(missing), sorted(missing)[:3]))
        if set(pair[tune]) & dr_pairs:
            raise SystemExit("%s: some of the 100 pairs are tuning pairs" % model)
        # ---- choice on the tuning pairs only ---- #
        layer_sets = [{tuple(cmp.layers_of(n)[0]) for n in j["conds"] if n.startswith("E_") and cmp.layers_of(n)[0]} for j in (jd, jm)]
        common = layer_sets[0] & layer_sets[1]
        scale_A = int(lab[tune].sum()) * int((~lab[tune]).sum())
        chosen = {}
        for arm, j in (("DAS", jd), ("meandiff", jm)):
            names = [n for n in j["conds"] if n.startswith("E_") and cmp.layers_of(n)[0] and len(cmp.layers_of(n)[0]) == 1
                     and tuple(cmp.layers_of(n)[0]) in common]
            chosen[arm] = cmp.rule_pick(j["conds"], names, scale_A)
        arrays = {"original": (zd["baseline__answer"].astype(bool), zd["baseline__ptrue"].astype(float), margin_of(zd, "baseline"))}
        for arm, z in (("DAS", zd), ("meandiff", zm)):
            n = chosen[arm]
            if n is not None:
                arrays[arm] = (z[n + "__answer"].astype(bool), z[n + "__ptrue"].astype(float), margin_of(z, n))
        b = jd["conds"]["baseline"]["A"]
        try:
            from core.config import load_model_config
            mc = load_model_config(model)
            model_cfg = {k: mc.get(k) for k in ("name", "path", "modelscope_repo", "hidden_size", "num_layers", "dtype", "python")}
        except Exception as e:                          # an old model whose config is not in this tree: record why
            model_cfg = {"error": repr(e)}
        mrep = {"files": {"DAS": fd, "meandiff": fm}, "chosen": chosen, "margin_stored_in_both": both_margin, "model_config": model_cfg,
                "runtime": {"DAS": jd.get("runtime"), "meandiff": jm.get("runtime")},
                "tune": {arm: ({k: jj["conds"][chosen[arm]]["A"][k] - b[k] for k in ("J", "ans", "AUROC", "B")} if chosen[arm] else None)
                         for arm, jj in (("DAS", jd), ("meandiff", jm))},
                "das_per_layer_chosen": (jd.get("das") or {}).get("per_layer", {}).get(str(cmp.layers_of(chosen["DAS"])[0][0])) if chosen["DAS"] else None,
                "sets": {}}
        n_test = len(set(pair[~tune]))
        if n_test != 529:
            raise SystemExit("%s: %d test pairs, the protocol's split has 529" % (model, n_test))
        for set_name, mask in (("100 pairs", (~tune) & np.isin(pair, sorted(dr_pairs))), ("529 pairs", ~tune)):
            pairs_here = np.array(sorted(set(pair[mask])))
            where = {p: np.where((pair == p) & mask)[0] for p in pairs_here}
            if any(len(v) != 2 for v in where.values()):
                raise SystemExit("%s: a report pair does not have exactly two statements" % model)
            rng = np.random.default_rng(args.seed)
            boot = [np.concatenate([where[p] for p in rng.choice(pairs_here, len(pairs_here), replace=True)]) for _ in range(args.n_boot)]
            idx = np.where(mask)[0]
            point = {arm: metrics(*(a[idx] for a in arrays[arm]), lab[idx]) for arm in arrays}
            bs = {arm: {k: np.array([metrics(*(a[ii] for a in arrays[arm]), lab[ii])[k] for ii in boot]) for k in KEYS} for arm in arrays}
            diffs = {}
            for a1, a0 in (("DAS", "original"), ("meandiff", "original"), ("DAS", "meandiff")):
                if a1 in arrays and a0 in arrays:
                    diffs["%s - %s" % (a1, a0)] = {k: {"point": point[a1][k] - point[a0][k], "ci95": ci(bs[a1][k] - bs[a0][k]),
                                                       "p_gt0": float(((bs[a1][k] - bs[a0][k]) > 0).mean())} for k in KEYS}
            mrep["sets"][set_name] = {"n_pairs": len(pairs_here), "point": point, "diffs": diffs}
        rep["models"][model] = mrep
    # ---- markdown ---- #
    out = ["# True/false comparison on the new models: original model, mean difference, DAS", "",
           "Rule on the tuning pairs (%s); reported on the Dr's 100 pre-fixed pairs (`%s`) and on all 529 test pairs (the 100 are part of them). "
           "Pair-bootstrap 95%% CIs, %d resamples, seed %d, the same resamples for every arm of a model. AUROC on p_true as in all earlier reports; "
           "AUROC on the log-odds margin alongside. Yes rate = the separate self-report pass (\"Do you know the answer?\"), not an actual answer rate. "
           "B = share of True answers (0.5 = no bias)." % (rep["rule"], os.path.basename(args.pairs_file), args.n_boot, args.seed), ""]
    for set_name in ("100 pairs", "529 pairs"):
        out += ["## %s" % set_name, "", "| model | arm | chosen on tuning pairs | AUROC | AUROC (margin) | J | Yes rate | B |", "|---|---|---|---|---|---|---|---|"]
        for model, m in rep["models"].items():
            s = m["sets"][set_name]
            for arm in ARMS:
                if arm not in s["point"]:
                    out.append("| %s | %s | none feasible | | | | | |" % (model, arm))
                    continue
                p = s["point"][arm]
                out.append("| %s | %s | %s | %.3f | %.3f | %+.3f | %.3f | %.3f |" % (model, arm, m["chosen"].get(arm, "-") if arm != "original" else "-",
                                                                                p["AUROC"], p["AUROC_margin"], p["J"], p["yes_rate"], p["B"]))
        out += ["", "| model | difference | AUROC [95% CI] | AUROC (margin) [CI] | J [CI] | Yes rate [CI] | B [CI] |", "|---|---|---|---|---|---|---|"]
        for model, m in rep["models"].items():
            for dn, dv in m["sets"][set_name]["diffs"].items():
                out.append("| %s | %s | %s |" % (model, dn, " | ".join("%+.3f [%+.3f, %+.3f]" % (dv[k]["point"], dv[k]["ci95"][0], dv[k]["ci95"][1]) for k in KEYS)))
        out.append("")
    out += ["## Provenance", ""]
    for model, m in rep["models"].items():
        out.append("- %s: DAS `%s` (sha256 %s…), mean difference `%s` (sha256 %s…); tuning-set deltas DAS %s, mean difference %s" % (
            model, m["files"]["DAS"]["npz"], m["files"]["DAS"]["npz_sha256"][:12], m["files"]["meandiff"]["npz"], m["files"]["meandiff"]["npz_sha256"][:12],
            json.dumps(m["tune"]["DAS"]), json.dumps(m["tune"]["meandiff"])))
        out.append("  - model: %s; runtime of the DAS sweep: %s" % (json.dumps(m["model_config"], ensure_ascii=False), json.dumps(m["runtime"]["DAS"])))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()

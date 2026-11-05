"""Review round 15 §2.2: the stricter e23_merge.py guard on REAL cache chunks and on synthetic conflicts. CPU, numpy only.

usage: chk_merge_guard.py CODE_DIR SRC_E23_DIR TAG MODEL OUT_DIR
  1. The real knob_sweep_<TAG>.{json,npz} in SRC_E23_DIR is split into two layer chunks (layers < 18 / >= 18, baseline in both)
     and merged in REVERSED order. The merged selection must equal the unsplit sweep's own selection (chosen E condition and its
     test deltas, copied numbers so exact), and the merged condition set must equal the original: the stricter guard accepts true
     same-run chunks.
  2. Synthetic conflicts derived from the same chunks, each keeping ids / masks / aggregate baseline identical:
     per-item baseline (two same-label test items with opposite predictions swap their p_true), labels, model, estimator,
     and a 1e-7 baseline rank reversal that leaves every threshold prediction unchanged.
     Each must be refused with the named reason.
  3. Missing core run identity or a common model different from --model is refused even when both chunks agree.
     Historical chunks missing the optional est_extra_pairs field remain accepted with the same selected result.
Exit 1 if any expectation fails.
"""
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

CODE, SRC, TAG, MODEL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
rep, fails = {}, []


def expect(name, cond, detail):
    rep[name] = {"ok": bool(cond), "detail": detail}
    if not cond:
        fails.append(name)


def layer_of(name):
    m = re.search(r"_L([\d-]+)_", name)
    return float(np.mean([int(x) for x in m.group(1).split("-")])) if m else None


j = json.load(open(SRC / ("knob_sweep_%s.json" % TAG)))
z = dict(np.load(SRC / ("knob_sweep_%s.npz" % TAG), allow_pickle=True))
conds = [c for c in j["conds"] if c != "baseline"]
lo = [c for c in conds if layer_of(c) is None or layer_of(c) < 18]
hi = [c for c in conds if layer_of(c) is not None and layer_of(c) >= 18]
assert lo and hi, "need conditions on both sides of layer 18"


def write_chunk(root, tag, names, json_mod=None, npz_mod=None):
    d = Path(root) / "results_tqa" / "E23" / MODEL
    d.mkdir(parents=True, exist_ok=True)
    jj = dict(j)
    jj["conds"] = {"baseline": j["conds"]["baseline"], **{c: j["conds"][c] for c in names}}
    jj["layers"] = sorted({int(l) for c in names for l in re.findall(r"_L(\d+)", c)} or j["layers"])
    jj.pop("selection", None); jj.pop("selection_on_est", None)
    if json_mod:
        json_mod(jj)
    zz = {k: v for k, v in z.items() if "__" not in k or k.split("__")[0] in set(names) | {"baseline"}}
    if npz_mod:
        npz_mod(zz)
    json.dump(jj, open(d / ("knob_sweep_%s.json" % tag), "w"), indent=1)
    np.savez(d / ("knob_sweep_%s.npz" % tag), **zz)


def merge(root, tags, out_tag, n_boot=200):
    proc = subprocess.run([sys.executable, str(CODE / "scripts" / "e23_merge.py"), "--root", str(root), "--model", MODEL, "--tags", *tags,
                           "--out-tag", out_tag, "--n-boot", str(n_boot)], capture_output=True, text=True)
    return proc


# 1. real chunks, merged in reversed order
real = OUT / "real"
write_chunk(real, "lo", lo)
write_chunk(real, "hi", hi)
proc = merge(real, ["hi", "lo"], "merged", n_boot=200)
(OUT / "real_merge.log").write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr)
merged = json.load(open(real / "results_tqa" / "E23" / MODEL / "knob_sweep_merged.json")) if proc.returncode == 0 else None
# Reference = the reviewed exact-count selector (e23_compare_estimators.rule_pick, round 14) applied to the unsplit conditions in
# sweep order. NOT the sweep's stored selection: a sweep written before round 14 stored the float tie-break choice (Qwen MD-288: L35 a2).
sys.path[:0] = [str(CODE / "scripts"), str(CODE)]
from e23_compare_estimators import metrics, rule_pick  # noqa: E402
labels_all, tune = z["labels"].astype(bool), z["half_a"].astype(bool)
scale = int(labels_all[tune].sum()) * int((~labels_all[tune]).sum())
ref_chosen = rule_pick(j["conds"], [c for c in j["conds"] if c.startswith("E_")], scale)
stored_chosen = (j.get("selection") or {}).get("E", {}).get("chosen")
exp_test = {k: j["conds"][ref_chosen]["B"][k] - j["conds"]["baseline"]["B"][k] for k in ("ans", "J", "AUROC", "B")} if ref_chosen else None
expect("real_chunks_reversed_merge_accepted", proc.returncode == 0, {"returncode": proc.returncode, "stderr_tail": proc.stderr[-300:]})
expect("real_merge_selection_equals_exact_rule_on_unsplit",
       merged is not None and merged["selection"]["E"]["chosen"] == ref_chosen and merged["selection"]["E"]["test"] == exp_test
       and set(merged["conds"]) == set(j["conds"]),
       {"merged_chosen": merged and merged["selection"]["E"]["chosen"], "exact_rule_on_unsplit": ref_chosen,
        "stored_in_sweep_json (pre-round-14 float tie-break if different)": stored_chosen,
        "merged_test": merged and merged["selection"]["E"]["test"], "expected_test": exp_test,
        "n_conds_merged": merged and len(merged["conds"]), "n_conds_sweep": len(j["conds"])})

# 2. synthetic conflicts (each derived from chunk 'lo' as the SECOND chunk, with 'hi' first)
half_a = z["half_a"].astype(bool); labels = z["labels"].astype(bool)
bp = z["baseline__ptrue"].astype(float)
test_idx = np.where(~half_a)[0]


def swap_two_same_label_opposite_pred(zz):
    pred = zz["baseline__ptrue"].astype(float) > 0.5
    for lab in (True, False):
        pos = [i for i in test_idx if labels[i] == lab and pred[i]]
        neg = [i for i in test_idx if labels[i] == lab and not pred[i]]
        if pos and neg:
            p = zz["baseline__ptrue"].astype(float).copy()
            p[pos[0]], p[neg[0]] = p[neg[0]], p[pos[0]]
            zz["baseline__ptrue"] = p
            return
    raise AssertionError("no swappable test items")


cases = {
    "per_item_baseline": dict(npz_mod=swap_two_same_label_opposite_pred, needle="different per-item baseline"),
    "labels": dict(npz_mod=lambda zz: zz.__setitem__("labels", np.concatenate([~labels[:2], labels[2:]])), needle="different labels"),
    "model": dict(json_mod=lambda jj: jj.__setitem__("model", "OTHER_MODEL"), needle="does not match --model"),
    "estimator": dict(json_mod=lambda jj: jj.__setitem__("e_estimator", "das" if jj.get("e_estimator") != "das" else "meandiff"), needle="not the same run"),
}
for name, c in cases.items():
    root = OUT / ("conflict_" + name)
    write_chunk(root, "hi", hi)
    write_chunk(root, "lo", lo, json_mod=c.get("json_mod"), npz_mod=c.get("npz_mod"))
    proc = merge(root, ["hi", "lo"], "bad", n_boot=20)
    (OUT / ("conflict_%s.log" % name)).write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr)
    expect("conflict_%s_refused" % name, proc.returncode != 0 and c["needle"] in proc.stderr,
           {"returncode": proc.returncode, "stderr_tail": proc.stderr.strip()[-220:]})

# 3. Rank reversal smaller than the former allclose tolerance. Both baseline
# arrays predict True everywhere and have the same ans/J/B, but their AUROCs
# are 0 and 1. Sharing either baseline would corrupt the other chunk's AUC/CI.
near_a = np.where(labels, .6000001, .6000002)
near_b = np.where(labels, .6000002, .6000001)
answers = z["baseline__answer"].astype(bool)


def near_baseline_json(probs):
    def modify(jj):
        baseline = copy.deepcopy(jj["conds"]["baseline"])
        for name, mask in (("A", half_a), ("B", ~half_a), ("all", np.ones(len(labels), dtype=bool))):
            baseline[name].update(metrics(answers[mask], probs[mask], labels[mask]))
        jj["conds"]["baseline"] = baseline
    return modify


near = OUT / "conflict_near_tie_ranking"
write_chunk(near, "hi", hi, json_mod=near_baseline_json(near_a), npz_mod=lambda zz: zz.__setitem__("baseline__ptrue", near_a))
write_chunk(near, "lo", lo, json_mod=near_baseline_json(near_b), npz_mod=lambda zz: zz.__setitem__("baseline__ptrue", near_b))
proc = merge(near, ["hi", "lo"], "bad", n_boot=20)
(OUT / "conflict_near_tie_ranking.log").write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr)
expect("conflict_near_tie_ranking_refused", proc.returncode != 0 and "different per-item baseline" in proc.stderr,
       {"returncode": proc.returncode, "max_abs_probability_diff": float(np.max(np.abs(near_a - near_b))),
        "same_threshold_predictions": bool(np.array_equal(near_a > .5, near_b > .5)),
        "baseline_AUROC_a": metrics(answers, near_a, labels)["AUROC"],
        "baseline_AUROC_b": metrics(answers, near_b, labels)["AUROC"],
        "stderr_tail": proc.stderr.strip()[-220:]})

# 4. Agreement cannot authenticate absent core identity. Test each required
# field independently so leaving any one of them unchecked is observable.
for field in ("model", "e_estimator", "wording"):
    root = OUT / ("missing_core_" + field)
    omit = lambda jj, key=field: jj.pop(key, None)
    write_chunk(root, "hi", hi, json_mod=omit)
    write_chunk(root, "lo", lo, json_mod=omit)
    proc = merge(root, ["hi", "lo"], "bad", n_boot=20)
    (OUT / ("missing_core_%s.log" % field)).write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr)
    expect("missing_core_%s_refused" % field,
           proc.returncode != 0 and "missing or invalid core run identity" in proc.stderr and field in proc.stderr,
           {"returncode": proc.returncode, "stderr_tail": proc.stderr.strip()[-220:]})

root = OUT / "common_wrong_model"
wrong_model = lambda jj: jj.__setitem__("model", "OTHER_MODEL")
write_chunk(root, "hi", hi, json_mod=wrong_model)
write_chunk(root, "lo", lo, json_mod=wrong_model)
proc = merge(root, ["hi", "lo"], "bad", n_boot=20)
(OUT / "common_wrong_model.log").write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr)
expect("common_wrong_model_refused", proc.returncode != 0 and "does not match --model" in proc.stderr,
       {"returncode": proc.returncode, "stderr_tail": proc.stderr.strip()[-220:]})

# est_extra_pairs is optional in historical sweeps. Omitting it from both
# pieces must not be confused with missing the model/estimator/prompt identity.
legacy = OUT / "legacy_missing_optional"
omit_optional = lambda jj: jj.pop("est_extra_pairs", None)
write_chunk(legacy, "hi", hi, json_mod=omit_optional)
write_chunk(legacy, "lo", lo, json_mod=omit_optional)
proc = merge(legacy, ["hi", "lo"], "merged", n_boot=200)
(OUT / "legacy_missing_optional.log").write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr)
legacy_result = json.load(open(legacy / "results_tqa" / "E23" / MODEL / "knob_sweep_merged.json")) if proc.returncode == 0 else None
expect("legacy_missing_optional_field_accepted", proc.returncode == 0 and legacy_result is not None
       and merged is not None and legacy_result["selection"] == merged["selection"]
       and set(legacy_result["conds"]) == set(j["conds"]),
       {"returncode": proc.returncode, "chosen": legacy_result and legacy_result["selection"]["E"]["chosen"],
        "expected_chosen": ref_chosen, "stderr_tail": proc.stderr.strip()[-220:]})

rep["failures"] = fails
rep["source"] = str(SRC / ("knob_sweep_%s.json" % TAG))
rep["chunks"] = {"lo": len(lo), "hi": len(hi)}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "merge_guard.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("PASSED: %d merge-guard expectations" % (len(rep) - 3))

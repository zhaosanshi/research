#!/usr/bin/env python3
"""Concatenate E23 knob-sweep chunks (same run, disjoint layer ranges) into ONE sweep json + npz (Claude, 2026-11-12).

Unlike e23_merge.py (which writes a selection report but no per-item arrays), this produces knob_sweep_<out-tag>.{json,npz}
exactly as a single e23_knob_sweep.py run over the union of the layers would, so report_2x2.py, e23_compare_estimators.py
and dr_tf_table.py read it unchanged. Every condition steers ONE layer, every chunk scores the same items with the same split,
and the sweep's conditions are ordered layer by layer; concatenating chunks in ascending layer order therefore reproduces the
single run's condition order, which the rules' tie-break ("first in sweep order") depends on.

Checks (any failure: exit non-zero, nothing written):
  * identical run identity (model, items, estimation set, split, wording, prompts flag, E alphas, E estimator, DAS dir / variant)
  * identical ids, labels, tuning mask, and bit-identical baseline arrays and baseline metrics in every chunk
  * layer ranges disjoint and not interleaved; with --expect-layers N the union is exactly 0..N-1
  * every non-baseline condition names one layer of its own chunk; the C alpha set is the same in every chunk
  * no condition name appears twice
The merged json drops the chunks' own "selection" / "selection_on_est" (each covered only its chunk's layers; the report
scripts re-apply the rules on the merged conditions) and records "merged_from".

usage: e23_concat_chunks.py --root $RES --model qwen3.8-27b --tags das_tfce_C_c0 das_tfce_C_c1 ... --out-tag das_tfce_C --expect-layers 64
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

IDENTITY = ("model", "n", "n_est", "num_layers", "tune_pairs", "teacher_split", "est_extra_pairs", "est_extra_ids", "wording",
            "direction_prompt_allow_idk", "e_alphas", "e_estimator", "runtime")
LAYER = re.compile(r"^(?:E|Cp|Cs)_L(\d+)_")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def die(msg):
    print("CONCAT REFUSED: " + msg, file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--expect-layers", type=int, default=None)
    args = ap.parse_args()
    d = os.path.join(args.root, "E23", args.model)
    out_json, out_npz = os.path.join(d, "knob_sweep_%s.json" % args.out_tag), os.path.join(d, "knob_sweep_%s.npz" % args.out_tag)
    if os.path.exists(out_json) or os.path.exists(out_npz):
        die("refusing to overwrite %s" % out_json)
    chunks = []
    for t in args.tags:
        jp, zp = os.path.join(d, "knob_sweep_%s.json" % t), os.path.join(d, "knob_sweep_%s.npz" % t)
        j, z = json.load(open(jp)), np.load(zp, allow_pickle=True)
        if list(j["conds"])[0] != "baseline":
            die("%s: first condition is not the baseline" % t)
        layers = list(j["layers"])
        if layers != sorted(layers) or len(set(layers)) != len(layers):
            die("%s: layers not unique ascending" % t)
        chunks.append({"tag": t, "j": j, "z": z, "layers": layers, "files": {"json": jp, "npz": zp, "json_sha256": sha(jp), "npz_sha256": sha(zp)}})
    chunks.sort(key=lambda c: c["layers"][0])
    ref = chunks[0]
    for c in chunks[1:]:
        for k in IDENTITY:
            if c["j"].get(k) != ref["j"].get(k):
                die("%s: %s differs from %s (%r vs %r)" % (c["tag"], k, ref["tag"], c["j"].get(k), ref["j"].get(k)))
        for k in ("dir", "variant"):
            if (c["j"].get("das") or {}).get(k) != (ref["j"].get("das") or {}).get(k):
                die("%s: das.%s differs" % (c["tag"], k))
        for k in ("ids", "labels", "half_a"):
            if not np.array_equal(c["z"][k], ref["z"][k]):
                die("%s: %s differ" % (c["tag"], k))
        for k in [f for f in ref["z"].files if f.startswith("baseline__")]:
            if k not in c["z"].files or not np.array_equal(c["z"][k], ref["z"][k]):
                die("%s: baseline array %s differs (not the same run conditions)" % (c["tag"], k))
        if c["j"]["conds"]["baseline"] != ref["j"]["conds"]["baseline"]:
            die("%s: baseline metrics differ" % c["tag"])
    das_dir = (ref["j"].get("das") or {}).get("dir")
    if das_dir:                 # every chunk must have used the DAS files that are in the directory NOW (no stale chunk mixed in)
        for c in chunks:
            for name, h in ((c["j"].get("das") or {}).get("checkpoint_sha256") or {}).items():
                p = os.path.join(das_dir, name)
                if not os.path.exists(p) or sha(p) != h:
                    die("%s: DAS file %s differs from the one this chunk used (re-trained or re-merged E1?)" % (c["tag"], name))
    for a, b in zip(chunks, chunks[1:]):
        if a["layers"][-1] >= b["layers"][0]:
            die("layer ranges overlap or interleave: %s %s vs %s %s" % (a["tag"], a["layers"], b["tag"], b["layers"]))
    union = [l for c in chunks for l in c["layers"]]
    n_layers = ref["j"].get("num_layers")
    if args.expect_layers is not None and (union != list(range(args.expect_layers)) or n_layers != args.expect_layers):
        die("union of layers %s is not exactly 0..%d (model has %s)" % (union, args.expect_layers - 1, n_layers))
    c_alpha_sets = []
    conds, arrays, seen = {"baseline": ref["j"]["conds"]["baseline"]}, {}, {"baseline"}
    for k in ref["z"].files:
        if k.startswith("baseline__") or k in ("ids", "labels", "half_a"):
            arrays[k] = ref["z"][k]
    for c in chunks:
        alphas = set()
        last_layer = -1
        for name, m in list(c["j"]["conds"].items())[1:]:
            mm = LAYER.match(name)
            if not mm or int(mm.group(1)) not in c["layers"]:
                die("%s: condition %s does not name one layer of this chunk" % (c["tag"], name))
            if int(mm.group(1)) < last_layer:
                die("%s: conditions are not in layer order at %s" % (c["tag"], name))
            last_layer = int(mm.group(1))
            if name in seen:
                die("condition %s appears twice" % name)
            seen.add(name)
            if name.startswith("C"):
                alphas.add(name.rsplit("_a", 1)[1])
            conds[name] = m
            for suf in ("__answer", "__ptrue", "__margin"):
                if name + suf in c["z"].files:
                    arrays[name + suf] = c["z"][name + suf]
                elif suf != "__margin":
                    die("%s: npz lacks %s" % (c["tag"], name + suf))
        c_alpha_sets.append(sorted(alphas))
    if len({tuple(a) for a in c_alpha_sets}) > 1:
        die("C alpha sets differ between chunks: %s" % c_alpha_sets)
    merged = {k: v for k, v in ref["j"].items() if k not in ("conds", "selection", "selection_on_est", "elapsed_s", "layers", "das")}
    merged["layers"] = union
    merged["all_layers_requested"] = n_layers is not None and union == list(range(n_layers))
    merged["conds"] = conds
    if ref["j"].get("das") is not None:
        das = {k: v for k, v in ref["j"]["das"].items() if k not in ("per_layer", "checkpoint_sha256", "dropped_layers")}
        das["per_layer"], das["checkpoint_sha256"], das["dropped_layers"] = {}, {}, []
        for c in chunks:
            das["per_layer"].update(c["j"]["das"].get("per_layer") or {})
            das["checkpoint_sha256"].update(c["j"]["das"].get("checkpoint_sha256") or {})
            das["dropped_layers"] += c["j"]["das"].get("dropped_layers") or []
        merged["das"] = das
    else:
        merged["das"] = None
    merged["elapsed_s"] = sum(float(c["j"].get("elapsed_s") or 0) for c in chunks)
    merged["merged_from"] = [{"tag": c["tag"], "layers": c["layers"], **c["files"]} for c in chunks]
    os.makedirs(d, exist_ok=True)
    np.savez(out_npz, **arrays)
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=1)
    print("concatenated %d chunks -> %s: %d layers, %d conditions" % (len(chunks), out_json, len(union), len(conds)))


if __name__ == "__main__":
    main()

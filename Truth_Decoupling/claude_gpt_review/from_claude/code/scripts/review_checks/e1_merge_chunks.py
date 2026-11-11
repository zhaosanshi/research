#!/usr/bin/env python3
"""Merge E1 DAS runs that trained disjoint layer chunks of ONE model into one E1 directory (Claude, 2026-11-12).

Why: the 27B model's per-layer DAS training (~7.5 min a layer) is split by --das-layers over several GPUs, each chunk under its own
TD_RESULTS_DIR so that no two processes write the same summary.json. train_das() re-seeds at the start of every layer
(core/directions.py: set_seed(seed) and OrthoBasis(seed=seed)), so a layer's direction does not depend on which chunk trained it.

Checks (any failure: exit non-zero, nothing written):
  * every chunk is an E1 directory of the same model: same directions_meandiff.npz arrays (bit-equal), same layer_scores.json,
    same summary fields except the per-layer DAS entries, the trained-layer list, start time and elapsed time
  * the chunks' DAS settings are identical (variant, seeds, loss, bidirectional, positions, anchor, prompt, rank, ce label ids, pairs)
  * trained layers are disjoint; with --expect-layers N their union is exactly 0..N-1; every trained layer has its .pt file(s)
Writes into --out (must not exist or be empty): directions_meandiff.npz and layer_scores.json copied from the first chunk, all DAS
.pt files copied byte for byte, and summary.json = the first chunk's summary with the DAS section merged and a "merged_from" record.

usage: e1_merge_chunks.py --chunks <root_a>/E1/<model> <root_b>/E1/<model> --out <root>/E1/<model> --expect-layers 64
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np

PT = re.compile(r"^das_([CE])_L(\d+)_r(\d+)_s(\d+)(?:_(.+))?\.pt$")
SAME_DAS_KEYS = ("variant", "seeds", "loss", "bidirectional", "positions", "anchor", "prompt_allow_idk", "balanced", "trim", "rank",
                 "ce_label_ids", "n_pairs")
SKIP_TOP = ("started", "elapsed_s", "das", "experiment")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def die(msg):
    print("MERGE REFUSED: " + msg, file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-layers", type=int, default=None)
    args = ap.parse_args()
    if len(args.chunks) < 2:
        die("need at least two chunks")
    if os.path.isdir(args.out) and os.listdir(args.out):
        die("output directory %s is not empty" % args.out)
    sums, trained = [], []
    ref_npz = np.load(os.path.join(args.chunks[0], "directions_meandiff.npz"))
    ref_ls = json.load(open(os.path.join(args.chunks[0], "layer_scores.json")))
    for c in args.chunks:
        for f in ("summary.json", "directions_meandiff.npz", "layer_scores.json"):
            if not os.path.exists(os.path.join(c, f)):
                die("%s has no %s" % (c, f))
        s = json.load(open(os.path.join(c, "summary.json")))
        sums.append(s)
        z = np.load(os.path.join(c, "directions_meandiff.npz"))
        if sorted(z.files) != sorted(ref_npz.files) or any(not np.array_equal(z[k], ref_npz[k]) for k in z.files):
            die("%s: mean-difference directions differ from the first chunk (not the same estimation run)" % c)
        if json.load(open(os.path.join(c, "layer_scores.json"))) != ref_ls:
            die("%s: layer_scores.json differs from the first chunk" % c)
        for k in set(s) | set(sums[0]):
            if k not in SKIP_TOP and s.get(k) != sums[0].get(k):
                die("%s: summary field %r differs from the first chunk" % (c, k))
        d = s.get("das") or {}
        for k in SAME_DAS_KEYS:
            if d.get(k) != (sums[0].get("das") or {}).get(k):
                die("%s: DAS setting %r differs (%r vs %r)" % (c, k, d.get(k), (sums[0].get("das") or {}).get(k)))
        tl = d.get("E_trained_layers")
        if not tl:
            die("%s: no E_trained_layers in summary.json (did the chunk finish?)" % c)
        trained.append(list(tl))
    flat = [l for t in trained for l in t]
    if len(flat) != len(set(flat)):
        die("chunks overlap in trained layers: %s" % sorted({l for l in flat if flat.count(l) > 1}))
    n_layers = sums[0].get("num_layers")
    if args.expect_layers is not None and (sorted(flat) != list(range(args.expect_layers)) or n_layers != args.expect_layers):
        die("trained layers %s (model has %s) are not exactly 0..%d" % (sorted(flat), n_layers, args.expect_layers - 1))
    variant, seeds, rank = sums[0]["das"]["variant"], sums[0]["das"]["seeds"], sums[0]["das"]["rank"]
    files = {}
    for c, tl in zip(args.chunks, trained):
        for l in tl:
            for s in seeds:
                name = "das_E_L%d_r%d_s%d%s.pt" % (l, rank, s, ("_" + variant) if variant else "")
                p = os.path.join(c, name)
                if not os.path.exists(p):
                    die("%s: missing %s" % (c, name))
                files[name] = p
        for p in glob.glob(os.path.join(c, "das_*.pt")):
            m = PT.match(os.path.basename(p))
            if not m or int(m.group(2)) not in tl:
                die("%s: unexpected DAS file %s (layer not in this chunk's trained list)" % (c, os.path.basename(p)))
    merged = json.loads(json.dumps(sums[0]))       # build and check the merged summary BEFORE writing anything (red team 2)
    das = merged["das"]
    for s in sums[1:]:
        for k, v in s["das"].items():
            if re.match(r"^E_L\d+_s\d+$", k) or re.match(r"^E_L\d+_seed_abscos$", k):
                if k in das:
                    die("per-layer DAS entry %s appears in two chunks" % k)
                das[k] = v
    das["E_trained_layers"] = sorted(flat)
    merged["elapsed_s"] = sum(float(s.get("elapsed_s") or 0) for s in sums)
    merged["merged_from"] = [{"dir": c, "trained_layers": tl, "started": s.get("started"), "elapsed_s": s.get("elapsed_s"),
                              "summary_sha256": sha(os.path.join(c, "summary.json"))} for c, tl, s in zip(args.chunks, trained, sums)]
    missing = [l for l in sorted(flat) if "E_L%d_s%d" % (l, seeds[0]) not in das]
    if missing:
        die("merged summary lacks per-layer DAS entries for layers %s" % missing)
    os.makedirs(args.out, exist_ok=True)
    for name, p in sorted(files.items()):
        shutil.copyfile(p, os.path.join(args.out, name))
        if sha(p) != sha(os.path.join(args.out, name)):
            die("copy of %s is not byte-identical" % name)
    shutil.copyfile(os.path.join(args.chunks[0], "directions_meandiff.npz"), os.path.join(args.out, "directions_meandiff.npz"))
    shutil.copyfile(os.path.join(args.chunks[0], "layer_scores.json"), os.path.join(args.out, "layer_scores.json"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(merged, f, indent=1)
    print("merged %d chunks -> %s: %d layers, %d DAS files" % (len(args.chunks), args.out, len(flat), len(files)))


if __name__ == "__main__":
    main()

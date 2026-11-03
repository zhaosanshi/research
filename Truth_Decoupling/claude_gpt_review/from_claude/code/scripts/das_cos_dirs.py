#!/usr/bin/env python3
"""Pairwise |cos| between rank-1 DAS directions across two E1 directories and across seeds.

Two uses (赵磊 2026-11-02, estimation-set size question):
  * dir-a = 96-pair training, dir-b = 288-pair training  -> did more data move the direction?
  * dir-a = seed 42, dir-b = seeds 43/44 at a few layers  -> is the 96-pair direction stable across random inits?
Files: das_<tag>_L<layer>_r1_s<seed>[_<variant>].pt (torch.save of core.directions.train_das). Sign is irrelevant (|cos|).
usage: python3 scripts/das_cos_dirs.py --dir-a A --dir-b B [--tag E] [--variant tf] [--layers 9 10 11] --out X.json
Without --layers: every layer present in BOTH directories. Writes JSON and prints a markdown table. CPU only.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch


def load_dir(d, tag, variant, rank=1):
    pat = re.compile(r"^das_%s_L(\d+)_r%d_s(\d+)%s\.pt$" % (re.escape(tag), rank, ("_" + re.escape(variant)) if variant else ""))
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "das_%s_L*_r%d_s*.pt" % (tag, rank)))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        layer, seed = int(m.group(1)), int(m.group(2))
        res = torch.load(f, map_location="cpu", weights_only=False)
        w = np.asarray(res["W"], dtype=np.float64)[:, 0]
        out.setdefault(layer, {})[seed] = {"w": w / (np.linalg.norm(w) + 1e-12), "loss": float(res["losses"][-1]),
                                           "n_pairs": int(res.get("n_pairs", -1))}
    return out


def abscos(u, v):
    return float(abs(np.dot(u, v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    ap.add_argument("--tag", default="E")
    ap.add_argument("--variant", default="tf")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--out", required=True, help="JSON output (refuses to overwrite); a .md table is written next to it")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("refusing to overwrite %s" % args.out)
    A, B = load_dir(args.dir_a, args.tag, args.variant), load_dir(args.dir_b, args.tag, args.variant)
    layers = args.layers if args.layers else sorted(set(A) & set(B))
    if not layers:
        raise SystemExit("no common layers between %s and %s (tag %s, variant %r)" % (args.dir_a, args.dir_b, args.tag, args.variant))
    rows = {}
    lines = ["# |cos| between DAS directions: `%s` (a) vs `%s` (b), tag %s, variant %s" % (args.dir_a, args.dir_b, args.tag, args.variant), "",
             "| L | a: seeds (loss, n_pairs) | b: seeds (loss, n_pairs) | abscos a x b: mean / min | abscos within a | abscos within b |", "|---|---|---|---|---|---|"]
    for l in layers:
        if l not in A or l not in B:
            lines.append("| %d | %s | %s | missing | | |" % (l, "present" if l in A else "missing", "present" if l in B else "missing"))
            rows[str(l)] = {"missing": True}
            continue
        sa, sb = sorted(A[l]), sorted(B[l])
        cross = {"%d x %d" % (i, j): abscos(A[l][i]["w"], B[l][j]["w"]) for i in sa for j in sb}
        within_a = {"%d x %d" % (i, j): abscos(A[l][i]["w"], A[l][j]["w"]) for k, i in enumerate(sa) for j in sa[k + 1:]}
        within_b = {"%d x %d" % (i, j): abscos(B[l][i]["w"], B[l][j]["w"]) for k, i in enumerate(sb) for j in sb[k + 1:]}
        rows[str(l)] = {"a": {str(s): {"loss": A[l][s]["loss"], "n_pairs": A[l][s]["n_pairs"]} for s in sa},
                        "b": {str(s): {"loss": B[l][s]["loss"], "n_pairs": B[l][s]["n_pairs"]} for s in sb},
                        "abscos_a_x_b": cross, "abscos_within_a": within_a, "abscos_within_b": within_b}
        fmt = lambda dd: ", ".join("s%d (%.3f, %d)" % (s, dd[s]["loss"], dd[s]["n_pairs"]) for s in sorted(dd))
        lines.append("| %d | %s | %s | %.3f / %.3f | %s | %s |" % (
            l, fmt(A[l]), fmt(B[l]), float(np.mean(list(cross.values()))), float(min(cross.values())),
            ", ".join("%s: %.3f" % kv for kv in within_a.items()) or "-", ", ".join("%s: %.3f" % kv for kv in within_b.items()) or "-"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"dir_a": args.dir_a, "dir_b": args.dir_b, "tag": args.tag, "variant": args.variant, "layers": rows}, f, indent=1)
    with open(os.path.splitext(args.out)[0] + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

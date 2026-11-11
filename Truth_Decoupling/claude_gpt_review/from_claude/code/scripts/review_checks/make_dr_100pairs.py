#!/usr/bin/env python3
"""Fix the 100 true/false test pairs of the Dr's 2026-11-10 plan BEFORE any new-model run (Claude, 2026-11-12).

The pairs are drawn from the 529 test pairs of the existing split, i.e. the TruthfulQA test pairs minus the 192 tuning
pairs that e23_knob_sweep.py --tune-pairs 192 draws (np.random.default_rng(42).choice over the sorted unique pair ids).
That tuning draw is recomputed here and checked against the tuning mask stored by an old sweep (--ref-npz), so the 100
pairs are guaranteed to lie outside both the estimation pairs (calib + dev) and the tuning pairs.
The draw itself: np.random.default_rng(20261110).choice(sorted test pair ids, 100, replace=False), written sorted.

usage: make_dr_100pairs.py --data-dir $TD_DATA_DIR --ref-npz <old knob_sweep_*.npz> --out configs/splits/dr_100pairs_20261110.json
"""
import argparse
import hashlib
import json
import os

import numpy as np

SEED_TUNE, N_TUNE = 42, 192          # E2b_tqa seed and the protocol's --tune-pairs
SEED_DR, N_DR = 20261110, 100        # the date of the Dr's plan; 100 pairs as the plan states


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ref-npz", required=True, help="an old E23 sweep npz (ids, half_a) to confirm the tuning draw")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("refusing to overwrite %s" % args.out)
    d = os.path.join(args.data_dir, "truthfulqa_tf")
    test = read_jsonl(os.path.join(d, "test.jsonl"))
    est = read_jsonl(os.path.join(d, "calib.jsonl")) + read_jsonl(os.path.join(d, "dev.jsonl"))
    pid = [str(r.get("pair_id") or r["id"]) for r in test]
    uniq = sorted(set(pid))
    tune = set(np.random.default_rng(SEED_TUNE).choice(uniq, N_TUNE, replace=False).tolist())
    z = np.load(args.ref_npz, allow_pickle=True)
    ref_ids = [str(x) for x in z["ids"]]
    ref_tune = {i.rsplit(":", 1)[0] for i, h in zip(ref_ids, z["half_a"]) if bool(h)}
    if ref_ids != [r["id"] for r in test]:
        raise SystemExit("the reference sweep scored different items than test.jsonl")
    if ref_tune != tune:
        raise SystemExit("recomputed tuning draw differs from the reference sweep's tuning mask")
    rest = sorted(set(uniq) - tune)
    est_pairs = {str(r.get("pair_id") or r["id"]) for r in est}
    if len(rest) != len(uniq) - N_TUNE or set(rest) & est_pairs:
        raise SystemExit("test pairs overlap the estimation pairs or have the wrong size")
    by_pair = {}
    for r in test:
        by_pair.setdefault(str(r.get("pair_id") or r["id"]), []).append(bool(r["label"]))
    chosen = sorted(np.random.default_rng(SEED_DR).choice(rest, N_DR, replace=False).tolist())
    if any(sorted(by_pair[p]) != [False, True] for p in chosen):
        raise SystemExit("a chosen pair is not one true + one false statement")
    cats = {}
    for r in test:
        p = str(r.get("pair_id") or r["id"])
        if p in chosen and r["label"]:
            c = (r.get("meta") or {}).get("category", "?")
            cats[c] = cats.get(c, 0) + 1
    rec = {
        "name": "dr_100pairs_20261110",
        "purpose": "the Dr's plan of 2026-11-10: the same 100 pre-fixed true/false pairs for every new model (report set only)",
        "rule": "np.random.default_rng(%d).choice(sorted remaining test pair ids, %d, replace=False), sorted" % (SEED_DR, N_DR),
        "source_split": "truthfulqa_tf/test.jsonl minus the %d tuning pairs of np.random.default_rng(%d).choice(sorted ids, %d)" % (N_TUNE, SEED_TUNE, N_TUNE),
        "n_test_pairs_available": len(rest),
        "tuning_draw_checked_against": args.ref_npz,
        "test_jsonl_sha256": hashlib.sha256(open(os.path.join(d, "test.jsonl"), "rb").read()).hexdigest(),
        "category_counts": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "pair_ids": chosen,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote %d pairs to %s; categories %s" % (len(chosen), args.out, rec["category_counts"]))


if __name__ == "__main__":
    main()

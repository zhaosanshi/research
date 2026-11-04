#!/usr/bin/env python3
"""Download, adapt, sample and split datasets into data/processed/<name>/{all,calib,dev,test}.jsonl.

Periphery script: everything dataset-specific lives in configs/datasets/*.json and data_adapters/.

Usage:
  python3 scripts/prepare_data.py --datasets got counterfact fever truthfulqa simpleqa halueval_qa gsm8k
  python3 scripts/prepare_data.py --datasets mmlu          # needs the `datasets` library (GPU box / hf-mirror)
Options:
  --raw-dir  (default <ROOT>/data/raw)   --out-dir (default <ROOT>/data/processed)
"""
import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

from core.config import load_dataset_config  # noqa: E402
from core.io_utils import atomic_json_dump, now_iso  # noqa: E402
from data_adapters import get_adapter  # noqa: E402
from data_adapters.base import balanced_sample_tf, sha1_file, split_records, stratified_sample, write_jsonl  # noqa: E402

ALL = ["got", "counterfact", "fever", "truthfulqa", "simpleqa", "halueval_qa", "mmlu", "gsm8k"]


def prepare(name, raw_dir, out_dir):
    cfg = load_dataset_config(name)
    print("== %s ==" % name, flush=True)
    recs = get_adapter(cfg["adapter"])(cfg, raw_dir)
    n_raw = len(recs)
    samp = cfg.get("sampling", {})
    seed = cfg.get("split", {}).get("seed", 42)
    n_total = samp.get("n_total")
    if cfg["task"] == "tf" and samp.get("balanced", False):
        recs = balanced_sample_tf(recs, n_total or len(recs), seed, by_pair=samp.get("by_pair", True))
    elif samp.get("stratify_by"):
        key = samp["stratify_by"]
        recs = stratified_sample(recs, n_total or len(recs), lambda r: str(r["meta"].get(key, "")), seed)
    elif n_total and n_total < len(recs):
        import random
        rng = random.Random(seed)
        recs = sorted(recs, key=lambda r: r["id"])
        rng.shuffle(recs)
        recs = recs[:n_total]
    sp = cfg.get("split", {})
    splits = split_records(recs, sp.get("calib_units", 0), sp.get("dev_units", 0), seed, unit=sp.get("unit", "record"))
    dest = os.path.join(out_dir, name)
    write_jsonl(recs, os.path.join(dest, "all.jsonl"))
    for k, v in splits.items():
        write_jsonl(v, os.path.join(dest, k + ".jsonl"))
    raw_files = {}
    raw_sub = os.path.join(raw_dir, cfg["adapter"] if cfg["adapter"] != "halueval_qa" else "halueval")
    if os.path.isdir(raw_sub):
        for fn in sorted(os.listdir(raw_sub)):
            p = os.path.join(raw_sub, fn)
            if os.path.isfile(p) and not fn.endswith(".part"):
                raw_files[fn] = {"sha1": sha1_file(p), "bytes": os.path.getsize(p)}
    manifest = {
        "dataset": name, "task": cfg["task"], "created": now_iso(), "n_raw_records": n_raw, "n_selected": len(recs),
        "splits": {k: len(v) for k, v in splits.items()},
        "label_balance": dict(Counter(str(r["label"]) for r in recs)) if cfg["task"] == "tf" else None,
        "per_subset": dict(Counter(r["meta"].get("subset", "") for r in recs)) if name == "got" else None,
        "sampling": samp, "split": sp, "source": cfg.get("source"), "raw_files": raw_files,
    }
    atomic_json_dump(manifest, os.path.join(dest, "manifest.json"))
    print("  raw=%d selected=%d splits=%s balance=%s" % (n_raw, len(recs), manifest["splits"], manifest["label_balance"]), flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "processed"))
    args = ap.parse_args()
    failures = []
    for name in args.datasets:
        try:
            prepare(name, args.raw_dir, args.out_dir)
        except Exception as e:  # noqa: BLE001 - report every failure loudly, keep going
            failures.append((name, repr(e)))
            print("  FAILED %s: %r" % (name, e), flush=True)
    if failures:
        print("\nFAILED datasets:", failures)
        sys.exit(1)
    print("\nall requested datasets prepared")


if __name__ == "__main__":
    main()

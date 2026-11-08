#!/usr/bin/env python3
"""Merge per-run API reply caches into one shared cache (TD_JUDGE2_CACHE) and record request keys that received
DIFFERENT replies (a judge that is not deterministic). Cells whose request key is listed in conflicts.json are routed
to the human by `adjudicate` (review_reason inconsistent_replies:<judge>) instead of being decided by whichever reply
happened to be read last (GPT round 23: identical inputs must not get different machine verdicts, and no reply may
be picked by convenience).

usage: judge2_cache_merge.py SHARED_CACHE_DIR SRC_DIR [SRC_DIR ...]
Every judge2_cache_<model>.jsonl under the sources (and the shared dir itself) is read; rows are appended to the shared
file in source order with exact duplicates (same key, same answer) dropped; conflicts.json lists, per cache file, the
keys with more than one distinct answer and the answers seen. Idempotent.
"""
import argparse
import glob
import json
import os
import sys
from collections import OrderedDict

CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE_ROOT)
from core.io_utils import read_jsonl, atomic_json_dump  # noqa: E402


def merge(shared_dir, src_dirs):
    os.makedirs(shared_dir, exist_ok=True)
    files = {}
    for d in [shared_dir] + list(src_dirs):
        for p in sorted(glob.glob(os.path.join(d, "judge2_cache_*.jsonl"))):
            files.setdefault(os.path.basename(p), []).append(p)
    report = {"keys": {}, "detail": {}, "files": {}}
    for name, paths in files.items():
        rows, seen = [], set()
        answers = OrderedDict()
        for p in paths:
            for r in read_jsonl(p):
                sig = (r["key"], r["answer"])
                answers.setdefault(r["key"], [])
                if r["answer"] not in answers[r["key"]]:
                    answers[r["key"]].append(r["answer"])
                if sig in seen:
                    continue
                seen.add(sig)
                rows.append(r)
        out = os.path.join(shared_dir, name)
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
        conflicts = {k: v for k, v in answers.items() if len(v) > 1}
        report["keys"][name] = sorted(conflicts)
        report["detail"][name] = conflicts
        report["files"][name] = {"sources": paths, "rows": len(rows), "unique_keys": len(answers), "conflicting_keys": len(conflicts)}
    atomic_json_dump(report, os.path.join(shared_dir, "conflicts.json"))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shared_dir"); ap.add_argument("src_dirs", nargs="*")
    a = ap.parse_args()
    rep = merge(a.shared_dir, a.src_dirs)
    print(json.dumps({"files": rep["files"], "conflicting_keys": {k: len(v) for k, v in rep["keys"].items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Read-only choice of DeepSeek Pro reply caches to seed a work dir with (Claude, 2026-11-14 02:xx; red team of chain p).
judge2_cache_merge.py stops a Pro pass when the merged rows give one request key two different answers (core
judge_cache_store.record_conflicts_unlocked: same row['key'], different row['answer']); a malformed reply and its retry can
already sit under one key inside a single cache. This picks, among the candidate cache dirs (in the given order of
preference), the largest set whose union has no such key: all of them, else the largest clean subset, preferring earlier
candidates; nothing when every candidate is itself in conflict. It reads only judge2_cache_<MODEL>.jsonl in each dir.
usage: ff_cache_pick.py [--model deepseek-v4-pro] DIR [DIR ...]   -> prints per-dir counts and a last line "use: DIR ..."
"""
import argparse
import itertools
import json
import os
import sys

CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE_ROOT)
from core.judge_cache_store import read_cache_rows  # noqa: E402


def answers(rows):
    out = {}
    for r in rows:
        vals = out.setdefault(r["key"], [])
        if r["answer"] not in vals:
            vals.append(r["answer"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("dirs", nargs="+")
    a = p.parse_args()
    rows = {}
    for d in a.dirs:
        f = os.path.join(d, "judge2_cache_%s.jsonl" % a.model)
        if not os.path.isfile(f):
            raise SystemExit("no cache file %s" % f)
        rows[d] = read_cache_rows(f)
        own = answers(rows[d])
        print("%s: %d rows, %d keys, %d conflicting inside" % (d, len(rows[d]), len(own), sum(len(v) > 1 for v in own.values())))
    for k in range(len(a.dirs), 0, -1):
        for subset in itertools.combinations(a.dirs, k):
            union = answers([r for d in subset for r in rows[d]])
            if not any(len(v) > 1 for v in union.values()):
                print("use: %s" % " ".join(subset))
                return
    print("use: ")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Human review of judge abstentions (赵磊's rule, 2026-11-08).

  list CACHE_DIR                      pending items: queued by the judge (D / no verdict) and not yet
                                      superseded by a human row in the semantic cache; JSON lines
  set  CACHE_DIR KEY LETTER --by WHO  record a verdict (A / B / C) as a superseding cache row

Workflow: the judge (core/judge.py) appends to CACHE_DIR/judge_human_review_queue.jsonl whenever
it abstains; Claude renders the pending items in Chinese for 赵磊; his A / B / C goes in with `set`;
the next grading run reads the human row (last row per key wins) and the item is no longer pending.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.judge import HUMAN_REVIEW_QUEUE, cache_filename, record_human_verdict  # noqa: E402
from core.io_utils import read_jsonl  # noqa: E402


def _latest_answers(cache_dir, model):
    path = os.path.join(cache_dir, cache_filename(model))
    latest = {}
    if os.path.exists(path):
        for row in read_jsonl(path):
            latest[row["key"]] = row
    return latest


def pending(cache_dir):
    queue_path = os.path.join(cache_dir, HUMAN_REVIEW_QUEUE)
    if not os.path.exists(queue_path):
        return []
    rows = list(read_jsonl(queue_path))
    latest = {}
    out = []
    for q in rows:
        if q["model"] not in latest:
            latest[q["model"]] = _latest_answers(cache_dir, q["model"])
        cur = latest[q["model"]].get(q["key"])
        if cur is not None and cur.get("source") == "human":
            continue
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list"); p_list.add_argument("cache_dir")
    p_set = sub.add_parser("set"); p_set.add_argument("cache_dir"); p_set.add_argument("key")
    p_set.add_argument("letter", choices=["A", "B", "C"]); p_set.add_argument("--by", required=True)
    p_set.add_argument("--note", default=None)
    a = ap.parse_args()
    if a.cmd == "list":
        items = pending(a.cache_dir)
        for q in items:
            print(json.dumps(q, ensure_ascii=False))
        print("# pending: %d" % len(items), file=sys.stderr)
        return
    # set: the queue row supplies the model, so the verdict lands in the right cache file
    queue_path = os.path.join(a.cache_dir, HUMAN_REVIEW_QUEUE)
    match = [q for q in read_jsonl(queue_path) if q["key"] == a.key] if os.path.exists(queue_path) else []
    if not match:
        raise SystemExit("key %s is not in %s — only queued items take a human verdict" % (a.key, queue_path))
    row = record_human_verdict(os.path.join(a.cache_dir, cache_filename(match[0]["model"])),
                               a.key, match[0]["model"], a.letter, a.by, a.note)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

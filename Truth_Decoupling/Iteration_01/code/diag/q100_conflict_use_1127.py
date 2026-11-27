#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Opus 5.5, 2026-11-27), work queue discussion/work_queue/2026-11-26_queue_p.md
# section 5, the 100-question passes. Read-only, no API call. Not reviewed by GPT.
# The full-run launchers stop when the merged Pro reply caches answer ANY request key in two ways. On 2026-11-27 08:30 the
# two TruthfulQA caches the passes are seeded with (judge_test_posprompt<tag>_test and judge_test_posprompt2<tag>_test,
# judged in parallel on 2026-11-21 with separate caches) disagree on 1 key (Gemma-3-12B) and 2 keys (Llama-3.1-8B): the same
# request got T from one call and F from the other. judge_two_axis_run.py single already leaves a cell whose request key is
# in the conflict index without a verdict (no reply is picked), and the launcher stops when a cell has no verdict. This
# check asks the narrower question: does any non-empty cell of THIS work dir use a conflicting key? Keys by the unchanged
# judge_two_axis_run.judge_request_keys (crow_pro_requests.py's way). Exit 1 if one does.
# usage (container, -w <code copy>, PYTHONPATH=<code copy>): python3 -B <path>/q100_conflict_use_1127.py WORK
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "review_checks"))
import judge_two_axis_run as J  # noqa: E402


def main():
    work = sys.argv[1]
    p = os.path.join(work, "judge2", "cache", "conflicts.json")
    conf = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    keys = conf.get("keys", {})
    bad = set(k for v in keys.values() for k in v) if isinstance(keys, dict) else set(keys)
    cfg = J.load_cfg()
    cells = [json.loads(l) for l in open(os.path.join(work, "judge2", "cells.jsonl"), encoding="utf-8")]
    todo = [c for c in cells if not c["gold_is_abstention"] and c["prediction"].strip()]
    hit = []
    for c in todo:
        ks = set(J.judge_request_keys(c, "pro", cfg))
        if ks & bad:
            hit.append((c["pair_id"], c["condition"], sorted(k[:12] for k in ks & bad)))
    print("work %s | conflicting keys in the merged caches %d | non-empty cells %d | cells using a conflicting key %d" % (
        work, len(bad), len(todo), len(hit)))
    for h in hit:
        print("   %s" % (h,))
    sys.exit(1 if hit else 0)


if __name__ == "__main__":
    main()

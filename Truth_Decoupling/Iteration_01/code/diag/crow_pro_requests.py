# Iteration_01 (Claude Opus 5.5, executor F, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_m.md section 1):
# before a DeepSeek Pro batch, count what it would send. Read-only, no API call, numbers only.
# For the cells of a work dir (judge2/cells.jsonl): cells, distinct (question, answer) pairs, cells Pro would grade
# (not an abstention-only gold, answer not empty, no valid Pro row yet), their two request keys each (computed by the
# unchanged judge_two_axis_run.judge_request_keys), and how many keys are not in any Pro reply cache given (the work
# dir's own judge2/cache and each extra cache dir). Expected cost = new requests / 2 x price per cell (off-peak
# 0.00046 CNY measured on 2026-11-19; upper bound 0.0008).
# usage (container, -w <code copy>, PYTHONPATH=<code copy>):
#   python3 -B <path>/crow_pro_requests.py WORK [CACHE_DIR ...]
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "review_checks"))
import judge_two_axis_run as J  # noqa: E402

PRICE, PRICE_HI = 0.00046, 0.0008


def keys_in(d):
    out = set()
    for p in glob.glob(os.path.join(d, "judge2_cache_deepseek-v4-pro.jsonl")):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                out.add(json.loads(l)["key"])
    return out


def main():
    work = sys.argv[1]
    caches = [os.path.join(work, "judge2", "cache")] + sys.argv[2:]
    cfg = J.load_cfg()
    cells = [json.loads(l) for l in open(os.path.join(work, "judge2", "cells.jsonl"), encoding="utf-8")]
    done = set()
    pro = os.path.join(work, "judge2", "pro.jsonl")
    if os.path.exists(pro):
        for l in open(pro, encoding="utf-8"):
            r = json.loads(l)
            if r.get("truth") is not None and r.get("info") is not None:
                done.add(r["cell_id"])
    known = set()
    for d in caches:
        if not os.path.isdir(d):
            raise SystemExit("no cache dir %s" % d)
        known |= keys_in(d)
    pairs = {(c["question"], c["prediction"]) for c in cells}
    todo = [c for c in cells if not c["gold_is_abstention"] and c["prediction"].strip() and c["cell_id"] not in done]
    req = set()
    for c in todo:
        req.update(J.judge_request_keys(c, "pro", cfg))
    new = req - known
    conds = {}
    for c in cells:
        conds[c.get("condition", "?")] = conds.get(c.get("condition", "?"), 0) + 1
    print("work %s" % work)
    print("cells %d in %d conditions; distinct (question, answer) %d; cells to grade %d; their distinct request keys %d" % (
        len(cells), len(conds), len(pairs), len(todo), len(req)))
    print("keys in the caches %d; NEW requests %d (= about %d new cells)" % (len(known), len(new), (len(new) + 1) // 2))
    print("expected cost %.2f CNY (off-peak %.5f / cell), upper %.2f CNY" % (len(new) / 2 * PRICE, PRICE, len(new) / 2 * PRICE_HI))


if __name__ == "__main__":
    main()

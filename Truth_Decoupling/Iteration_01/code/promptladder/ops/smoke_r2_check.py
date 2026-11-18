# Exact eight-question check of the informational smoke (Claude Opus 5.5, 2026-11-18), the manual form of GPT's re-check
# section 3 item 1: per model, the control conf and the three ladder arms each hold exactly 8 rows, no repeats, the same
# question ids in the same order, equal to the first 8 rows of the run's own driver file gen/E_L<L>_conf-w-p2_tune.json;
# the control's text equals the driver's text row by row (the launcher also compares tokens). Read-only.
import json
import os
import sys

R = "/data/sanshizhao/Truth_Decoupling"
PL = R + "/Iteration_01/promptladder"
RUNS = {"llama": ("results_e44_factorial_das/llama3.1-8b_side_E2", 11), "qwen9": ("results_newmodels_ff_20261112/e44/qwen3.5-9b_E2", 15),
        "qwen27": ("results_newmodels_ff_20261112/e44/qwen3.8-27b-dm", 28), "g12": ("Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b", 22),
        "g27": ("Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b", 26)}
ARMS = ("conf", "P1", "P2", "P3")
bad = 0
for key, (run, el) in RUNS.items():
    name = run.split("/")[-1]
    driver = json.load(open("%s/%s/gen/E_L%d_conf-w-p2_tune.json" % (R, run, el), encoding="utf-8"))
    want = [r["pair_id"] for r in driver["rows"][:8]]
    dtext = {r["pair_id"]: r["text"] for r in driver["rows"][:8]}
    for root in ("smoke_r2_20261118", "smoke_r2_20261118_2gpu"):
        out = "%s/%s/%s" % (PL, root, name)
        if not os.path.isdir(out):
            continue
        passed = os.path.exists(out + "/smoke_pass.json")
        msgs = []
        for arm in ARMS:
            f = "%s/gen/t8.%s--E%d-w-p2_tune.json" % (out, arm, el)
            if not os.path.exists(f):
                msgs.append("%s missing" % arm)
                continue
            rows = json.load(open(f, encoding="utf-8"))["rows"]
            ids = [r["pair_id"] for r in rows]
            ok = len(rows) == 8 and len(set(ids)) == 8 and ids == want
            if arm == "conf":
                ok = ok and all(r["text"] == dtext.get(r["pair_id"]) for r in rows)
            empty = sum(1 for r in rows if not str(r.get("text", "")).strip())
            msgs.append("%s %s (%d rows, %d empty)" % (arm, "ok" if ok else "MISMATCH", len(rows), empty))
            bad += not ok
        print("%-7s %-24s smoke_pass=%s | %s" % (key, root, passed, "; ".join(msgs)))
sys.exit(1 if bad else 0)

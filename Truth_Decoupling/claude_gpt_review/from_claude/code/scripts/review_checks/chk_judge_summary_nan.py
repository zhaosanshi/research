#!/usr/bin/env python3
"""Round 16 red-team claim, checked and REFUTED (Claude, 2026-11-05).

usage: chk_judge_summary_nan.py CODE_DIR

Claim: E24's qa_rates() returns precision_answered = NaN for a condition with zero answered items (core/metrics.py:101);
the summary is written to tune_scored.json and copied into tune_scored_items.json (judge_summary); e25_qa_das.freeze()
compares the two copies with `!=`; since nan != nan, two identical summaries would compare unequal and freeze() would
fail with "semantic grades are stale, foreign or use the wrong rubric". Plausible trigger: a shortlisted candidate whose
192 tuning answers are all judged abstain (Gemma-4B mean-difference arms produce garbage in continuous mode).

Fact: CPython's json decoder returns ONE shared float object for every "NaN" literal (json.decoder.NaN via
parse_constant), and container equality short-circuits on identity before calling __eq__. Two independently reloaded
copies therefore compare EQUAL, and freeze() is unaffected. No production change was made; this script pins the
behaviour so a future json implementation change would be noticed. Exit 1 on any failed expectation.
"""
import json
import math
import os
import sys
import tempfile

CODE = sys.argv[1]
sys.path[:0] = [os.path.join(CODE, "scripts"), CODE]
import common  # noqa: F401,E402
from core.metrics import qa_rates  # noqa: E402
import e25_qa_das as pipeline  # noqa: E402

fails = []


def expect(name, ok, detail=None):
    print(("ok   " if ok else "FAIL ") + name, "" if detail is None else detail)
    if not ok:
        fails.append(name)


rates = qa_rates(["abstain"] * 5)
expect("qa_rates_zero_answered_gives_nan_precision", isinstance(rates["precision_answered"], float)
       and math.isnan(rates["precision_answered"]), rates)
summary = {"rates": {"E_x": {"rates": rates, "delta_vs_baseline": {"precision_answered": {"value": float("nan"),
                                                                                           "ci95": [float("nan"), float("nan")]}}}}}
with tempfile.TemporaryDirectory() as td:
    a, b = os.path.join(td, "tune_scored.json"), os.path.join(td, "tune_scored_items.json")
    pipeline.save(a, summary)                               # the pipeline's own writer (atomic_json_dump, NaN allowed)
    pipeline.save(b, {"judge_summary": summary})
    raw_a, raw_b = pipeline.read(a), pipeline.read(b)["judge_summary"]   # the pipeline's own reader (json.loads)
    expect("file_contains_literal_NaN", "NaN" in open(a).read())
    expect("reloaded_nan_is_one_shared_object",
           raw_a["rates"]["E_x"]["rates"]["precision_answered"] is raw_b["rates"]["E_x"]["rates"]["precision_answered"])
    expect("reloaded_copies_compare_EQUAL_so_freeze_is_unaffected", raw_a == raw_b and not (raw_a != raw_b))
    expect("finite_rates_round_trip_exactly", all(raw_a["rates"]["E_x"]["rates"][k] == rates[k]
                                                  for k in ("correct", "wrong", "abstain", "answer_rate")))
    # The abstract claim itself is true for distinct NaN objects; it just does not arise through json.loads.
    expect("distinct_nan_objects_would_differ (why the claim looked plausible)",
           {"x": float("nan")} != {"x": float("nan")})
print("failures:", fails)
sys.exit(1 if fails else 0)

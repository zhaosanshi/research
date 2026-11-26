#!/usr/bin/env python3
# Iteration_01 exploration (executor you, Claude Opus 5.5, 2026-11-27), queue P section 5 segment A: the GPU step of the
# batch-1 self-judge margins of the new FalseQA answers listed by diag/falseqa_gate2_1127.py export-margins (sources
# fqa_gate2_1127s_* for the smoke run, fqa_gate2_1127_* for the full run). diag/falseqa_score.py with only the source
# names changed: selfjudge_two_answers.score1 unchanged (batch size 1, its path identity check first).
# usage (model interpreter, from /workspace/Truth_Decoupling): python -B Iteration_01/diag/falseqa_score_gate2_1127.py score MODEL smoke|full
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import selfjudge_probe as SP          # noqa: E402
import selfjudge_two_answers as TA    # noqa: E402

SRC = {"smoke": ("fqa_gate2_1127s_items", "fqa_gate2_1127s_chk", "fqa_gate2_1127s_chka", "fqa_gate2_1127s_chkb"),
       "full": ("fqa_gate2_1127_items", "fqa_gate2_1127_chk", "fqa_gate2_1127_chka", "fqa_gate2_1127_chkb")}
SP.SOURCES.update(SRC["smoke"] + SRC["full"])

if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "score" or sys.argv[2] not in TA.ORDER or sys.argv[3] not in SRC:
        raise SystemExit("usage: falseqa_score_gate2_1127.py score MODEL smoke|full")
    TA.score1(sys.argv[2], list(SRC[sys.argv[3]]))

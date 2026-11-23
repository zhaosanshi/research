#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, executor E, 2026-11-23), queue L section 11: the GPU step of the batch-1
# self-judge margins of the FalseQA answers that diag/falseqa_read.py export-margins listed (sources fqa0_* for the
# smoke rows, fqa1_* for the full run): selfjudge_two_answers.score1 unchanged (batch size 1, its path identity check
# first), as diag/fill_only2_score.py does for its fo_* sources. Imports only selfjudge_probe and selfjudge_two_answers.
# usage (model interpreter, from /workspace/Truth_Decoupling): python -B Iteration_01/diag/falseqa_score.py score MODEL smoke|full
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import selfjudge_probe as SP          # noqa: E402
import selfjudge_two_answers as TA    # noqa: E402

SRC = {"smoke": ("fqa0_items", "fqa0_chk", "fqa0_chka", "fqa0_chkb"), "full": ("fqa1_items", "fqa1_chk", "fqa1_chka", "fqa1_chkb")}
SP.SOURCES.update(SRC["smoke"] + SRC["full"])

if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "score" or sys.argv[2] not in TA.ORDER or sys.argv[3] not in SRC:
        raise SystemExit("usage: falseqa_score.py score MODEL smoke|full")
    TA.score1(sys.argv[2], list(SRC[sys.argv[3]]))

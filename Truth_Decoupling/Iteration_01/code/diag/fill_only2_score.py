#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, executor A, 2026-11-23), queue L section 9: the GPU step of the self-judge
# margins that diag/fill_only2.py export listed (sources fo_items / fo_chk / fo_chka / fo_chkb, the source names of the
# header of diag/fill_only.py): selfjudge_two_answers.score1 unchanged (batch size 1, its path identity check first), as
# selfjudge_stack_pick.py score does for its sp_* sources. Imports only selfjudge_probe and selfjudge_two_answers.
# usage (model interpreter, from /workspace/Truth_Decoupling): python -B Iteration_01/diag/fill_only2_score.py score MODEL
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import selfjudge_probe as SP          # noqa: E402
import selfjudge_two_answers as TA    # noqa: E402

FO_SRC = ("fo_items", "fo_chk", "fo_chka", "fo_chkb")
SP.SOURCES.update(FO_SRC)

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "score" or sys.argv[2] not in TA.ORDER:
        raise SystemExit("usage: fill_only2_score.py score MODEL")
    TA.score1(sys.argv[2], list(FO_SRC))

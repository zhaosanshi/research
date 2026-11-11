#!/usr/bin/env bash
# Extra generations for shortlisted layers of a declared SIDE branch, same reviewed entry point and code tree as the
# first round (the side branch's own run dir and its own truth-level flags).
#   BRANCH=M|E2|MD  LAYERS="11 12"  ARMS="x.rand.m1 x_F.w.p1"  GPU=n
set -euo pipefail
: "${BRANCH:?}" "${LAYERS:?}" "${ARMS:?}" "${GPU:?}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16                # the server is shared: do not spawn hundreds of CPU threads
ROOT=/workspace/Truth_Decoupling
MAIN=$ROOT/results_e44_factorial_das/llama3.1-8b
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
case "$BRANCH" in
  M)  RUN=${MAIN}_side_M;      FACTOR=(--e-off M) ;;
  E2) RUN=${MAIN}_side_E2;     FACTOR=(--e-on E2) ;;
  MD) RUN=${MAIN}_side_initmd; FACTOR=() ;;
  *) echo "unknown BRANCH $BRANCH" >&2; exit 2 ;;
esac
cd $ROOT/review_e44_claude/claude_gpt_review/from_claude/code
echo "=== $(date '+%F %T') [side $BRANCH gpu$GPU] extra generation: layers $LAYERS arms $ARMS ==="
python scripts/e44_factorial_das.py generate --source-run "$SRC" --run-dir "$RUN" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 \
  --family E "${FACTOR[@]}" --layers $LAYERS --split tune --arms $ARMS
echo "=== $(date '+%F %T') [side $BRANCH gpu$GPU] extra generation done ==="

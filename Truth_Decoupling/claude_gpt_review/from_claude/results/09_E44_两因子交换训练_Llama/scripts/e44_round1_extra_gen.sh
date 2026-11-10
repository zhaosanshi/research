#!/usr/bin/env bash
# Extra generations for shortlisted layers of the main hypothesis (plan 28, sections 3.2 and 3.3), same reviewed entry point.
#   FAMILY=E|C  LAYERS="3 10"  ARMS="x.rand.m1 x_F.w.p1"  GPU=n   (E: second sign of the random control + the format control; later the fixed-shift arms on conf / caut)
set -euo pipefail
: "${FAMILY:?}" "${LAYERS:?}" "${ARMS:?}" "${GPU:?}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1
ROOT=/workspace/Truth_Decoupling
RUN=$ROOT/results_e44_factorial_das/llama3.1-8b
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
cd $ROOT/review_e44_claude/claude_gpt_review/from_claude/code
echo "=== $(date '+%F %T') extra generation: $FAMILY layers $LAYERS arms $ARMS ==="
python scripts/e44_factorial_das.py generate --source-run "$SRC" --run-dir "$RUN" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 \
  --family "$FAMILY" --layers $LAYERS --split tune --arms $ARMS
echo "=== $(date '+%F %T') extra generation done ==="

#!/usr/bin/env bash
# Real-model acceptance of the two-write engine of scripts/e44_eval_extra.py (GPT round 34, A1), version
# 4438a282 as reviewed and repaired by GPT (04ee956). Runs inside the container. No training, no judge calls.
#   K=n GPU=k [ARM=...]   selfcheck on the first n batches of the tuning split (K=0: the whole split)
# The selfcheck itself checks: zero/zero = no write; (first, zero) = single first write; (zero, second) = single
# second write (token for token); then one generation with both writes non-zero: same positions and decode steps in
# both hooks, stored norms equal the single-write runs, recorded amounts equal the planned ones.
set -euo pipefail
: "${K:?}" "${GPU:?}"
ARM=${ARM:-x:E10.w.p2+C31.w.m1}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
ROOT=/workspace/Truth_Decoupling
MAIN=$ROOT/results_e44_factorial_das/llama3.1-8b
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
CODE=$ROOT/review_e44_claude/claude_gpt_review/from_claude/code
cd "$CODE"
sha256sum scripts/e44_eval_extra.py configs/prompts/e44_eval_extra_v1.json
FB=()
[ "$K" -gt 0 ] && FB=(--first-batches "$K")
echo "=== $(date '+%F %T') selfcheck $ARM, tune, first batches $K, gpu $GPU ==="
python scripts/e44_eval_extra.py selfcheck --source-run "$SRC" --run-dir "$MAIN" --arms "$ARM" --split tune \
  --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 ${FB[@]+"${FB[@]}"}
echo "=== $(date '+%F %T') done ==="

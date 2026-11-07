#!/bin/bash
# Claude, 2026-11-07: execution-level reproduction of two GPT round-19 diagnostics on Qwen3-8B with the GitHub code
# (63e3674 snapshot in review_r19_claude/), written to a fresh directory; GPT's own outputs are only read afterwards.
#   GPU0  e27_trace_corruption.py   tqa:138 / 150 / 412, release DAS-init adapter (same command as unicode/review.md)
#   GPU1  e27_frozen_initial_controls.py generate, first three tune questions, GPT's early random / DAS adapters
# No judge phase, no training. usage: bash r19_reproduce.sh
set -u
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/review_r19_claude/claude_gpt_review/from_claude/code
OUT=$ROOT/results_r19_claude_20261107
mkdir -p $HOST/results_r19_claude_20261107/logs
run() {  # GPU LOGNAME script args...
  local gpu=$1 log=$2; shift 2
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_DATA_DIR=$ROOT/code/data/processed -e TD_MODEL_ROOT=$ROOT/models \
    -e TD_RESULTS_DIR=$OUT -e CUDA_VISIBLE_DEVICES=$gpu -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -w $CODE truth_dec python3 "$@" > $HOST/results_r19_claude_20261107/logs/$log.log 2>&1
  echo $? > $HOST/results_r19_claude_20261107/logs/$log.exit
}
run 0 qwen_trace scripts/review_checks/e27_trace_corruption.py \
  --source-run $ROOT/results_qa_das_v1u/qwen3-8b/full \
  --adapter-run $ROOT/results_e27_release_20261106/qwen3-8b/ce_alpha3_ep3/das_s42 \
  --split tune --pair-ids tqa:138 tqa:150 tqa:412 --out $OUT/qwen_trace.json &
run 1 qwen_random_pilot3 scripts/review_checks/e27_frozen_initial_controls.py generate \
  --source-run $ROOT/results_qa_das_v1u/qwen3-8b/full \
  --random-run $ROOT/results_e27_search_codex_20261106/qwen3-8b/ce_alpha3_ep3/random \
  --das-run $ROOT/results_e27_search_codex_20261106/qwen3-8b/ce_alpha3_ep3/das \
  --run-dir $OUT/random_controls/qwen3-8b/pilot3 --split tune --limit 3 &
wait
date > $HOST/results_r19_claude_20261107/logs/ALL.DONE

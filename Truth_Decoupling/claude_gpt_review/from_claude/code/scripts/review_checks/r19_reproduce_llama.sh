#!/bin/bash
# Claude, 2026-11-07: reproduce GPT round 19's two Llama-3.1-8B GPU diagnostics with the GitHub code (63e3674 snapshot).
#   GPU0  e27_trace_corruption.py, six tune questions, release DAS-init adapter (models/real_trace_review.md)
#   GPU1  e27_pure_das_layer_controls.py generate, all 192 tune questions, original DAS at L10 / L29 (models/pure_das_layer_review.md)
# Generation only: no judge phase, no training. usage: bash r19_reproduce_llama.sh
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
run 0 llama_trace scripts/review_checks/e27_trace_corruption.py \
  --source-run $ROOT/results_qa_das_v1u/llama3.1-8b/full \
  --adapter-run $ROOT/results_e27_release_20261106/llama3.1-8b/ce_alpha3_ep3/das_s42 \
  --split tune --pair-ids tqa:138 tqa:150 tqa:412 tqa:136 tqa:647 tqa:685 --out $OUT/llama_trace.json &
run 1 llama_pure_das_tune192 scripts/review_checks/e27_pure_das_layer_controls.py generate \
  --source-run $ROOT/results_qa_das_v1u/llama3.1-8b/full \
  --run-dir $OUT/llama_pure_das_tune192 --all-questions &
wait
date > $HOST/results_r19_claude_20261107/logs/LLAMA.DONE

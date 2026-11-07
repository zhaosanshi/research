#!/bin/bash
# Claude, 2026-11-07: reproduce the generation phase of three GPT round-19 192-question tune runs with the GitHub code
# (63e3674 snapshot in review_r19_claude/), into a fresh directory; outputs are compared with GPT's afterwards.
#   GPU0  frozen-random five-arm control, Qwen3-8B   (GPT's early random / DAS adapters, as in round 19)
#   GPU1  frozen-random five-arm control, Llama-3.1-8B
#   GPU2  deployment-dose diagnostic, Qwen3-8B        (release DAS-init adapter, alpha 1 / 2 / 3)
# Generation only: no judge phase, no training. usage: bash r19_reproduce_full.sh
set -u
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/review_r19_claude/claude_gpt_review/from_claude/code
OUT=$ROOT/results_r19_claude_20261107
EARLY=$ROOT/results_e27_search_codex_20261106
mkdir -p $HOST/results_r19_claude_20261107/logs
run() {  # GPU LOGNAME script args...
  local gpu=$1 log=$2; shift 2
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_DATA_DIR=$ROOT/code/data/processed -e TD_MODEL_ROOT=$ROOT/models \
    -e TD_RESULTS_DIR=$OUT -e CUDA_VISIBLE_DEVICES=$gpu -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -w $CODE truth_dec python3 "$@" > $HOST/results_r19_claude_20261107/logs/$log.log 2>&1
  echo $? > $HOST/results_r19_claude_20261107/logs/$log.exit
}
for M in qwen3-8b llama3.1-8b; do
  GPU=0; [ $M = llama3.1-8b ] && GPU=1
  run $GPU random_full192_$M scripts/review_checks/e27_frozen_initial_controls.py generate \
    --source-run $ROOT/results_qa_das_v1u/$M/full \
    --random-run $EARLY/$M/ce_alpha3_ep3/random --das-run $EARLY/$M/ce_alpha3_ep3/das \
    --run-dir $OUT/random_controls/$M/full192 --split tune --all-questions &
done
run 2 dose_full192_qwen3-8b scripts/review_checks/e27_generation_dose.py generate \
  --source-run $ROOT/results_qa_das_v1u/qwen3-8b/full \
  --adapter-run $ROOT/results_e27_release_20261106/qwen3-8b/ce_alpha3_ep3/das_s42 \
  --run-dir $OUT/generation_dose/qwen3-8b/full192 --all-tune &
wait
date > $HOST/results_r19_claude_20261107/logs/FULL.DONE

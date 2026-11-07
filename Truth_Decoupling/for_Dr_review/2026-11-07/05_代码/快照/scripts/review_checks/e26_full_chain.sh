#!/usr/bin/env bash
# E26 matched actual-coverage chain for ONE model (Claude, 2026-11-06), following the GPT round-18 runbook
# (from_gpt/main_experiment_runbook_20261106.md sections 4 and 6). Runs on the host; every stage runs in the container.
#
# usage: e26_full_chain.sh MODEL GPU TAG [ADAPT_ROOT]
#   CODE_PATH / RESULTS_PATH are required container paths; no stale code/run directory defaults.
#   TAG        run name under RESULTS_PATH/<MODEL>/ ; a TAG ending in "_pilot3" adds --pilot 3 to prepare
#   ADAPT_ROOT optional E27 adapter root holding das/ md/ random/ (all three always passed together)
# Stages in order: prepare, collect-tune, judge-tune, freeze, collect-test, judge-test, report. Any failure stops the chain
# (judge stages retry up to 3 times, because the script itself saves partial grades and asks to be rerun on API errors).
# Each run has its own judge cache directory, so parallel runs never write the same cache file.
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
  echo "usage: e26_full_chain.sh MODEL GPU TAG [ADAPT_ROOT]" >&2; exit 2
fi
MODEL=$1; GPU=$2; TAG=$3; ADAPT=${4:-}
[[ $MODEL == llama3.1-8b || $MODEL == qwen3-8b ]] || { echo "unsupported model" >&2; exit 2; }
[[ $GPU =~ ^[0-9]+$ && $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid GPU or run tag" >&2; exit 2; }
ROOT=${TD_PROJECT_ROOT:-/workspace/Truth_Decoupling}
HOST=${TD_HOST_PROJECT_ROOT:-/data/sanshizhao/Truth_Decoupling}
: "${CODE_PATH:?Set CODE_PATH to the reviewed code directory inside the container}"
: "${RESULTS_PATH:?Set RESULTS_PATH to a NEW result directory inside the container}"
CODE=$CODE_PATH
RES=$RESULTS_PATH
case "$RES" in "$ROOT"/*) ;; *) echo "RESULTS_PATH must be under TD_PROJECT_ROOT" >&2; exit 2 ;; esac
HOSTRES=$HOST/${RES#"$ROOT"/}
CONTAINER=${TD_CONTAINER:-truth_dec}
CONFIG_NAME=${E26_CONFIG:-E26_matched_coverage}
RUN=$RES/$MODEL/$TAG
CACHE=$RES/$MODEL/${TAG}_judge_cache
mkdir -p "$HOSTRES/$MODEL"
HOSTLOG=$HOSTRES/$MODEL/$TAG.log
QA_SOURCE=$ROOT/results_qa_das_v1u/$MODEL/full
TF_DAS_DIR=$ROOT/results_das_tfce/E1/$MODEL
TF_DAS_SWEEP=$ROOT/results_das_tfce/E23/$MODEL/knob_sweep_das_tfce_full.json
TF_MD_SWEEP=$ROOT/results_das_tf/E23/$MODEL/knob_sweep_meandiff_full.json

run() {
  docker exec -u 1000:1001 -e HF_HOME="$ROOT/hf_cache" -e TD_DATA_DIR="$ROOT/code/data/processed" -e TD_MODEL_ROOT="$ROOT/models" \
    -e CUDA_VISIBLE_DEVICES="$GPU" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e DEEPSEEK_API_KEY_FILE="$ROOT/logs/.deepseek_key" -w "$CODE" "$CONTAINER" \
    python3 scripts/e26_matched_coverage.py "$@" >> "$HOSTLOG" 2>&1
}
step() {
  local name=$1; shift
  echo "== $(date '+%F %T') $name $MODEL $TAG (GPU $GPU)" >> "$HOSTLOG"
  run "$@" || { echo "== $(date '+%F %T') STEP FAILED: $name $MODEL $TAG" >> "$HOSTLOG"; exit 1; }
}
judge_step() {
  local name=$1; shift
  for attempt in 1 2 3; do
    echo "== $(date '+%F %T') $name $MODEL $TAG (GPU $GPU) attempt $attempt" >> "$HOSTLOG"
    if run "$@"; then return 0; fi
    echo "== $(date '+%F %T') $name attempt $attempt did not finish; retrying in 60 s" >> "$HOSTLOG"; sleep 60
  done
  echo "== $(date '+%F %T') STEP FAILED: $name $MODEL $TAG after 3 attempts" >> "$HOSTLOG"; exit 1
}

PREP=(prepare --config "$CONFIG_NAME" --model "$MODEL" --run-dir "$RUN" --qa-run "$QA_SOURCE" --tf-das-dir "$TF_DAS_DIR"
      --tf-das-sweep "$TF_DAS_SWEEP" --tf-md-sweep "$TF_MD_SWEEP")
case "$TAG" in *_pilot3) PREP+=(--pilot 3) ;; esac
[ -n "$ADAPT" ] && PREP+=(--qa-adapters "$ADAPT/das" "$ADAPT/md" "$ADAPT/random")

echo "== $(date '+%F %T') E26 CHAIN START $MODEL $TAG GPU $GPU code=$CODE adapters=${ADAPT:-none}" >> "$HOSTLOG"
step prepare "${PREP[@]}"
step collect-tune collect-tune --model "$MODEL" --run-dir "$RUN"
judge_step judge-tune judge-tune --model "$MODEL" --run-dir "$RUN" --judge-exp E4_qa --cache-dir "$CACHE"
step freeze freeze --model "$MODEL" --run-dir "$RUN"
step collect-test collect-test --model "$MODEL" --run-dir "$RUN"
judge_step judge-test judge-test --model "$MODEL" --run-dir "$RUN" --judge-exp E4_qa --cache-dir "$CACHE"
step report report --model "$MODEL" --run-dir "$RUN"
echo "== $(date '+%F %T') E26 CHAIN DONE $MODEL $TAG" >> "$HOSTLOG"

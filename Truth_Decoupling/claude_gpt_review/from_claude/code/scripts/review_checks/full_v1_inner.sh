#!/usr/bin/env bash
# Runs INSIDE truth_dec: QA-DAS v1 full run for one model, one explicit phase after another, stopping after STOP_AFTER
# (default: freeze, i.e. the whole tuning chain; test generation/judging is a separate later launch).
# usage: full_v1_inner.sh MODEL
# env: TD_RESULTS_DIR TD_DATA_DIR TD_MODEL_ROOT CUDA_VISIBLE_DEVICES DEEPSEEK_API_KEY_FILE STOP_AFTER
set -u
MODEL=$1
ROOT=/workspace/Truth_Decoupling
RES=${TD_RESULTS_DIR:-$ROOT/results_qa_das_v1}
RUN=$RES/$MODEL/full_v1
STOP=${STOP_AFTER:-freeze}
mkdir -p "$RES/$MODEL"
cd "$ROOT/review_e25/claude_gpt_review/from_claude/code" || exit 1
phase() {   # phase NAME [extra args]
  local name=$1; shift
  echo "== $(date '+%F %T') $name $MODEL (GPU ${CUDA_VISIBLE_DEVICES:-?})"
  python3 scripts/e25_qa_das.py "$name" --model "$MODEL" --run-dir "$RUN" "$@" || { echo "== $(date '+%F %T') PHASE FAILED: $name $MODEL"; exit 1; }
  if [ "$name" = "$STOP" ]; then echo "== $(date '+%F %T') STOPPED AFTER $name $MODEL (as requested)"; exit 0; fi
}
if [ -f "$RUN/bank.json" ]; then
  echo "== $(date '+%F %T') estimate already exists for $MODEL, resuming later phases"
  [ "$STOP" = estimate ] && exit 0
else
  phase estimate --all-layers
fi
phase train
phase tune-e
phase generate-tune
phase judge-tune
phase freeze
echo "== $(date '+%F %T') FULL V1 TUNE DONE $MODEL"

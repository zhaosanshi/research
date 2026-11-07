#!/usr/bin/env bash
# E27 generation-adapter rerun with the RELEASE code (Claude, 2026-11-06).
# Historical reason: old adapters did not record training-code hashes. Round 19 subsequently verified that Qwen weights,
# loss traces and generated texts reproduce exactly; differing judge labels are not a generation-code failure.
# These are supervised generation adapters, including the random-initialized trained controls, not pure DAS.
# The historical rerun used commit 7987310 and the
# configuration GPT froze from tuning BEFORE its test: layer from the v1u frozen DAS choice (L34 for Qwen), alpha 3,
# 3 epochs, ranking weight 0, no EOS. Random initialisation gets three seeds (the delivery had one). No selection is made
# among the runs below; all are reported.
#
# usage: e27_release_rerun.sh MODEL GPU
# Set TD_REVIEW_CODE to the reviewed code directory and TD_E27_RESULT_TAG to a
# fresh result tag. Runs serially with one model-level cache and no concurrent writers.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: e27_release_rerun.sh MODEL GPU' >&2; exit 2; }
MODEL=$1; GPU=$2
[[ $MODEL =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && $GPU =~ ^[0-9]+$ ]] || { echo 'invalid model or GPU' >&2; exit 2; }
ROOT=${TD_PROJECT_ROOT:-/workspace/Truth_Decoupling}
HOST=${TD_HOST_PROJECT_ROOT:-/data/sanshizhao/Truth_Decoupling}
CODE=${TD_REVIEW_CODE:?Set TD_REVIEW_CODE to the reviewed code directory; never silently use historical code}
SRC=$ROOT/results_qa_das_v1u/$MODEL/full
# New executions must preserve the historical release artifacts. Explicitly set
# a fresh result tag to launch this supervised-adapter control again.
TAG=${TD_E27_RESULT_TAG:?Set TD_E27_RESULT_TAG to a new result directory name}
[[ $TAG =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid result tag" >&2; exit 1; }
OUT=$ROOT/$TAG/$MODEL/ce_alpha3_ep3
mkdir -p "$HOST/$TAG/$MODEL"
HOSTLOG=$HOST/$TAG/$MODEL/chain.log

run() {
  docker exec -u 1000:1001 -e "TD_DATA_DIR=$ROOT/code/data/processed" -e "TD_MODEL_ROOT=$ROOT/models" \
    -e "TD_RESULTS_DIR=$ROOT/$TAG/$MODEL" -e "HF_HOME=$ROOT/hf_cache" \
    -e "CUDA_VISIBLE_DEVICES=$GPU" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e "DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key" -w "$CODE" truth_dec \
    python3 scripts/e27_qa_generation.py "$@" >> "$HOSTLOG" 2>&1
}
step() {
  local name=$1; shift
  echo "== $(date '+%F %T') $name $MODEL (GPU $GPU)" >> "$HOSTLOG"
  run "$@" || { echo "== $(date '+%F %T') STEP FAILED: $name $MODEL" >> "$HOSTLOG"; exit 1; }
}
judge_step() {
  local name=$1; shift
  for attempt in 1 2 3; do
    echo "== $(date '+%F %T') $name $MODEL (GPU $GPU) attempt $attempt" >> "$HOSTLOG"
    if run "$@"; then return 0; fi
    echo "== $(date '+%F %T') $name attempt $attempt did not finish; retrying in 60 s" >> "$HOSTLOG"; sleep 60
  done
  echo "== $(date '+%F %T') STEP FAILED: $name $MODEL after 3 attempts" >> "$HOSTLOG"; exit 1
}

echo "== $(date '+%F %T') E27 RELEASE RERUN START $MODEL GPU $GPU code=$CODE" >> "$HOSTLOG"
for spec in "das 42" "md 42" "random 42" "random 43" "random 44"; do
  set -- $spec; INIT=$1; SEED=$2; RUN=$OUT/${INIT}_s$SEED
  COMMON=(--source-run "$SRC" --run-dir "$RUN")
  step "train $INIT s$SEED" train "${COMMON[@]}" --init "$INIT" --alpha 3 --epochs 3 --ranking-weight 0 --seed "$SEED"
  step "check $INIT s$SEED" check "${COMMON[@]}"
  step "generate-tune $INIT s$SEED" generate-tune "${COMMON[@]}"
  judge_step "judge-tune $INIT s$SEED" judge-tune "${COMMON[@]}" --judge-exp E4_qa
  step "freeze $INIT s$SEED" freeze "${COMMON[@]}" --primary-quality-rule R0
  step "generate-test $INIT s$SEED" generate-test "${COMMON[@]}"
  judge_step "judge-test $INIT s$SEED" judge-test "${COMMON[@]}" --judge-exp E4_qa
done
echo "== $(date '+%F %T') E27 RELEASE RERUN DONE $MODEL" >> "$HOSTLOG"

#!/usr/bin/env bash
# E27 generation-adapter rerun with the RELEASE code (Claude, 2026-11-06).
# Why: the round-18 Qwen tuning/test adapters were trained by an earlier, unsaved code version (their adapter.json lacks the
# include_eos / stop_token_id fields the release code always writes, and no output records a code hash), so the reported
# 140 -> 189 cannot be reproduced from GitHub as delivered. This rerun uses the GitHub code (commit 7987310) and the
# configuration GPT froze from tuning BEFORE its test: layer from the v1u frozen DAS choice (L34 for Qwen), alpha 3,
# 3 epochs, ranking weight 0, no EOS. Random initialisation gets three seeds (the delivery had one). No selection is made
# among the runs below; all are reported.
#
# usage: e27_release_rerun.sh MODEL GPU
# Runs sequentially (one judge cache per run dir, no concurrent writers). Any failure stops the chain; judge stages retry.
set -euo pipefail
MODEL=$1; GPU=$2
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/review_e26_claude/claude_gpt_review/from_claude/code
SRC=$ROOT/results_qa_das_v1u/$MODEL/full
OUT=$ROOT/results_e27_release_20261106/$MODEL/ce_alpha3_ep3
mkdir -p "$HOST/results_e27_release_20261106/$MODEL"
HOSTLOG=$HOST/results_e27_release_20261106/$MODEL/chain.log

run() {
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_DATA_DIR=$ROOT/code/data/processed -e TD_MODEL_ROOT=$ROOT/models \
    -e TD_RESULTS_DIR=$ROOT/results_e27_release_20261106/$MODEL \
    -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec \
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
  step "generate-test $INIT s$SEED" generate-test "${COMMON[@]}"
  judge_step "judge-test $INIT s$SEED" judge-test "${COMMON[@]}" --judge-exp E4_qa
done
echo "== $(date '+%F %T') E27 RELEASE RERUN DONE $MODEL" >> "$HOSTLOG"

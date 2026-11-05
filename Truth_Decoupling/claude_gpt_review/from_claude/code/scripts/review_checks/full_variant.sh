#!/usr/bin/env bash
# Host side: full_variant.sh MODEL GPU CONFIG RESULTS_SUBDIR [STOP_AFTER]
#   -> runs full_variant_inner.sh in truth_dec; results under /workspace/Truth_Decoupling/RESULTS_SUBDIR/MODEL/full,
#      log appended to /data/sanshizhao/Truth_Decoupling/RESULTS_SUBDIR/MODEL/full.log
set -u
MODEL=$1; GPU=$2; CONFIG=$3; SUB=$4; STOP=${5:-judge-diagnostics}
ROOT=/workspace/Truth_Decoupling
HOSTLOG=/data/sanshizhao/Truth_Decoupling/$SUB/$MODEL/full.log
mkdir -p "$(dirname "$HOSTLOG")"
docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_DATA_DIR=$ROOT/code/data/processed \
  -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e STOP_AFTER=$STOP -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key \
  -w $ROOT truth_dec bash $ROOT/review_e25/claude_gpt_review/from_claude/code/scripts/review_checks/full_variant_inner.sh "$MODEL" "$CONFIG" "$ROOT/$SUB" >> "$HOSTLOG" 2>&1
echo "== $(date '+%F %T') launcher exit $? $MODEL config=$CONFIG" >> "$HOSTLOG"

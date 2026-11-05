#!/usr/bin/env bash
# Host side: full_v1_v2.sh MODEL GPU [STOP_AFTER]  -> runs full_v1_inner_v2.sh in truth_dec; log appended to full_v1.log
set -u
MODEL=$1; GPU=$2; STOP=${3:-judge-diagnostics}
ROOT=/workspace/Truth_Decoupling
RES=$ROOT/results_qa_das_v1
HOSTLOG=/data/sanshizhao/Truth_Decoupling/results_qa_das_v1/$MODEL/full_v1.log
mkdir -p "$(dirname "$HOSTLOG")"
docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed \
  -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e STOP_AFTER=$STOP -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key \
  -w $ROOT truth_dec bash $ROOT/review_e25/checks/gpt16/full_v1_inner_v2.sh "$MODEL" >> "$HOSTLOG" 2>&1
echo "== $(date '+%F %T') launcher v2 exit $? $MODEL" >> "$HOSTLOG"

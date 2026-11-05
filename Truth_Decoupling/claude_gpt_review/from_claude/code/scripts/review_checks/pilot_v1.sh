#!/usr/bin/env bash
# Host side: pilot_v1.sh MODEL LAYER GPU -> runs the in-container pilot, log at results_qa_das_v1/MODEL/pilot3_v1.log
set -u
MODEL=$1; LAYER=$2; GPU=$3
ROOT=/workspace/Truth_Decoupling
RES=$ROOT/results_qa_das_v1
HOSTLOG=/data/sanshizhao/Truth_Decoupling/results_qa_das_v1/$MODEL/pilot3_v1.log
mkdir -p "$(dirname "$HOSTLOG")"
docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed \
  -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -w $ROOT truth_dec bash $ROOT/review_e25/checks/gpt16/pilot_v1_inner.sh "$MODEL" "$LAYER" > "$HOSTLOG" 2>&1
echo "exit $? $MODEL"

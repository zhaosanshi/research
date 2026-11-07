#!/usr/bin/env bash
# E42 launcher (host side): e42_run.sh MODEL GPU PHASE [extra e42 args...]
#   MODEL: llama3.1-8b | qwen3-8b | gemma3-4b ; PHASE: oracle-check | oracle-tune | select-check | select-tune | judge-tune
# Runs inside the truth_dec container as uid 1000 (never root: root-owned results cannot be deleted by ubuntu).
# Code: the Claude r23 snapshot; results: results_e42_20261108/<model>/<phase-tag>; judge key only via *_FILE.
set -euo pipefail
MODEL=${1:?model}; GPU=${2:?gpu}; PHASE=${3:?phase}; shift 3
ROOT=/workspace/Truth_Decoupling
CODE=$ROOT/review_r23_claude/claude_gpt_review/from_claude/code
RES=$ROOT/results_e42_20261108
TAG=${E42_TAG:-$PHASE}
SOURCE=$ROOT/results_qa_das_v1u/$MODEL/full
RUN=$RES/$MODEL/$TAG
mkdir -p /data/sanshizhao/Truth_Decoupling/results_e42_20261108/$MODEL/$TAG
docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed -e TD_MODEL_ROOT=$ROOT/models \
  -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 -w $CODE truth_dec python3 scripts/e42_qa_oracle_and_select.py "$PHASE" \
  --source-run "$SOURCE" --run-dir "$RUN" "$@"

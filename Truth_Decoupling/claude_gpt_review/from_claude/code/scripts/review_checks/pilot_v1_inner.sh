#!/usr/bin/env bash
# Runs INSIDE truth_dec: estimate -> train -> check for the 3-question QA-DAS v1 pilot of one model.
# usage: pilot_v1_inner.sh MODEL LAYER   (env: TD_RESULTS_DIR, TD_DATA_DIR, TD_MODEL_ROOT, CUDA_VISIBLE_DEVICES)
set -u
MODEL=$1; LAYER=$2
ROOT=/workspace/Truth_Decoupling
RES=${TD_RESULTS_DIR:-$ROOT/results_qa_das_v1}
RUN=$RES/$MODEL/pilot3_v1
mkdir -p "$RES/$MODEL"
cd "$ROOT/review_e25/claude_gpt_review/from_claude/code" || exit 1
echo "== $(date '+%F %T') estimate $MODEL layer $LAYER pilot 3 (GPU ${CUDA_VISIBLE_DEVICES:-?}) code=$(pwd)"
python3 scripts/e25_qa_das.py estimate --model "$MODEL" --run-dir "$RUN" --layers "$LAYER" --pilot 3 || { echo "ESTIMATE FAILED $MODEL"; exit 1; }
echo "== $(date '+%F %T') train"
python3 scripts/e25_qa_das.py train --model "$MODEL" --run-dir "$RUN" || { echo "TRAIN FAILED $MODEL"; exit 1; }
echo "== $(date '+%F %T') check"
python3 scripts/e25_qa_das.py check --model "$MODEL" --run-dir "$RUN" || { echo "CHECK FAILED $MODEL"; exit 1; }
echo "== $(date '+%F %T') PILOT DONE $MODEL"

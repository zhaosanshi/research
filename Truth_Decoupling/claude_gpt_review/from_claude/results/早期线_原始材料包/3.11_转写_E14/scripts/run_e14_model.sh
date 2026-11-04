#!/usr/bin/env bash
# E14 question-level (MC1/MC2/argmax) evaluation for one model, with the layers E11 used for that model.
# usage: run_e14_model.sh <gpu> <model> <E1> <E2> <E3> <probe> <write> [mc_dataset exclude_dataset exp e1exp root tag]
set -u
GPU="$1"; MODEL="$2"; E1="$3"; E2="$4"; E3="$5"; PL="$6"; WL="$7"
MC="${8:-truthfulqa_mc}"; EX="${9:-truthfulqa_tf}"; EXP="${10:-E2b_tqa}"; E1EXP="${11:-E1_directions_tqa}"; ROOT="${12:-results_tqa}"; TAG="${13:-}"
BASE=/data/sanshizhao/Truth_Decoupling
cd "$BASE"; mkdir -p logs
DEX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e CUDA_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 \
  -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/$ROOT -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed \
  -w /workspace/Truth_Decoupling/code/experiments truth_dec"
TAGARG=""; [ -n "$TAG" ] && TAGARG="--tag $TAG"
echo "[$(date +%H:%M:%S)] e14 start $MODEL $MC E=$E1,$E2,$E3 probe=$PL write=$WL"
$DEX python3 scripts/e14_mc_eval.py --exp "$EXP" --e1-exp "$E1EXP" --model "$MODEL" --mc-dataset "$MC" --exclude-dataset "$EX" \
  --e-layers $E1 $E2 $E3 --probe-layer $PL --write-layer $WL $TAGARG
echo "[$(date +%H:%M:%S)] e14 rc=$? $MODEL $MC"

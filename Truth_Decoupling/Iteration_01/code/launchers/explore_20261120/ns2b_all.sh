#!/bin/bash
# 2026-11-21 01:1x, second allocation of ns2_all.sh (its two waiting lanes were stopped; its Gemma-3-12B lane keeps
# running): Llama and Qwen3.5-9B move to the free memory of GPU 1 and start now; the two 27B models keep GPUs 2,3.
# New step after each model, three(): the release-ladder passes (E + C toward answering at 0.25 / 0.5 / 1 / 2) on the
# TEST split at the strengths named by diag/ns_choose.py from the tune split only (N1's best tune strength, and the
# stability filter's best tune strength when it uses C). Function one() is the one of ns2_all.sh, unchanged.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
STRENGTHS="0.5 1 1.5 3 4"
one() {  # one MODEL GPUS CAUT(1|0)
  local MODEL=$1 GPUS=$2 caut=$3 PY EL CL t sp s k single ladder cau arms
  read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || { echo "ns2 $MODEL: not in the config" >> $L/ns2_all.log; return 1; }
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
  single=""; ladder=""; cau=""
  for s in $STRENGTHS; do single="$single x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do ladder="$ladder x:E$EL.w.p$s+C$CL.w.m$k"; done; done
  if [ $caut = 1 ]; then
    for k in 0.25 0.5 1 2; do cau="$cau x:C$CL.w.p$k"; done
    for s in 0.5 1 2; do for k in 0.25 0.5 1 2; do cau="$cau x:E$EL.w.p$s+C$CL.w.p$k"; done; done
  fi
  for sp in tune test; do
    if [ $sp = tune ]; then arms="$single $cau $ladder"; else arms="$single $cau"; fi
    for t in popqa triviaqa; do
      "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split $sp \
          --out $NS/gen/$t/$MODEL/$sp --arms $arms >> $L/ns2_$MODEL.log 2>&1
      echo "ns2 $MODEL $t $sp exit=$? $(date '+%T')" >> $L/ns2_all.log
    done
  done
  touch $L/NS2_${MODEL}_DONE
}
three() {  # three MODEL GPUS : test ladders at the tune-chosen strengths
  local MODEL=$1 GPUS=$2 PY EL CL t s k ss arms
  read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || return 1
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
  for t in popqa triviaqa; do
    ss=$(python3 -B Iteration_01/diag/ns_choose.py $t $MODEL 2>> $L/ns3_$MODEL.log | tail -1)
    case "$ss" in ""|*[!0-9.\ ]*) echo "ns3 $MODEL $t choose FAILED ($ss)" >> $L/ns2_all.log; continue ;; esac
    arms=""
    for s in $ss; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
    "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split test \
        --out $NS/gen/$t/$MODEL/test --arms $arms >> $L/ns3_$MODEL.log 2>&1
    echo "ns3 $MODEL $t test ladders at E $ss exit=$? $(date '+%T')" >> $L/ns2_all.log
  done
  touch $L/NS3_${MODEL}_DONE
}
echo "$(date '+%F %T') second allocation start" >> $L/ns2_all.log
( one llama3.1-8b 1 0; three llama3.1-8b 1; one qwen3.5-9b 1 1; three qwen3.5-9b 1 ) &
( until [ -e $L/NS2_gemma3-12b_DONE ]; do sleep 60; done; three gemma3-12b 3
  until [ -e $L/LOW_GPU_g27_DONE ] && [ -e $L/NSCAUT_llama3.1-8b_DONE ]; do sleep 60; done
  one qwen3.8-27b 2,3 1; three qwen3.8-27b 2,3; one gemma3-27b 2 1; three gemma3-27b 2 ) &
wait
touch $L/NS2B_DONE

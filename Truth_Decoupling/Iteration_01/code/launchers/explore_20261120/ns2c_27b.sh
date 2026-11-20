#!/bin/bash
# 2026-11-21 01:2x, replaces the second lane of ns2b_all.sh (stopped while it was waiting; the first lane, Llama and
# Qwen3.5-9B on GPU 1, keeps running): the two 27B models get a shorter list. Reason: a 500-question pass of a 27B model
# takes several minutes and the full list is 47 passes per task; on Gemma-3-12B (six strengths) and on all five models
# at strength 2 the release step adds nothing, so for the 27B models only what the table needs is generated:
#   tune: single E and the release ladder at the five strengths (the choice of every setting needs them);
#   test: single E + ladder at the strengths named by diag/ns_choose.py from tune (function three(), unchanged).
# The cautious-C arms are left out for the 27B models (they can be added if the three smaller models call for it).
# The Gemma-3-12B step of the stopped lane (three gemma3-12b 3) is kept as it was.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
STRENGTHS="0.5 1 1.5 3 4"
one27() {  # one27 MODEL GPUS : tune only, single E + ladder at the five strengths
  local MODEL=$1 GPUS=$2 PY EL CL t s k arms
  read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || { echo "ns2c $MODEL: not in the config" >> $L/ns2_all.log; return 1; }
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
  arms=""
  for s in $STRENGTHS; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
  for t in popqa triviaqa; do
    "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split tune \
        --out $NS/gen/$t/$MODEL/tune --arms $arms >> $L/ns2_$MODEL.log 2>&1
    echo "ns2c $MODEL $t tune exit=$? $(date '+%T')" >> $L/ns2_all.log
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
echo "$(date '+%F %T') 27B lane (short list) start" >> $L/ns2_all.log
until [ -e $L/NS2_gemma3-12b_DONE ]; do sleep 60; done; three gemma3-12b 3
until [ -e $L/LOW_GPU_g27_DONE ] && [ -e $L/NSCAUT_llama3.1-8b_DONE ]; do sleep 60; done
one27 qwen3.8-27b 2,3; three qwen3.8-27b 2,3; one27 gemma3-27b 2; three gemma3-27b 2
touch $L/NS2C_DONE

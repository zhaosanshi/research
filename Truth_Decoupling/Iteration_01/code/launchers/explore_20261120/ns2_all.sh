#!/bin/bash
# 2026-11-21 00:5x, replaces ns_dose_all.sh (stopped): on the two factual datasets the release ladder toward answering
# adds nothing at any E strength on tune (Gemma-3-12B, both datasets; strength 2, five models), so the test passes of the
# 20 ladder arms per strength are not worth their time. Per model:
#   tune: single E and E + C toward answering (0.25 / 0.5 / 1 / 2) at E 0.5 / 1 / 1.5 / 3 / 4  (the evidence that the
#         ladder adds nothing there), plus the cautious-C arms of ns_caut.sh;
#   test: single E at the five strengths, plus the cautious-C arms.
# Existing complete passes are skipped by newsets_gen.py itself. Llama's cautious arms come from ns_caut.sh (running).
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
rm -f $L/ns2_all.log $L/NS2_DONE $L/NS2_*_DONE
echo "$(date '+%F %T') start" >> $L/ns2_all.log
( one gemma3-12b 3 1 ) &
( until [ -e $L/LOW_GPU_llama_DONE ]; do sleep 60; done; one llama3.1-8b 0 0; one qwen3.5-9b 0 1 ) &
( until [ -e $L/LOW_GPU_g27_DONE ] && [ -e $L/NS2_gemma3-12b_DONE ] && [ -e $L/NSCAUT_llama3.1-8b_DONE ]; do sleep 60; done
  one qwen3.8-27b 2,3 1; one gemma3-27b 2 1 ) &
wait
touch $L/NS2_DONE

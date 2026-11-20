#!/bin/bash
# 2026-11-21 00:3x (the lead: on TriviaQA and PopQA the strengths may differ from the TruthfulQA ones; find where N1 is
# better than E alone and than C alone): the N1 arms at E strengths 0.5 / 1 / 1.5 / 3 / 4 (2 exists) on the two new
# datasets, tune (192) and test (500) splits. Per strength: single E and E + C at 0.25 / 0.5 / 1 / 2, each a full pass
# (scripts/newsets/newsets_gen.py gen, exactly the call of launchers/newsets_c2c3_20261119.sh; same output directories,
# so the existing strength-2 files sit next to the new ones). Scoring by alias match needs no judge call
# (diag/ns_dose_table.py); DeepSeek Pro only later, for the settings chosen on tune.
# Lanes: GPU 0 Llama then Qwen3.5-9B (after low_all.sh's Llama part); GPU 3 Gemma-3-12B; then Qwen3.8-27B on GPUs 2,3 and
# Gemma-3-27B on GPU 2, once low_all.sh's Gemma-3-27B part, the Gemma-3-12B lane and the Llama pilot have left the cards.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
STRENGTHS=${STRENGTHS:-0.5 1 1.5 3 4}
one() {  # one MODEL GPUS
  local MODEL=$1 GPUS=$2 PY EL CL t sp s k arms rc
  read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || { echo "nsdose $MODEL: not in the config" >> $L/nsdose_all.log; return 1; }
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
  arms=""
  for s in $STRENGTHS; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
  for sp in tune test; do
    for t in triviaqa popqa; do
      "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split $sp \
          --out $NS/gen/$t/$MODEL/$sp --arms $arms >> $L/nsdose_$MODEL.log 2>&1
      rc=$?
      echo "nsdose $MODEL $t $sp exit=$rc $(date '+%T')" >> $L/nsdose_all.log
    done
  done
  touch $L/NSDOSE_${MODEL}_DONE
}
rm -f $L/nsdose_all.log $L/NSDOSE_DONE $L/NSDOSE_*_DONE
echo "$(date '+%F %T') start, strengths $STRENGTHS" >> $L/nsdose_all.log
( until [ -e $L/LOW_GPU_llama_DONE ]; do sleep 60; done; one llama3.1-8b 0; one qwen3.5-9b 0 ) &
( one gemma3-12b 3 ) &
( until [ -e $L/LOW_GPU_g27_DONE ] && [ -e $L/NSDOSE_gemma3-12b_DONE ] && grep -q "keep llama mmlu exit" $L/keep_all.log 2>/dev/null; do sleep 60; done
  one qwen3.8-27b 2,3; one gemma3-27b 2 ) &
wait
touch $L/NSDOSE_DONE

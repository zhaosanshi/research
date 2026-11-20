#!/bin/bash
# 2026-11-21 01:3x, exploration: confirmation passes on the SECOND batch of test questions (test2) for the cautious-C
# combination on TriviaQA / PopQA. Per task the three arms named by diag/ns_caut_confirm.py --choose (single E, single C
# toward caution, both together; each chosen on tune + the first test batch) are generated on test2. Table:
# python3 -B Iteration_01/diag/ns_caut_confirm.py MODEL ...
# usage: ns4_test2.sh MODEL GPUS 'SHELL CONDITION TO WAIT FOR'   (the condition is evaluated every 60 s; log lines and
#        the marker NS4_<MODEL>_DONE go to the log directory)
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
MODEL=$1; GPUS=$2; COND=${3:-true}
[ -n "$MODEL" ] && [ -n "$GPUS" ] || { echo "usage: ns4_test2.sh MODEL GPUS [CONDITION]"; exit 2; }
PY=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || exit 2
until eval "$COND"; do sleep 60; done
echo "$(date '+%F %T') ns4 $MODEL start on GPU $GPUS" >> $L/ns4_all.log
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
for t in popqa triviaqa; do
  arms=$(python3 -B Iteration_01/diag/ns_caut_confirm.py --choose $t $MODEL 2>> $L/ns4_$MODEL.log | tail -1)
  case "$arms" in x:*) ;; *) echo "ns4 $MODEL $t choose FAILED ($arms) $(date '+%T')" >> $L/ns4_all.log; continue ;; esac
  echo "ns4 $MODEL $t chosen: $arms" >> $L/ns4_all.log
  "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_bank_test2_20261120.jsonl --split test2 \
      --out $NS/gen/$t/$MODEL/test2 --arms $arms >> $L/ns4_$MODEL.log 2>&1
  echo "ns4 $MODEL $t test2 exit=$? $(date '+%T')" >> $L/ns4_all.log
done
touch $L/NS4_${MODEL}_DONE

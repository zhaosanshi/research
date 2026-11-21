#!/bin/bash
# 2026-11-21 23:5x, exploration, a pilot on one or two models first (the lead: a small run shows the rough result):
# TriviaQA / PopQA at the write rule "prompt" with the C strength allowed to take the CAUTIOUS sign (arm sign p), E at
# the main-table layer. At the training rule this was the one composition that beat single E on fresh questions
# (summary section 12.3; three models); the question here is whether it does at the rule the method would use.
#   1. tune (192): single C toward caution at 0.25 / 0.5 / 1 / 2 and, for every strength of the sweep, E + C toward
#      caution at the four strengths (single E too where the sweep of ns_prompt.sh has not produced it);
#   2. diag/nsc_prompt_table.py --choose MODEL chooses every setting on tune (file nsc_choice_<MODEL>.txt);
#   3. only the chosen arms on test and test2 (500 each).
# Arms whose file already exists are skipped, so a rerun continues. Table: diag/nsc_prompt_table.py --models MODEL.
# usage: nsc_prompt.sh MODEL GPUS E_LAYER "SWEEP STRENGTHS" [FILE_TO_WAIT_FOR]
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
NSP=$CODE/scripts/newsets/ns_positions.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
MODEL=$1; GPUS=$2; EL=$3; SWEEP=$4; WAITF=${5:-}
[ -n "$SWEEP" ] || { echo "usage: nsc_prompt.sh MODEL GPUS E_LAYER \"SWEEP STRENGTHS\" [FILE_TO_WAIT_FOR]"; exit 2; }
if [ -n "$WAITF" ]; then until [ -e "$WAITF" ]; do sleep 20; done; fi
read -r PY CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || { echo "nsc $MODEL: not in the config" >> $L/nsc_all.log; exit 1; }
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
gen() {  # gen SPLIT TASK ARM...: the arms of this split and task that have no file yet
  local sp=$1 t=$2 todo="" a f; shift 2
  local items=$NS/bank/${t}_bank_20261119.jsonl; [ $sp = test2 ] && items=$NS/bank/${t}_bank_test2_20261120.jsonl
  local d=$HNS/gen_prompt/$t/$MODEL/$sp
  mkdir -p $d
  [ -e $d/x.json ] || cp -p $HNS/gen/$t/$MODEL/$sp/x.json $d/x.json || { echo "nsc $MODEL $t $sp: no no-write pass to copy" >> $L/nsc_all.log; return 1; }
  for a in "$@"; do f=${a//:/--}; f=${f//+/-and-}; f=${f//./-}; [ -e $d/$f.json ] || todo="$todo $a"; done
  [ -n "$todo" ] || { echo "nsc $MODEL $t $sp: nothing to do $(date '+%T')" >> $L/nsc_all.log; return 0; }
  "${DX[@]}" $PY -I $NSP prompt gen --models $M --model $MODEL --task $t --items $items --split $sp \
      --out $NS/gen_prompt/$t/$MODEL/$sp --arms $todo >> $L/nsc_$MODEL.log 2>&1
  echo "nsc $MODEL $t $sp ($(echo $todo | wc -w) arms) exit=$? $(date '+%T')" >> $L/nsc_all.log
}
arms=""
for k in 0.25 0.5 1 2; do arms="$arms x:C$CL.w.p$k x:C$CL.w.m$k"; done
for s in $SWEEP; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.p$k"; done; done
echo "$(date '+%F %T') nsc $MODEL start on GPU $GPUS: E$EL sweep $SWEEP, C$CL toward caution" >> $L/nsc_all.log
for t in popqa triviaqa; do gen tune $t $arms; done
python3 -B Iteration_01/diag/nsc_prompt_table.py --choose $MODEL > $L/nsc_choice_$MODEL.txt 2>> $L/nsc_$MODEL.log || { echo "nsc $MODEL: the choice on tune failed" >> $L/nsc_all.log; exit 1; }
# the sweep of ns_prompt.sh may still be writing this model's test / test2 files: let it finish first
while ps -eo args | grep -q "[n]s_prompt.sh $MODEL "; do sleep 20; done
while read -r t more; do
  [ -n "$more" ] || continue
  for sp in test test2; do gen $sp $t $more; done
done < $L/nsc_choice_$MODEL.txt
touch $L/NSC_${MODEL}_DONE

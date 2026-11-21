#!/bin/bash
# 2026-11-21 23:5x, exploration, a pilot on one or two models first (the lead: a small run shows the rough result):
# TriviaQA / PopQA at the write rule "prompt" with the C strength allowed to take the CAUTIOUS sign (arm sign p), E at
# the main-table layer. At the training rule this was the one composition that beat single E on fresh questions
# (summary section 12.3; three models); the question here is whether it does at the rule the method would use.
#   1. tune (192): single C toward caution at 0.25 / 0.5 / 1 / 2 and, for every strength of the sweep, E + C toward
#      caution at the four strengths (single E and single C toward answering too where the sweep of ns_prompt.sh has
#      not produced them);
#   2. diag/nsc_prompt_table.py --choose MODEL chooses every setting on tune (file nsc_choice_<MODEL>.txt);
#   3. only the chosen arms on test and test2 (500 each).
# Arms whose file already exists are skipped, so a rerun continues. Table: diag/nsc_prompt_table.py --models MODEL.
# 2026-11-22 00:2x, after the read-only review (the Llama, Gemma-3-12B and Gemma-3-27B runs used the version before):
# the logged exit code is the generator's own (it was the status of a command substitution, always 0); a model that is
# not in the config stops the script; the wait for a running ns_prompt.sh of the same model comes before the tune step
# too; the DONE marker is written only when no step failed (NSC_<MODEL>_FAILED otherwise).
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
print(m["python"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)"
[ -n "$PY" ] && [ -n "$CL" ] || { echo "nsc $MODEL: not in the config" >> $L/nsc_all.log; exit 1; }
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
FAIL=0
gen() {  # gen SPLIT TASK ARM...: the arms of this split and task that have no file yet
  local sp=$1 t=$2 todo="" a f rc; shift 2
  local items=$NS/bank/${t}_bank_20261119.jsonl; [ $sp = test2 ] && items=$NS/bank/${t}_bank_test2_20261120.jsonl
  local d=$HNS/gen_prompt/$t/$MODEL/$sp
  mkdir -p $d
  [ -e $d/x.json ] || cp -p $HNS/gen/$t/$MODEL/$sp/x.json $d/x.json || { echo "nsc $MODEL $t $sp: no no-write pass to copy" >> $L/nsc_all.log; FAIL=1; return 1; }
  for a in "$@"; do f=${a//:/--}; f=${f//+/-and-}; f=${f//./-}; [ -e $d/$f.json ] || todo="$todo $a"; done
  [ -n "$todo" ] || { echo "nsc $MODEL $t $sp: nothing to do $(date '+%T')" >> $L/nsc_all.log; return 0; }
  "${DX[@]}" $PY -I $NSP prompt gen --models $M --model $MODEL --task $t --items $items --split $sp \
      --out $NS/gen_prompt/$t/$MODEL/$sp --arms $todo >> $L/nsc_$MODEL.log 2>&1
  rc=$?
  [ $rc = 0 ] || FAIL=1
  echo "nsc $MODEL $t $sp ($(echo $todo | wc -w) arms) exit=$rc $(date '+%T')" >> $L/nsc_all.log
}
arms=""
for k in 0.25 0.5 1 2; do arms="$arms x:C$CL.w.p$k x:C$CL.w.m$k"; done
for s in $SWEEP; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.p$k"; done; done
# the sweep of ns_prompt.sh may still be writing this model's files: let it finish first
while ps -eo args | grep -q "[n]s_prompt.sh $MODEL "; do sleep 20; done
echo "$(date '+%F %T') nsc $MODEL start on GPU $GPUS: E$EL sweep $SWEEP, C$CL toward caution" >> $L/nsc_all.log
for t in popqa triviaqa; do gen tune $t $arms; done
python3 -B Iteration_01/diag/nsc_prompt_table.py --choose $MODEL > $L/nsc_choice_$MODEL.txt 2>> $L/nsc_$MODEL.log || { echo "nsc $MODEL: the choice on tune failed" >> $L/nsc_all.log; touch $L/NSC_${MODEL}_FAILED; exit 1; }
while read -r t more; do
  [ -n "$more" ] || continue
  for sp in test test2; do gen $sp $t $more; done
done < $L/nsc_choice_$MODEL.txt
if [ $FAIL = 0 ]; then touch $L/NSC_${MODEL}_DONE; else echo "nsc $MODEL: finished with failed steps" >> $L/nsc_all.log; touch $L/NSC_${MODEL}_FAILED; fi

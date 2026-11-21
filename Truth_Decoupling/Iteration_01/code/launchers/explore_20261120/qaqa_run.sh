#!/bin/bash
# 2026-11-22 01:4x, exploration: (QA)^2 for one model with every setting frozen from TruthfulQA at the write rule
# "prompt" (the general form of qaqa_pilot.sh; reading rule and predictions written before the first generation in
# discussion/work_queue/2026-11-22_queue.md section 2.8). Nothing is chosen on these questions.
# Arms: no write (x), prompt only (conf_E2), single C, single E, E + C at 0.25 / 0.5 / 1 / 2 (N1 is assembled from
# them by diag/qaqa_pilot_table.py). Generation through scripts/newsets/ns_task.py (newsets_gen.py and ns_positions.py
# unchanged). The judge (unchanged two-axis rubric, DeepSeek Pro, reference = the set's abstractive answer) runs in the
# background, so that the next model can use the card; it is not started in peak hours.
# Markers: QAQA_<TASK>_<MODEL>_GEN_DONE, then QAQA_<TASK>_<MODEL>_DONE or _FAILED.
# usage: qaqa_run.sh TASK BANK_FILE_NAME WORK_NAME MODEL GPUS E_LAYER E_STRENGTH SINGLE_C_STRENGTH
#   e.g. qaqa_run.sh qaqafull qaqa_full_bank_20261122.jsonl test_full570 llama3.1-8b 1 16 3.5 0.25
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
NT=$CODE/scripts/newsets/ns_task.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
TASK=$1; BANKF=$2; WORK=$3; MODEL=$4; GPUS=$5; EL=$6; S=$7; KC=$8
[ -n "$KC" ] || { echo "usage: qaqa_run.sh TASK BANK_FILE_NAME WORK_NAME MODEL GPUS E_LAYER E_STRENGTH SINGLE_C_STRENGTH"; exit 2; }
BANK=$NS/bank/$BANKF
OUT=$NS/gen_prompt/$TASK/$MODEL/test
W=$NS/judge/$TASK/$MODEL/$WORK; HW=$HNS/judge/$TASK/$MODEL/$WORK
log=$L/${TASK}_$MODEL.log
MK=$L/QAQA_${TASK}_${MODEL}
say() { echo "$(date '+%F %T') $TASK $MODEL: $*" >> $L/qaqa_all.log; }
rm -f ${MK}_GEN_DONE ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
[ -s $HNS/bank/$BANKF ] || fail "no bank $BANKF"
plan=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || fail "model not in the config"
read -r PY CL <<< "$plan"
[ -n "$PY" ] && [ -n "$CL" ] || fail "model not in the config"
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
   -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
E=x:E$EL.w.p$S
WARMS="$E x:C$CL.w.m$KC"
for k in 0.25 0.5 1 2; do WARMS="$WARMS $E+C$CL.w.m$k"; done
say "start on GPU $GPUS: E$EL at $S, C$CL, single C $KC"
"${DX[@]}" $PY -I $NT $TASK none gen --models $M --model $MODEL --task $TASK --items $BANK --split test --out $OUT \
    --arms x conf_E2 >> $log 2>&1 || fail "no-write generation"
"${DX[@]}" $PY -I $NT $TASK prompt gen --models $M --model $MODEL --task $TASK --items $BANK --split test --out $OUT \
    --arms $WARMS >> $log 2>&1 || fail "write generation"
if [ ! -e $HW/judge_input.json ]; then
  "${J[@]}" python3 -I $NT $TASK none export --task $TASK --model $MODEL --items $BANK --split test --gen-dir $OUT \
      --only-arms x conf_E2 $WARMS --out $W/judge_input.json >> $log 2>&1 || fail "export"
fi
if [ ! -e $HW/judge2/cells.jsonl ]; then
  "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation judge_input.json >> $log 2>&1 || fail "cells"
fi
say "generation and export done"
touch ${MK}_GEN_DONE
( secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
  if [ "$secs" -lt 900 ]; then say "FAILED: peak hours or too close to them (${secs}s left), judge not started"; touch ${MK}_FAILED; exit 1; fi
  if "${J[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $W --judge pro >> $log 2>&1 &&
     "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W --judge pro > $HW/single_pro_stdout.md 2>> $log; then
    say "judged"; touch ${MK}_DONE
  else
    say "FAILED: judge"; touch ${MK}_FAILED; exit 1
  fi ) > /dev/null 2>&1 < /dev/null &

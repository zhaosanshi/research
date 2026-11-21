#!/bin/bash
# 2026-11-22 01:3x, exploration, a pilot the lead approved at 01:15 ("try it"): does the composition hold on a second
# question set of the TruthfulQA kind? (QA)^2 (Kim et al., ACL 2023): search-engine questions, half of them with a
# false or unverifiable assumption, with reference answers. One model, 100 evaluation questions (fixed seed, 50 valid /
# 40 false assumption / 10 unverifiable; scripts/newsets/qaqa_bank.py), everything frozen from TruthfulQA at the write
# rule "prompt": the main-table E layer and strength, the two caps and the single-C strength chosen on the TruthfulQA
# tune split. Nothing is chosen on these questions.
# Arms: no write (x), prompt only (conf_E2), single C, single E, E + C at 0.25 / 0.5 / 1 / 2 (N1 is assembled from
# them by diag/qaqa_pilot_table.py). Generation through scripts/newsets/ns_task.py (newsets_gen.py and ns_positions.py
# unchanged); judge = the unchanged two-axis rubric with DeepSeek Pro, reference = the set's abstractive answer.
# About 800 judge cells. Off-peak only.
# usage: qaqa_pilot.sh MODEL GPUS E_LAYER E_STRENGTH SINGLE_C_STRENGTH
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
NT=$CODE/scripts/newsets/ns_task.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
MODEL=$1; GPUS=$2; EL=$3; S=$4; KC=$5
[ -n "$KC" ] || { echo "usage: qaqa_pilot.sh MODEL GPUS E_LAYER E_STRENGTH SINGLE_C_STRENGTH"; exit 2; }
BANK=$NS/bank/qaqa_pilot_bank_20261122.jsonl
OUT=$NS/gen_prompt/qaqa/$MODEL/test
W=$NS/judge/qaqa/$MODEL/test_pilot100; HW=$HNS/judge/qaqa/$MODEL/test_pilot100
log=$L/qaqa_$MODEL.log
say() { echo "$(date '+%F %T') qaqa $MODEL: $*" >> $L/qaqa_all.log; }
rm -f $L/QAQA_${MODEL}_DONE $L/QAQA_${MODEL}_FAILED
fail() { say "FAILED: $*"; touch $L/QAQA_${MODEL}_FAILED; exit 1; }
[ -s $HNS/bank/qaqa_pilot_bank_20261122.jsonl ] || fail "no pilot bank"
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
"${DX[@]}" $PY -I $NT qaqa none gen --models $M --model $MODEL --task qaqa --items $BANK --split test --out $OUT \
    --arms x conf_E2 >> $log 2>&1 || fail "no-write generation"
"${DX[@]}" $PY -I $NT qaqa prompt gen --models $M --model $MODEL --task qaqa --items $BANK --split test --out $OUT \
    --arms $WARMS >> $log 2>&1 || fail "write generation"
say "generation done"
if [ ! -e $HW/judge_input.json ]; then
  "${J[@]}" python3 -I $NT qaqa none export --task qaqa --model $MODEL --items $BANK --split test --gen-dir $OUT \
      --only-arms x conf_E2 $WARMS --out $W/judge_input.json >> $log 2>&1 || fail "export"
fi
secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 7200)))')
[ "$secs" -ge 900 ] || fail "peak hours or too close to them (${secs}s left): the judge is not started"
if [ ! -e $HW/judge2/cells.jsonl ]; then
  "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation judge_input.json >> $log 2>&1 || fail "cells"
fi
"${J[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $W --judge pro >> $log 2>&1 || fail "pro judge"
"${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W --judge pro > $HW/single_pro_stdout.md 2>> $log || fail "single"
say "judged"
touch $L/QAQA_${MODEL}_DONE

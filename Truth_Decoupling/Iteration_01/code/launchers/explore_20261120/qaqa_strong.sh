#!/bin/bash
# 2026-11-22 05:2x, exploration: on (QA)^2, the equal-answer-count strong comparator of the TruthfulQA main table
# (summary section 13.3): must-answer prompt (conf, no honest sentence) + E at the main-table layer and strength,
# written at "prompt"; the answers are ranked by the model's own confidence (mean token log-probability, scored with E
# at "prompt" as well) and the table keeps the N most confident ones, N = the number of answers of N1.
# The no-write must-answer pass (conf) is generated and judged too: it is the reference of the first-difference check
# of the confidence scorer and a comparator of its own (summary section 13.4, third row).
# Method, readings and prediction: discussion/work_queue/2026-11-22_queue.md section 2.8, committed (e05faa9) before
# this file was written and before any of these answers existed.
# Steps: generate conf (no write) and conf:E<L>.w.p<S> at "prompt" -> export + cells into the judge work dir WORK ->
# the judge in the background (DeepSeek Pro, two-axis rubric, not started in peak hours) -> on the card, the confidence
# of both passes (scoring batch 8; batch 1 only when the 98% greedy-reproduction check is missed at 8; neither that
# check nor the first-difference check is relaxed).
# Markers: QAQA_STRONG_<TASK>_<MODEL>_GPU_DONE (generation and both confidence files), _DONE (judged), _FAILED.
# usage: qaqa_strong.sh TASK BANK_FILE_NAME WORK_NAME MODEL GPUS E_LAYER E_STRENGTH
#   e.g. qaqa_strong.sh qaqafull qaqa_full_bank_20261122.jsonl test_full570_strong llama3.1-8b 1 16 3.5
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
NT=$CODE/scripts/newsets/ns_task.py
NCP=$CODE/scripts/newsets/ns_conf_pos.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
TASK=$1; BANKF=$2; WORK=$3; MODEL=$4; GPUS=$5; EL=$6; S=$7
[ -n "$S" ] || { echo "usage: qaqa_strong.sh TASK BANK_FILE_NAME WORK_NAME MODEL GPUS E_LAYER E_STRENGTH"; exit 2; }
BANK=$NS/bank/$BANKF
OUT=$NS/gen_prompt/$TASK/$MODEL/test; HOUT=$HNS/gen_prompt/$TASK/$MODEL/test
W=$NS/judge/$TASK/$MODEL/$WORK; HW=$HNS/judge/$TASK/$MODEL/$WORK
CF=$NS/conf/$TASK/$MODEL/test; HCF=$HNS/conf/$TASK/$MODEL/test
ARM=conf:E$EL.w.p$S
TOK=conf--E$EL-w-p$(echo "$S" | tr . -)          # newsets_gen.arm_token of ARM
log=$L/qaqastrong_${TASK}_$MODEL.log
MK=$L/QAQA_STRONG_${TASK}_${MODEL}
say() { echo "$(date '+%F %T') qaqa strong $TASK $MODEL: $*" >> $L/qaqa_all.log; }
rm -f ${MK}_GPU_DONE ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
[ -s $HNS/bank/$BANKF ] || fail "no bank $BANKF"
plan=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || fail "model not in the config"
PY=$plan
[ -n "$PY" ] || fail "model not in the config"
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
   -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
say "start on GPU $GPUS: conf and $ARM at prompt"
"${DX[@]}" $PY -I $NT $TASK none gen --models $M --model $MODEL --task $TASK --items $BANK --split test --out $OUT \
    --arms conf >> $log 2>&1 || fail "no-write generation"
"${DX[@]}" $PY -I $NT $TASK prompt gen --models $M --model $MODEL --task $TASK --items $BANK --split test --out $OUT \
    --arms $ARM >> $log 2>&1 || fail "write generation"
[ -s $HOUT/conf.json ] && [ -s $HOUT/$TOK.json ] || fail "a generation file is missing ($TOK)"
if [ ! -e $HW/judge_input.json ]; then
  "${J[@]}" python3 -I $NT $TASK none export --task $TASK --model $MODEL --items $BANK --split test --gen-dir $OUT \
      --only-arms conf $ARM --out $W/judge_input.json >> $log 2>&1 || fail "export"
fi
if [ ! -e $HW/judge2/cells.jsonl ]; then
  "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation judge_input.json >> $log 2>&1 || fail "cells"
fi
say "generation and export done"
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
# confidence of one pass: $1 = host path of the output, the rest = the command without --score-batch
score() {
  local hout=$1; shift
  if [ -e "$hout" ]; then say "confidence $(basename $hout) already there"; return 0; fi
  local tmp=$L/.qaqastrong_${TASK}_${MODEL}_$$.out
  "$@" > $tmp 2>&1; local rc=$?
  cat $tmp >> $log
  if [ "$rc" -eq 0 ]; then rm -f $tmp; say "confidence $(basename $hout) done (scoring batch 8)"; return 0; fi
  if ! grep -q "teacher-forced arg-max reproduces only" $tmp; then rm -f $tmp; return 1; fi
  rm -f $tmp
  say "confidence $(basename $hout): the 98% check was missed at scoring batch 8, once more at scoring batch 1"
  "$@" --score-batch 1 >> $log 2>&1 || return 1
  say "confidence $(basename $hout) done (scoring batch 1)"
}
score $HCF/conf.json "${DX[@]}" $PY -I $NT $TASK none conf --task $TASK --models $M --model $MODEL --items $BANK \
    --gen-file $OUT/conf.json --out $CF/conf.json || fail "confidence of the no-write must-answer pass"
score $HCF/$TOK-at-prompt.json "${DX[@]}" $PY -I $NCP $TASK prompt conf --task $TASK --models $M --model $MODEL --items $BANK \
    --gen-file $OUT/$TOK.json --ref-gen-file $OUT/conf.json --out $CF/$TOK-at-prompt.json || fail "confidence of $ARM at prompt"
[ -s $HCF/conf.json ] && [ -s $HCF/$TOK-at-prompt.json ] || fail "a confidence file is missing"
touch ${MK}_GPU_DONE
say "generation and confidence done"

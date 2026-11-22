#!/bin/bash
# 2026-11-22 20:4x, exploration, queue H section 5: cell 0 = plain prompt + honest sentence (x_E2,
# configs/prompts/e44_eval_extra_h22_v1.json), NO write, through scripts/newsets/ns_h22.py. One model, one card (lock).
#   1. checks on 16 questions ((QA)^2 and PopQA test): the no-write plain prompt x regenerated through the entry = the
#      existing x.json (the entry changes nothing else); x_E2 on the first 2 batches = the first 16 rows of the full
#      x_E2 pass (the same setting run twice);
#   2. x_E2 on (QA)^2 570 (newsets/gen_prompt/h22/qaqafull/<model>/test/x_E2.json; judge work dir
#      newsets/judge/qaqafull/<model>/test_full570_h22z, DeepSeek Pro off peak, reply cache of test_full570 merged first);
#   3. x_E2 on PopQA / TriviaQA test and test2 (newsets/gen_prompt/h22/<task>/<model>/<split>/x_E2.json; alias match in
#      the table program, no judge).
# Markers H22Z_<MODEL>_GPU_DONE / _FAILED, H22Z_<MODEL>_JUDGE_DONE / _FAILED.  usage: h22z_model.sh KEY GPU
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2
[ -n "$GPU" ] || { echo "usage: h22z_model.sh KEY GPU"; exit 2; }
TF=$ROOT/envs/tf519/bin/python
case $KEY in
  llama) MODEL=llama3.1-8b; PYM=python3; NEED=20000 ;;
  q9)    MODEL=qwen3.5-9b; PYM=$TF; NEED=22000 ;;
  q27)   MODEL=qwen3.8-27b; PYM=$TF; NEED=62000 ;;
  g12)   MODEL=gemma3-12b; PYM=python3; NEED=30000 ;;
  g27)   MODEL=gemma3-27b; PYM=python3; NEED=62000 ;;
  *) echo "unknown key $KEY"; exit 2 ;;
esac
MK=$L/H22Z_$MODEL; log=$L/h22z_$MODEL.log
say() { echo "$(date '+%F %T') h22z $MODEL: $*" >> $L/h22_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_GPU_DONE
NH=$ROOT/Iteration_01/code/scripts/newsets/ns_h22.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
LOCK=$HNS/selfjudge/.gpu$GPU.lock
exec 9>"$LOCK" || fail "cannot open $LOCK"
until flock -n 9; do sleep 60; done
while true; do
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1 == g {print $2}')
  [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $GPU"
  if [ $GPU = 1 ]; then LIM=66560; else LIM=73000; fi
  [ $((USED + NEED)) -le $LIM ] && break
  say "GPU $GPU holds $USED MiB, $NEED more would exceed $LIM; waiting"; sleep 60
done
say "start on GPU $GPU ($USED MiB in use)"
DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I $NH)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$ROOT/Iteration_01/code
   -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $ROOT/Iteration_01/code truth_dec)
bankf() { if [ $2 = test2 ]; then echo $NS/bank/$1_bank_test2_20261120.jsonl; else echo $NS/bank/$1_bank_20261119.jsonl; fi; }
gen() {  # gen TASK SPLIT OUT ARMS... [--first-batches 2]
  local task=$1 split=$2 out=$3; shift 3
  local items=$(bankf $task $split)
  [ $task = qaqafull ] && items=$NS/bank/qaqa_full_bank_20261122.jsonl
  "${DQ[@]}" $PYM -I $NH $task none gen --models $M --model $MODEL --task $task --items $items --split $split --out $out "$@" >> $log 2>&1
}
first() { ls $HOST/${1#$ROOT/}/s*.$2.json 2>/dev/null | head -1 | sed "s#^$HOST#$ROOT#"; }
# ---- (QA)^2: check, full pass, export
for task in qaqafull popqa; do
  CK=$NS/gen_prompt/h22chk/z/$task/$MODEL/test
  gen $task test $CK --arms x x_E2 --first-batches 2 || fail "check generation $task"
  a=$(first $CK x); b=$(first $CK x_E2)
  [ -n "$a" ] && [ -n "$b" ] || fail "check files $task"
  "${DC[@]}" compare $a $NS/gen_prompt/$task/$MODEL/test/x.json >> $log 2>&1 || fail "entry check $task: x differs"
done
say "entry checks passed"
QOUT=$NS/gen_prompt/h22/qaqafull/$MODEL/test
gen qaqafull test $QOUT --arms x_E2 || fail "generation qaqa"
"${DC[@]}" compare $(first $NS/gen_prompt/h22chk/z/qaqafull/$MODEL/test x_E2) $QOUT/x_E2.json >> $log 2>&1 || fail "twice check qaqa"
QW=$NS/judge/qaqafull/$MODEL/test_full570_h22z; HQW=$HNS/judge/qaqafull/$MODEL/test_full570_h22z
if [ ! -e $HQW/judge_input.json ]; then
  "${J[@]}" python3 -I $NH qaqafull none export --task qaqafull --model $MODEL --items $NS/bank/qaqa_full_bank_20261122.jsonl --split test \
      --gen-dir $QOUT --only-arms x_E2 --out $QW/judge_input.json >> $log 2>&1 || fail "export qaqa"
fi
[ -e $HQW/judge2/cells.jsonl ] || "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $QW --generation judge_input.json >> $log 2>&1 || fail "cells qaqa"
say "qaqa generated and exported"
# ---- fact sets
for task in popqa triviaqa; do
  for split in test test2; do
    gen $task $split $NS/gen_prompt/h22/$task/$MODEL/$split --arms x_E2 || fail "generation $task $split"
  done
done
"${DC[@]}" compare $(first $NS/gen_prompt/h22chk/z/popqa/$MODEL/test x_E2) $NS/gen_prompt/h22/popqa/$MODEL/test/x_E2.json >> $log 2>&1 || fail "twice check popqa"
say "fact sets generated; card released"
flock -u 9; exec 9>&-
touch ${MK}_GPU_DONE
# ---- judge of (QA)^2 in the background (same guard as h22_model.sh)
(
  rm -f ${MK}_JUDGE_DONE ${MK}_JUDGE_FAILED
  mkdir -p $HQW/judge2/cache
  "${J[@]}" python3 scripts/review_checks/judge2_cache_merge.py $QW/judge2/cache $NS/judge/qaqafull/$MODEL/test_full570/judge2/cache >> $log 2>&1 || { touch ${MK}_JUDGE_FAILED; exit 1; }
  "${J[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $QW/judge2/cache/conflicts.json >> $log 2>&1 || { touch ${MK}_JUDGE_FAILED; exit 1; }
  secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
  if [ "$secs" -lt 900 ]; then say "judge not started: peak hours or too close"; touch ${MK}_JUDGE_FAILED; exit 1; fi
  if "${J[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $QW --judge pro >> $log 2>&1 &&
     "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $QW --judge pro > $HQW/single_pro_stdout.md 2>> $log; then
    say "judged qaqa"; touch ${MK}_JUDGE_DONE
  else
    say "FAILED: judge qaqa"; touch ${MK}_JUDGE_FAILED; exit 1
  fi
) > /dev/null 2>&1 < /dev/null &

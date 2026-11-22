#!/bin/bash
# 2026-11-22 23:2x, exploration, queue I section 7 (discussion/work_queue/2026-11-22_queue_i.md), one model, one card:
#   7.1 (QA)^2 570: x_E2:E<L>.w.p<s_sel> and x_E2:E<L>.w.p<s_sel>+C<CL>.w.m<c>, c = 0.25 0.5 1 2, at "prompt" through
#       scripts/newsets/ns_h22.py (gen dir newsets/gen_prompt/h22/qaqafull/<model>/test; judge work dir test_full570_h22n);
#   7.4 PopQA / TriviaQA test and test2: x_E2:E<L>.w.p<s_sel> at "prompt" (gen dirs newsets/gen_prompt/h22/<task>/...);
#   7.2 TruthfulQA tune and test: x_E2:C<CL>.w.m<c>, c = 0.5 1 2, at "prompt" through e44_positions_h22.py (export and
#       work dir n1sc_<split>).
# s_sel from newsets/h22/n1s_choice_<model>.json, layers from diag/n1_on_sentence.py plan (the main table's records).
# Checks on 16 questions (first 2 batches): the entry changes nothing (existing arms regenerated: x_E2:C<CL>.w.m0.25 on
# (QA)^2, x:E<L>.w.p<main> on PopQA test); the same setting twice (the 16-question run = the first 16 rows of the full
# pass) for (QA)^2 E and E + C 0.25, PopQA test E, TruthfulQA tune C 0.5; write records = the main table's (n1s_check
# parts). Then the card is released and the judges ((QA)^2, TruthfulQA tune, test) run in the background, off peak, with
# the reply caches of earlier work dirs merged first. Markers N1S2_<MODEL>_GPU_DONE / _FAILED, N1S2_<MODEL>_<qaqa|tune|test>_JUDGE_DONE.
# usage: n1s2_model.sh KEY GPU
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2
[ -n "$GPU" ] || { echo "usage: n1s2_model.sh KEY GPU"; exit 2; }
TF=$ROOT/envs/tf519/bin/python
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
case $KEY in
  llama) MODEL=llama3.1-8b; CODE=$ROOT/code_e45_20261112; PYM=python3; run=results_e44_factorial_das/llama3.1-8b_side_E2
         crun=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full; NEED=20000 ;;
  q9)    MODEL=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.5-9b_E2
         crun=$E44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b; NEED=22000 ;;
  q27)   MODEL=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
         crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm; NEED=62000 ;;
  g12)   MODEL=gemma3-12b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/12b/e44/gemma3-12b
         crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/$G3/12b/source/gemma3-12b; NEED=30000 ;;
  g27)   MODEL=gemma3-27b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
         crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/$G3/27b/source/gemma3-27b; NEED=62000 ;;
  *) echo "unknown key $KEY"; exit 2 ;;
esac
MK=$L/N1S2_$MODEL; log=$L/n1s2_$MODEL.log
say() { echo "$(date '+%F %T') n1s2 $MODEL: $*" >> $L/n1s_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_GPU_DONE
NEWP=$ROOT/Iteration_01/code/configs/prompts/e44_eval_extra_h22_v1.json
WR=$ROOT/Iteration_01/code/scripts/review_checks/e44_positions_h22.py
NC=$ROOT/Iteration_01/code/scripts/newsets/n1s_check.py
NH=$ROOT/Iteration_01/code/scripts/newsets/ns_h22.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B Iteration_01/diag/n1_on_sentence.py)
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
read -r EL S SH CL <<< "$("${RD[@]}" plan $MODEL 2>> $log)"
[ -n "$CL" ] || fail "plan"
SS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["strength_dot"])' $HNS/h22/n1s_choice_$MODEL.json) || fail "no s_sel"
D() { echo "$1" | tr . -; }
say "plan E$EL main $S, s_sel $SS, C$CL"
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
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
DT=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
bankf() { case $1 in qaqafull) echo $NS/bank/qaqa_full_bank_20261122.jsonl ;; *) if [ $2 = test2 ]; then echo $NS/bank/$1_bank_test2_20261120.jsonl; else echo $NS/bank/$1_bank_20261119.jsonl; fi ;; esac; }
gen() {  # gen TASK SPLIT OUT ARMS...
  local task=$1 split=$2 out=$3; shift 3
  "${DQ[@]}" $PYM -I $NH $task prompt gen --models $M --model $MODEL --task $task --items $(bankf $task $split) --split $split --out $out "$@" >> $log 2>&1
}
first() { ls $HOST/${1#$ROOT/}/s*.$2.json 2>/dev/null | head -1 | sed "s#^$HOST#$ROOT#"; }
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
EA=x_E2:E$EL.w.p$SS
# ---------------------------------------------------------------- 7.1 (QA)^2
QA=$NS/gen_prompt/qaqafull/$MODEL/test; QO=$NS/gen_prompt/h22/qaqafull/$MODEL/test; QC=$NS/gen_prompt/h22chk/n1s2/qaqafull/$MODEL/test
ARMS="$EA"; for c in 0.25 0.5 1 2; do ARMS="$ARMS $EA+C$CL.w.m$c"; done
gen qaqafull test $QC --arms x_E2:C$CL.w.m0.25 $EA $EA+C$CL.w.m0.25 --first-batches 2 || fail "qaqa check generation"
"${DC[@]}" $NH compare $(first $QC $(tok x_E2:C$CL.w.m0.25)) $QO/$(tok x_E2:C$CL.w.m0.25).json >> $log 2>&1 || fail "qaqa entry check"
gen qaqafull test $QO --arms $ARMS || fail "qaqa generation"
for a in $EA $EA+C$CL.w.m0.25; do "${DC[@]}" $NH compare $(first $QC $(tok $a)) $QO/$(tok $a).json >> $log 2>&1 || fail "qaqa twice check $a"; done
"${DC[@]}" $NC parts $QO/$(tok $EA).json $QA/x--E$EL-w-p$(D $S).json $SS $S E >> $log 2>&1 || fail "qaqa parts E"
for c in 0.25 0.5 1 2; do
  f=$QO/$(tok $EA+C$CL.w.m$c).json; mf=$QA/x--E$EL-w-p$(D $S)-and-C$CL-w-m$(D $c).json
  "${DC[@]}" $NC parts $f $mf $SS $S E >> $log 2>&1 && "${DC[@]}" $NC parts $f $mf $SS $S C >> $log 2>&1 || fail "qaqa parts $c"
done
QW=$NS/judge/qaqafull/$MODEL/test_full570_h22n; HQW=$HNS/judge/qaqafull/$MODEL/test_full570_h22n
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$ROOT/Iteration_01/code -w $ROOT/Iteration_01/code truth_dec)
if [ ! -e $HQW/judge_input.json ]; then
  "${J[@]}" python3 -I $NH qaqafull none export --task qaqafull --model $MODEL --items $(bankf qaqafull test) --split test \
      --gen-dir $QO --only-arms $ARMS --out $QW/judge_input.json >> $log 2>&1 || fail "qaqa export"
fi
[ -e $HQW/judge2/cells.jsonl ] || "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $QW --generation judge_input.json >> $log 2>&1 || fail "qaqa cells"
say "7.1 generated, checked, exported"
# ---------------------------------------------------------------- 7.4 fact sets
FC=$NS/gen_prompt/h22chk/n1s2/popqa/$MODEL/test
gen popqa test $FC --arms x:E$EL.w.p$S $EA --first-batches 2 || fail "fact check generation"
"${DC[@]}" $NH compare $(first $FC $(tok x:E$EL.w.p$S)) $NS/gen_prompt/popqa/$MODEL/test/$(tok x:E$EL.w.p$S).json >> $log 2>&1 || fail "fact entry check"
for task in popqa triviaqa; do
  for split in test test2; do
    out=$NS/gen_prompt/h22/$task/$MODEL/$split
    gen $task $split $out --arms $EA || fail "fact generation $task $split"
    "${DC[@]}" $NC parts $out/$(tok $EA).json $NS/gen_prompt/$task/$MODEL/$split/x--E$EL-w-p$(D $S).json $SS $S E >> $log 2>&1 || fail "fact parts $task $split"
  done
done
"${DC[@]}" $NH compare $(first $FC $(tok $EA)) $NS/gen_prompt/h22/popqa/$MODEL/test/$(tok $EA).json >> $log 2>&1 || fail "fact twice check"
say "7.4 generated and checked"
# ---------------------------------------------------------------- 7.2 TruthfulQA P + C
COMMON=(--source-run $SRC --run-dir $ROOT/$run --e-on $E_ON --c-run-dir $ROOT/$crun --c-e-on $C_E_ON
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $NEWP)
G=$ROOT/$run/eval_extra/gen
CARMS=""; for c in 0.5 1 2; do CARMS="$CARMS x_E2:C$CL.w.m$c"; done
"${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split tune --first-batches 2 --arms x_E2:C$CL.w.m0.5 >> $log 2>&1 || fail "TruthfulQA check generation"
for sp in tune test; do
  "${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split $sp --arms $CARMS >> $log 2>&1 || fail "TruthfulQA generation $sp"
  GEN=""
  for a in $CARMS; do
    f=$(tok $a)-at-prompt_$sp.json; GEN="$GEN $f"
    c=$(echo $a | sed 's/.*w\.m//')
    "${DC[@]}" $NC parts $G/$f $G/x--C$CL-w-m$(D $c)-at-prompt_$sp.json 1 1 C >> $log 2>&1 || fail "TruthfulQA parts $a $sp"
  done
  NAME=n1sc_$sp; W=$run/eval_extra/judge_${sp}_$NAME
  if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then
    "${DT[@]}" $PYM $WR prompt export "${COMMON[@]}" --split $sp --gen $GEN --name $NAME >> $log 2>&1 || fail "TruthfulQA export $sp"
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$sp/$NAME.json $HOST/$W/judge_input.json || fail "judge input $sp"
    "${DT[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "cells $sp"
  fi
done
"${DC[@]}" $NH compare $G/t16.$(tok x_E2:C$CL.w.m0.5)-at-prompt_tune.json $G/$(tok x_E2:C$CL.w.m0.5)-at-prompt_tune.json >> $log 2>&1 || fail "TruthfulQA twice check"
say "7.2 generated, checked, exported; card released"
flock -u 9; exec 9>&-
touch ${MK}_GPU_DONE
judge() {  # judge TAG CODE WORK(container) OLD_WORK_DIRS(container)...
  local tag=$1 code=$2 w=$3; shift 3
  local hw=$HOST/${w#$ROOT/} secs
  local JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$code
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $code truth_dec)
  rm -f ${MK}_${tag}_JUDGE_DONE ${MK}_${tag}_JUDGE_FAILED
  mkdir -p $hw/judge2/cache
  for old in "$@"; do
    [ -d $HOST/${old#$ROOT/}/judge2/cache ] || continue
    "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $w/judge2/cache $old/judge2/cache >> $log 2>&1 || { touch ${MK}_${tag}_JUDGE_FAILED; return 1; }
  done
  "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $w/judge2/cache/conflicts.json >> $log 2>&1 || { touch ${MK}_${tag}_JUDGE_FAILED; return 1; }
  secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
  if [ "$secs" -lt 900 ]; then say "judge $tag not started: peak hours or too close"; touch ${MK}_${tag}_JUDGE_FAILED; return 1; fi
  if "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $w --judge pro >> $log 2>&1 &&
     "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $w --judge pro > $hw/single_pro_stdout.md 2>> $log; then
    say "judged $tag"; touch ${MK}_${tag}_JUDGE_DONE
  else
    say "FAILED: judge $tag"; touch ${MK}_${tag}_JUDGE_FAILED; return 1
  fi
}
( judge qaqa $ROOT/Iteration_01/code $QW $NS/judge/qaqafull/$MODEL/test_full570_h22 $NS/judge/qaqafull/$MODEL/test_full570_h22z $NS/judge/qaqafull/$MODEL/test_full570 ) > /dev/null 2>&1 < /dev/null &
for sp in tune test; do
  ( judge $sp $CODE $ROOT/$run/eval_extra/judge_${sp}_n1sc_$sp $ROOT/$run/eval_extra/judge_${sp}_h22c4_$sp $ROOT/$run/eval_extra/judge_${sp}_pospromptc_$sp ) > /dev/null 2>&1 < /dev/null &
done
say "judges started"

#!/bin/bash
# 2026-11-23 01:4x, exploration, queue I section 8 (discussion/work_queue/2026-11-22_queue_i.md): E strength of the
# sentence method chosen on several tune splits; s_half = s_sel / 2. One model, one card (its lock).
# PHASE gen:  PopQA / TriviaQA tune: x_E2 (no write), x_E2:E(s_sel), x_E2:E(s_half); test and test2: x_E2:E(s_half)
#             (alias match later, no judge); (QA)^2 570: x_E2:E(s_half) and x_E2:E(s_half)+C(c), c = 0.25 0.5 1 2 (judge work
#             dir test_full570_h22n3; all 570 are judged, the choice reads the 100 pilot questions only); TruthfulQA tune
#             at s_half (E and E + C, export n1s3_tune) only where step one has no s_half pass (Qwen3.8-27B, Gemma-3-12B).
# PHASE test: TruthfulQA test at s_half (export n1s3_test), for the models whose choice is s_half.
# Checks (16 questions): the entry changes nothing (x_E2:E(s_sel) regenerated on PopQA test; x_E2:C<CL>.w.m0.25 on
# TruthfulQA tune in step one), the same setting twice (PopQA tune E(s_half), (QA)^2 E(s_half) and E(s_half)+C 0.25,
# TruthfulQA tune through n1s_check gpu), write records = the main table's, amount = main x strength ratio.
# Judges: DeepSeek Pro off peak, the reply cache of the S0 pass only (h22z / h22c4), the clock guard of qaqa_run.sh (stops
# 5 minutes before 9:00 Beijing time on a weekday). Markers N1S3_<MODEL>_<PHASE>_GPU_DONE / _FAILED,
# N1S3_<MODEL>_<qaqa|tune|test>_JUDGE_DONE / _FAILED.   usage: n1s3_model.sh KEY GPU gen|test
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2; PH=$3
[ "$PH" = gen ] || [ "$PH" = test ] || { echo "usage: n1s3_model.sh KEY GPU gen|test"; exit 2; }
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
MK=$L/N1S3_${MODEL}; log=$L/n1s3_${MODEL}_$PH.log
say() { echo "$(date '+%F %T') n1s3 $MODEL: $*" >> $L/n1s_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_${PH}_FAILED; exit 1; }
rm -f ${MK}_${PH}_FAILED ${MK}_${PH}_GPU_DONE
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
SH=$(python3 -c "import sys; h=float(sys.argv[1])/2; print(('%.4f' % h).rstrip('0').rstrip('.'))" $SS)
say "plan E$EL main $S, s_sel $SS, s_half $SH, C$CL ($PH)"
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
gen() {  # gen TASK POS SPLIT OUT ARMS...
  local task=$1 pos=$2 split=$3 out=$4; shift 4
  "${DQ[@]}" $PYM -I $NH $task $pos gen --models $M --model $MODEL --task $task --items $(bankf $task $split) --split $split --out $out "$@" >> $log 2>&1
}
first() { ls $HOST/${1#$ROOT/}/s*.$2.json 2>/dev/null | head -1 | sed "s#^$HOST#$ROOT#"; }
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
ES=x_E2:E$EL.w.p$SS; EH=x_E2:E$EL.w.p$SH
COMMON=(--source-run $SRC --run-dir $ROOT/$run --e-on $E_ON --c-run-dir $ROOT/$crun --c-e-on $C_E_ON
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $NEWP)
G=$ROOT/$run/eval_extra/gen
TQ_ARMS="$EH"; for c in 0.25 0.5 1 2; do TQ_ARMS="$TQ_ARMS $EH+C$CL.w.m$c"; done
tq_pass() {  # tq_pass SPLIT NAME: selfcheck, generation, checks, export of the five s_half arms on TruthfulQA
  local sp=$1 NAME=$2 GEN="" a f c mf
  "${DT[@]}" $PYM $WR prompt selfcheck "${COMMON[@]}" --split $sp --arms $EH+C$CL.w.m0.25 >> $log 2>&1 || fail "selfcheck $sp"
  "${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split $sp --arms $TQ_ARMS >> $log 2>&1 || fail "TruthfulQA generation $sp"
  for a in $TQ_ARMS; do
    f=$(tok $a)-at-prompt_$sp.json; GEN="$GEN $f"
    if [[ $a == *+C* ]]; then
      c=$(echo $a | sed 's/.*w\.m//'); mf=$G/x--E$EL-w-p$(D $S)-and-C$CL-w-m$(D $c)-at-prompt_$sp.json
      "${DC[@]}" $NC parts $G/$f $mf $SH $S E >> $log 2>&1 && "${DC[@]}" $NC parts $G/$f $mf $SH $S C >> $log 2>&1 || fail "TruthfulQA parts $a $sp"
    else
      "${DC[@]}" $NC parts $G/$f $G/x--E$EL-w-p$(D $S)-at-prompt_$sp.json $SH $S E >> $log 2>&1 || fail "TruthfulQA parts $a $sp"
    fi
  done
  local W=$run/eval_extra/judge_${sp}_$NAME
  if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then
    "${DT[@]}" $PYM $WR prompt export "${COMMON[@]}" --split $sp --gen $GEN --name $NAME >> $log 2>&1 || fail "TruthfulQA export $sp"
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$sp/$NAME.json $HOST/$W/judge_input.json || fail "judge input $sp"
    "${DT[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "cells $sp"
  fi
}
if [ $PH = gen ]; then
  # ---- fact sets
  FC=$NS/gen_prompt/h22chk/n1s3/popqa/$MODEL
  gen popqa prompt test $FC/test --arms $ES --first-batches 2 || fail "fact entry check generation"
  "${DC[@]}" $NH compare $(first $FC/test $(tok $ES)) $NS/gen_prompt/h22/popqa/$MODEL/test/$(tok $ES).json >> $log 2>&1 || fail "fact entry check"
  gen popqa prompt tune $FC/tune --arms $EH --first-batches 2 || fail "fact twice check generation"
  for task in popqa triviaqa; do
    out=$NS/gen_prompt/h22/$task/$MODEL/tune
    gen $task none tune $out --arms x_E2 || fail "fact S0 tune $task"
    gen $task prompt tune $out --arms $ES $EH || fail "fact SE tune $task"
    for a in $ES $EH; do
      s=$(echo $a | sed 's/.*w\.p//')
      "${DC[@]}" $NC parts $out/$(tok $a).json $NS/gen_prompt/$task/$MODEL/tune/x--E$EL-w-p$(D $S).json $s $S E >> $log 2>&1 || fail "fact parts $task tune $a"
    done
    for split in test test2; do
      out=$NS/gen_prompt/h22/$task/$MODEL/$split
      gen $task prompt $split $out --arms $EH || fail "fact SE(s_half) $task $split"
      "${DC[@]}" $NC parts $out/$(tok $EH).json $NS/gen_prompt/$task/$MODEL/$split/x--E$EL-w-p$(D $S).json $SH $S E >> $log 2>&1 || fail "fact parts $task $split"
    done
  done
  "${DC[@]}" $NH compare $(first $FC/tune $(tok $EH)) $NS/gen_prompt/h22/popqa/$MODEL/tune/$(tok $EH).json >> $log 2>&1 || fail "fact twice check"
  say "fact sets generated and checked"
  # ---- (QA)^2
  QA=$NS/gen_prompt/qaqafull/$MODEL/test; QO=$NS/gen_prompt/h22/qaqafull/$MODEL/test; QC=$NS/gen_prompt/h22chk/n1s3/qaqafull/$MODEL/test
  ARMS="$EH"; for c in 0.25 0.5 1 2; do ARMS="$ARMS $EH+C$CL.w.m$c"; done
  gen qaqafull prompt test $QC --arms $EH $EH+C$CL.w.m0.25 --first-batches 2 || fail "qaqa check generation"
  gen qaqafull prompt test $QO --arms $ARMS || fail "qaqa generation"
  for a in $EH $EH+C$CL.w.m0.25; do "${DC[@]}" $NH compare $(first $QC $(tok $a)) $QO/$(tok $a).json >> $log 2>&1 || fail "qaqa twice check $a"; done
  "${DC[@]}" $NC parts $QO/$(tok $EH).json $QA/x--E$EL-w-p$(D $S).json $SH $S E >> $log 2>&1 || fail "qaqa parts E"
  for c in 0.25 0.5 1 2; do
    f=$QO/$(tok $EH+C$CL.w.m$c).json; mf=$QA/x--E$EL-w-p$(D $S)-and-C$CL-w-m$(D $c).json
    "${DC[@]}" $NC parts $f $mf $SH $S E >> $log 2>&1 && "${DC[@]}" $NC parts $f $mf $SH $S C >> $log 2>&1 || fail "qaqa parts $c"
  done
  QW=$NS/judge/qaqafull/$MODEL/test_full570_h22n3; HQW=$HNS/judge/qaqafull/$MODEL/test_full570_h22n3
  J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
     -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$ROOT/Iteration_01/code -w $ROOT/Iteration_01/code truth_dec)
  if [ ! -e $HQW/judge_input.json ]; then
    "${J[@]}" python3 -I $NH qaqafull none export --task qaqafull --model $MODEL --items $(bankf qaqafull test) --split test \
        --gen-dir $QO --only-arms $ARMS --out $QW/judge_input.json >> $log 2>&1 || fail "qaqa export"
  fi
  [ -e $HQW/judge2/cells.jsonl ] || "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $QW --generation judge_input.json >> $log 2>&1 || fail "qaqa cells"
  say "qaqa generated, checked, exported"
  # ---- TruthfulQA tune at s_half where step one has none
  if [ ! -s $HOST/$run/eval_extra/gen/$(tok $EH)-at-prompt_tune.json ]; then
    CK=$NS/h22/n1s3_check_$MODEL.json
    [ -s $HOST/${CK#$ROOT/} ] || "${DQ[@]}" $PYM -I $NC gpu --models $M --model $MODEL --el $EL --cl $CL --s $SH \
        --base-file $G/../base_x_E2_tune.json --out $CK >> $log 2>&1 || fail "TruthfulQA checks (E at 0, twice)"
    tq_pass tune n1s3_tune
    "${DC[@]}" $NC cross $CK $G/$(tok $EH)-at-prompt_tune.json $G/$(tok $EH+C$CL.w.m0.25)-at-prompt_tune.json >> $log 2>&1 || fail "TruthfulQA check across runs"
    TQ_TUNE=1
  fi
  say "gen phase done; card released"
else
  tq_pass test n1s3_test
  say "test phase done; card released"
fi
flock -u 9; exec 9>&-
touch ${MK}_${PH}_GPU_DONE
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
if [ $PH = gen ]; then
  ( judge qaqa $ROOT/Iteration_01/code $QW $NS/judge/qaqafull/$MODEL/test_full570_h22z ) > /dev/null 2>&1 < /dev/null &
  [ -n "$TQ_TUNE" ] && ( judge tune $CODE $ROOT/$run/eval_extra/judge_tune_n1s3_tune $ROOT/$run/eval_extra/judge_tune_h22c4_tune ) > /dev/null 2>&1 < /dev/null &
else
  ( judge test $CODE $ROOT/$run/eval_extra/judge_test_n1s3_test $ROOT/$run/eval_extra/judge_test_h22c4_test ) > /dev/null 2>&1 < /dev/null &
fi
say "judges started"

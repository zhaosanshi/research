#!/bin/bash
# 2026-11-23, exploration, queue K (discussion/work_queue/2026-11-23_queue_k.md): the E direction trained with a keep term,
# Llama-3.1-8B, E at layer 16 (the main table's), C at layer 19 (unchanged), every write at "prompt".
# Code copy code_keepfix_20261123 = code_e45_20261112 + scripts/e44_keepfix.py (nothing else changed).
# PHASE prep:     the plain model's x answers on the TriviaQA / PopQA train splits (keepfix_ns_train.py -> newsets_gen.py,
#                 greedy, 48 tokens, batch 8), the keep set (keepfix_keep_set.py), the run dirs (frozen answers copied from
#                 the main run, sha256 checked), the keep-term unit test and the memory of one training step
#                 (e44_keepfix.py unittest). GPU NEED 20000 MiB; the unit test is capped at the card's room.
# PHASE traingen: lambda 0 in a check run dir, compared with the main E (cosine >= 0.9999, else stop); lambda 1 = E_k; the
#                 sign / unit write amount check of E_k (signcheck: else stop); the aligned strength s' = A / gap; tune
#                 generations: TruthfulQA (x_E2 + E_k at 3.5 and s', each with C19 at 0.25 0.5 1 2; x + E_k at 3.5 and s'),
#                 (QA)^2 570 (the same ten x_E2 arms; and the main E at half strength 1.75 with its four C arms), PopQA /
#                 TriviaQA tune (x_E2 + E_k at 3.5 and s'; x_E2, x_E2 + main E at 3.5 and 1.75 for comparison); checks;
#                 exports; judge cells.
#                 GPU NEED: the measured step memory (argument 3, MiB) for training, 20000 for generation.
# PHASE judge:    DeepSeek Pro (non-thinking, the judge script's config), one old reply cache merged per work dir, off peak
#                 only (weekday 9-12 / 14-18 Beijing time are peak; a run stops 5 minutes before a peak and resumes after it).
# Markers KEEPFIX_<MODEL>_<PHASE>_DONE / _FAILED; log Iteration_01/logs/newsets_20261119/keepfix_<MODEL>_<PHASE>.log.
# usage: keepfix_llama.sh prep GPU | traingen GPU TRAIN_NEED_MIB | judge
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
PH=$1; GPU=$2; TNEED=$3
case "$PH" in prep|traingen|judge) ;; *) echo "usage: keepfix_llama.sh prep GPU | traingen GPU TRAIN_NEED_MIB | judge"; exit 2 ;; esac
MODEL=llama3.1-8b; CODE=$ROOT/code_keepfix_20261123; PYM=python3
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
MAINR=results_e44_factorial_das/llama3.1-8b_side_E2; CRUN=results_e44_factorial_das/llama3.1-8b
KR=Iteration_01/results/keepfix/llama3.1-8b/e44
R1=$KR/llama3.1-8b_keepfix; R0=$KR/llama3.1-8b_keepfix_l0
EL=16; CL=19; S=3.5; NEED=20000
KF=$ROOT/Iteration_01/code/keepfix_20261123
NSK=Iteration_01/newsets/keepfix
KS=$NSK/keep_set_$MODEL.json; KIDS=$NSK/keep_ids_$MODEL.json
AMT=$MAINR/eval_extra/gen/x_E2--E16-w-p3-5-at-prompt_tune.json
NEWP=$ROOT/Iteration_01/code/configs/prompts/e44_eval_extra_h22_v1.json
WR=$ROOT/Iteration_01/code/scripts/review_checks/e44_positions_h22.py
NH=$ROOT/Iteration_01/code/scripts/newsets/ns_h22.py
NC=$ROOT/Iteration_01/code/scripts/newsets/n1s_check.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
KM=$KF/keepfix_models_20261123.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
BANKQ=$NS/bank/qaqa_full_bank_20261122.jsonl
QW=$NS/judge/qaqafull/$MODEL/keepfix_pilot100
MK=$L/KEEPFIX_${MODEL}_$PH; log=$L/keepfix_${MODEL}_$PH.log
say() { echo "$(date '+%F %T') keepfix $MODEL $PH: $*" >> $L/keepfix_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
first() { ls $HOST/${1#$ROOT/}/s*.$2.json 2>/dev/null | head -1 | sed "s#^$HOST#$ROOT#"; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
DT=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$ROOT/Iteration_01/code -w $ROOT/Iteration_01/code truth_dec)
KEEPA=(--keep-set $ROOT/$KS --keep-amount-from $ROOT/$AMT --keep-prompts-extra $NEWP)
TA=(--source-run $SRC --e-on E2 --family E --layers $EL --val-n 16 --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
lock_gpu() {  # lock_gpu NEED_MIB: the card's lock, then wait until NEED more MiB stay within its limit; sets USED, LIM, CAP
  local need=$1
  if [ "$GPU" = 0 ] && ! { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; }; then fail "GPU 0 only after the CAA grid"; fi
  LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$GPU.lock
  exec 9>"$LOCK" || fail "cannot open $LOCK"
  flock 9 || fail "cannot lock $LOCK"            # blocks in the lock's queue until the holder releases it
  if [ "$GPU" = 1 ]; then LIM=66560; else LIM=73000; fi
  while true; do
    USED=$(mem $GPU)
    [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $GPU"
    [ $((USED + need)) -le $LIM ] && break
    say "GPU $GPU holds $USED MiB, $need more would exceed $LIM; waiting"; sleep 60
  done
  CAP=$((LIM - USED - 1024))
  say "start on GPU $GPU ($USED MiB in use, limit $LIM, cap for PyTorch allocations $CAP MiB)"
}
# ---------------------------------------------------------------------------- judge functions
budget() {  # seconds until 5 minutes before the next weekday peak (Beijing time), at most 6 h; 0 inside a peak
  TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))'
}
judge_one() {  # judge_one TAG CODE WORK(container) OLD_WORK(container)
  local tag=$1 code=$2 w=$3 old=$4 hw=$HOST/${3#$ROOT/} secs rc log=$L/keepfix_${MODEL}_judge_$1.log
  local JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$code
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $code truth_dec)
  rm -f ${MK}_${tag}_DONE ${MK}_${tag}_FAILED
  [ -e $hw/judge2/cells.jsonl ] || { say "judge $tag: no cells"; touch ${MK}_${tag}_FAILED; return 1; }
  if [ ! -e ${MK}_${tag}_MERGED ]; then
    "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $w/judge2/cache $old/judge2/cache >> $log 2>&1 || { say "judge $tag: merge failed"; touch ${MK}_${tag}_FAILED; return 1; }
    "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $w/judge2/cache/conflicts.json >> $log 2>&1 || { say "judge $tag: cache conflict"; touch ${MK}_${tag}_FAILED; return 1; }
    echo "$old" > ${MK}_${tag}_MERGED
  fi
  while true; do
    secs=$(budget)
    if [ "$secs" -lt 900 ]; then sleep 300; continue; fi
    say "judge $tag: run for at most $secs s"
    "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $w --judge pro >> $log 2>&1
    rc=$?
    if [ $rc = 0 ]; then break; fi
    if [ $rc != 124 ]; then say "FAILED: judge $tag exit $rc"; touch ${MK}_${tag}_FAILED; return 1; fi
    say "judge $tag: stopped before the peak; resumes after it"
  done
  "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $w --judge pro > $hw/single_pro_stdout.md 2>> $log || { say "FAILED: single $tag"; touch ${MK}_${tag}_FAILED; return 1; }
  say "judged $tag"; touch ${MK}_${tag}_DONE
}
W_kf=$ROOT/$R1/eval_extra/judge_tune_kf_tune; O_kf=$ROOT/$MAINR/eval_extra/judge_tune_n1s_tune
W_kfx=$ROOT/$R1/eval_extra/judge_tune_kfx_tune; O_kfx=$ROOT/$MAINR/eval_extra/judge_tune_pospromptl16_tune
W_qaqa=$QW; O_qaqa=$NS/judge/qaqafull/$MODEL/test_full570_h22n1
W_qaqah=$NS/judge/qaqafull/$MODEL/keepfix_main_pilot100; O_qaqah=$NS/judge/qaqafull/$MODEL/test_full570_h22n1
jstart() {  # jstart TAG: one judge in the background (marker KEEPFIX_<MODEL>_<PHASE>_<TAG>_DONE / _FAILED)
  local t=$1 w o code
  eval w=\$W_$t; eval o=\$O_$t
  case $t in kf|kfx) code=$CODE ;; *) code=$ROOT/Iteration_01/code ;; esac
  ( exec 9>&-; judge_one $t $code $w $o ) > /dev/null 2>&1 < /dev/null &
}
passed() { "${DC[@]}" -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('passed') is True else 1)" $1; }
rm -f ${MK}_FAILED ${MK}_DONE
# ============================================================================ prep
if [ $PH = prep ]; then
  lock_gpu $NEED
  for task in triviaqa popqa; do
    out=$NS/keepfix/gen/$task/$MODEL/train
    if [ ! -s $HOST/${out#$ROOT/}/x.json ] || [ ! -s $HOST/${out#$ROOT/}/x_E2.json ]; then
      "${DQ[@]}" $PYM -I $KF/keepfix_ns_train.py gen --models $M --model $MODEL --task $task \
          --items $NS/bank/${task}_bank_20261119.jsonl --split train --arms x x_E2 --out $out >> $log 2>&1 || fail "train-split answers $task"
    fi
  done
  say "train-split answers done"
  [ -s $HOST/$KS ] || "${DC[@]}" $KF/keepfix_keep_set.py --model $MODEL --gen-root $NS/keepfix/gen --out-set $ROOT/$KS --out-ids $ROOT/$KIDS >> $log 2>&1 || fail "keep set"
  for d in $R1 $R0; do mkdir -p $HOST/$d || fail "mkdir $d"; done
  for f in frozen_train.json frozen_tune.json frozen_test.json; do
    [ -s $HOST/$R1/$f ] || cp -p $HOST/$MAINR/$f $HOST/$R1/$f || fail "copy $f"
  done
  [ -s $HOST/$R0/frozen_train.json ] || cp -p $HOST/$MAINR/frozen_train.json $HOST/$R0/frozen_train.json || fail "copy frozen_train to R0"
  for f in $R1/frozen_train.json $R1/frozen_tune.json $R1/frozen_test.json $R0/frozen_train.json; do
    a=$(sha256sum $HOST/$f | cut -c1-64); b=$(sha256sum $HOST/$MAINR/$(basename $f) | cut -c1-64)
    [ "$a" = "$b" ] || fail "copied $f differs from the main run's"
  done
  say "run dirs ready (frozen answers identical to the main run's)"
  UT=$ROOT/$NSK/unittest_$MODEL.json
  [ -s $HOST/${UT#$ROOT/} ] || "${DT[@]}" $PYM scripts/e44_keepfix.py unittest "${TA[@]}" --run-dir $ROOT/$R1 "${KEEPA[@]}" \
      --keep-check-dir $ROOT/$MAINR --keep-report $UT --keep-mem-cap-mib $CAP >> $log 2>&1 || fail "unit test"
  passed $UT || fail "unit test report says not passed"
  say "unit test passed; used before start $USED MiB"
  echo "USED_BEFORE_UNITTEST $USED" >> $log
  flock -u 9; exec 9>&-
  touch ${MK}_DONE; say "card released"; exit 0
fi
# ============================================================================ traingen
if [ $PH = traingen ]; then
  [ -n "$TNEED" ] || fail "traingen needs TRAIN_NEED_MIB"
  [ -e $L/KEEPFIX_${MODEL}_prep_DONE ] || fail "prep not done"
  lock_gpu $TNEED
  if [ ! -s $HOST/$R0/dirs/E_L$EL.pt ]; then
    "${DT[@]}" $PYM scripts/e44_keepfix.py train "${TA[@]}" --run-dir $ROOT/$R0 "${KEEPA[@]}" --keep-lambda 0 --keep-mem-cap-mib $CAP >> $log 2>&1 || fail "lambda 0 training"
  fi
  "${DT[@]}" $PYM scripts/e44_keepfix.py compare $ROOT/$R0/dirs/E_L$EL.pt $ROOT/$MAINR/dirs/E_L$EL.pt --min-cos 0.9999 > $HOST/$NSK/compare_l0_$MODEL.json 2>> $log \
      || fail "lambda 0 does not reproduce the main E (cosine below 0.9999): stop"
  say "lambda 0 reproduces: $(cat $HOST/$NSK/compare_l0_$MODEL.json | cut -c1-200)"
  if [ ! -s $HOST/$R1/dirs/E_L$EL.pt ]; then
    "${DT[@]}" $PYM scripts/e44_keepfix.py train "${TA[@]}" --run-dir $ROOT/$R1 "${KEEPA[@]}" --keep-lambda 1 --keep-mem-cap-mib $CAP >> $log 2>&1 || fail "lambda 1 training"
  fi
  "${DT[@]}" $PYM scripts/e44_keepfix.py compare $ROOT/$R1/dirs/E_L$EL.pt $ROOT/$MAINR/dirs/E_L$EL.pt --min-cos -1 > $HOST/$NSK/compare_l1_$MODEL.json 2>> $log || fail "compare E_k"
  "${DT[@]}" $PYM scripts/e44_keepfix.py signcheck $ROOT/$R1/dirs/E_L$EL.pt --l0 $ROOT/$R0/dirs/E_L$EL.pt > $HOST/$NSK/signcheck_$MODEL.json 2>> $log \
      || fail "sign / unit write amount check of E_k: stop"
  SP=$("${DT[@]}" $PYM scripts/e44_keepfix.py aligned $ROOT/$R1/dirs/E_L$EL.pt --keep-amount-from $ROOT/$AMT 2>> $log)
  [[ $SP =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "aligned strength"
  echo "$SP" > $HOST/$NSK/aligned_s_$MODEL.txt
  say "E_k trained; aligned strength s' = $SP; $(cat $HOST/$NSK/compare_l1_$MODEL.json | cut -c1-200)"
  ES=x_E2:E$EL.w.p$S; EA=x_E2:E$EL.w.p$SP; SH=1.75; EH=x_E2:E$EL.w.p$SH
  XARMS="$ES $EA"; for s in $S $SP; do for c in 0.25 0.5 1 2; do XARMS="$XARMS x_E2:E$EL.w.p$s+C$CL.w.m$c"; done; done
  TX="x:E$EL.w.p$S x:E$EL.w.p$SP"
  COMMON=(--source-run $SRC --run-dir $ROOT/$R1 --e-on E2 --c-run-dir $ROOT/$CRUN --c-e-on E1
          --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $NEWP)
  G1=$ROOT/$R1/eval_extra/gen; GM=$ROOT/$MAINR/eval_extra/gen
  # ---- TruthfulQA tune
  CK=$ROOT/$NSK/n1s_check_$MODEL.json
  [ -s $HOST/${CK#$ROOT/} ] || "${DQ[@]}" $PYM -I $NC gpu --models $KM --model $MODEL --el $EL --cl $CL --s $S \
      --base-file $GM/../base_x_E2_tune.json --out $CK >> $log 2>&1 || fail "TruthfulQA checks (E_k at 0 = no write = saved P, twice)"
  passed $CK || fail "TruthfulQA check report says not passed"
  "${DT[@]}" $PYM $WR prompt selfcheck "${COMMON[@]}" --split tune --arms $ES+C$CL.w.m0.25 >> $log 2>&1 || fail "selfcheck"
  "${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split tune --arms $XARMS $TX >> $log 2>&1 || fail "TruthfulQA generation"
  "${DC[@]}" $NC cross $CK $G1/$(tok $ES)-at-prompt_tune.json $G1/$(tok $ES+C$CL.w.m0.25)-at-prompt_tune.json >> $log 2>&1 || fail "TruthfulQA check across runs"
  for c in 0.25 0.5 1 2; do
    for s in $S $SP; do
      "${DC[@]}" $NC parts $G1/$(tok x_E2:E$EL.w.p$s+C$CL.w.m$c)-at-prompt_tune.json $GM/$(tok $ES+C$CL.w.m$c)-at-prompt_tune.json 1 1 C >> $log 2>&1 || fail "C part $s $c"
    done
  done
  "${DC[@]}" $KF/keepfix_table.py check-tq --run $ROOT/$R1 --main $ROOT/$MAINR --arms $XARMS $TX >> $log 2>&1 || fail "TruthfulQA write records / P"
  GEN1=""; for a in $XARMS; do GEN1="$GEN1 $(tok $a)-at-prompt_tune.json"; done
  GENX=""; for a in $TX; do GENX="$GENX $(tok $a)-at-prompt_tune.json"; done
  for pair in "kf_tune:$GEN1" "kfx_tune:$GENX"; do
    NAME=${pair%%:*}; GEN=${pair#*:}; W=$R1/eval_extra/judge_tune_$NAME
    if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then
      "${DT[@]}" $PYM $WR prompt export "${COMMON[@]}" --split tune --gen $GEN --name $NAME >> $log 2>&1 || fail "export $NAME"
      mkdir -p $HOST/$W/judge2/cache && cp $HOST/$R1/eval_extra/judge_tune/$NAME.json $HOST/$W/judge_input.json || fail "judge input $NAME"
      "${DT[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "cells $NAME"
    fi
  done
  say "TruthfulQA tune generated, checked, exported; balance before: $(bash $L/balance.sh)"
  jstart kf; jstart kfx
  # ---- (QA)^2 570 (judged: the 100 pilot questions)
  QO=$NS/gen_prompt/keepfix/qaqafull/$MODEL/test; QC=$NS/gen_prompt/keepfix_chk/qaqafull/$MODEL/test
  QM=$NS/gen_prompt/h22/qaqafull/$MODEL/test
  gq() { "${DQ[@]}" $PYM -I $NH qaqafull prompt gen --models $KM --model $MODEL --task qaqafull --items $BANKQ --split test "$@" >> $log 2>&1; }
  gq --out $QC --arms $ES $ES+C$CL.w.m0.25 --first-batches 2 || fail "qaqa check generation"
  gq --out $QO --arms $XARMS || fail "qaqa generation"
  for a in $ES $ES+C$CL.w.m0.25; do "${DC[@]}" $NH compare $(first $QC $(tok $a)) $QO/$(tok $a).json >> $log 2>&1 || fail "qaqa twice $a"; done
  for c in 0.25 0.5 1 2; do
    for s in $S $SP; do
      "${DC[@]}" $NC parts $QO/$(tok x_E2:E$EL.w.p$s+C$CL.w.m$c).json $QM/$(tok $ES+C$CL.w.m$c).json 1 1 C >> $log 2>&1 || fail "qaqa C part $s $c"
    done
  done
  if [ ! -e $HOST/${QW#$ROOT/}/judge_input.json ]; then
    "${J[@]}" python3 -I $NH qaqafull none export --task qaqafull --model $MODEL --items $BANKQ --split test \
        --gen-dir $QO --only-arms $XARMS --out $QW/judge_input_full570.json >> $log 2>&1 || fail "qaqa export"
    "${DC[@]}" $KF/keepfix_qaqa_pilot.py $QW/judge_input_full570.json $QW/judge_input.json >> $log 2>&1 || fail "qaqa pilot subset"
  fi
  [ -e $HOST/${QW#$ROOT/}/judge2/cells.jsonl ] || "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $QW --generation judge_input.json >> $log 2>&1 || fail "qaqa cells"
  # the main E at half strength (P + E(1.75) and + C), generated here from the main run (read only), judged on the pilot
  QH=$NS/gen_prompt/keepfix_main/qaqafull/$MODEL/test; QHC=$NS/gen_prompt/keepfix_main_chk/qaqafull/$MODEL/test
  QWH=$NS/judge/qaqafull/$MODEL/keepfix_main_pilot100
  HARMS="$EH"; for c in 0.25 0.5 1 2; do HARMS="$HARMS $EH+C$CL.w.m$c"; done
  gqm() { "${DQ[@]}" $PYM -I $NH qaqafull prompt gen --models $M --model $MODEL --task qaqafull --items $BANKQ --split test "$@" >> $log 2>&1; }
  gqm --out $QHC --arms $EH+C$CL.w.m0.25 --first-batches 2 || fail "qaqa half check generation"
  gqm --out $QH --arms $HARMS || fail "qaqa half generation"
  "${DC[@]}" $NH compare $(first $QHC $(tok $EH+C$CL.w.m0.25)) $QH/$(tok $EH+C$CL.w.m0.25).json >> $log 2>&1 || fail "qaqa half twice"
  "${DC[@]}" $NC parts $QH/$(tok $EH).json $QM/$(tok $ES).json $SH $S E >> $log 2>&1 || fail "qaqa half E part"
  for c in 0.25 0.5 1 2; do
    f=$QH/$(tok $EH+C$CL.w.m$c).json; mf=$QM/$(tok $ES+C$CL.w.m$c).json
    "${DC[@]}" $NC parts $f $mf $SH $S E >> $log 2>&1 && "${DC[@]}" $NC parts $f $mf $SH $S C >> $log 2>&1 || fail "qaqa half parts $c"
  done
  if [ ! -e $HOST/${QWH#$ROOT/}/judge_input.json ]; then
    "${J[@]}" python3 -I $NH qaqafull none export --task qaqafull --model $MODEL --items $BANKQ --split test \
        --gen-dir $QH --only-arms $HARMS --out $QWH/judge_input_full570.json >> $log 2>&1 || fail "qaqa half export"
    "${DC[@]}" $KF/keepfix_qaqa_pilot.py $QWH/judge_input_full570.json $QWH/judge_input.json >> $log 2>&1 || fail "qaqa half pilot subset"
  fi
  [ -e $HOST/${QWH#$ROOT/}/judge2/cells.jsonl ] || "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $QWH --generation judge_input.json >> $log 2>&1 || fail "qaqa half cells"
  say "qaqa generated, checked, exported (pilot 100; E_k and the main E at half strength)"
  jstart qaqa; jstart qaqah
  # ---- PopQA / TriviaQA tune (alias match, no judge)
  gf() { local task=$1 pos=$2 models=$3; shift 3
         "${DQ[@]}" $PYM -I $NH $task $pos gen --models $models --model $MODEL --task $task --items $NS/bank/${task}_bank_20261119.jsonl --split tune "$@" >> $log 2>&1; }
  FC=$NS/gen_prompt/keepfix_chk/popqa/$MODEL/tune
  gf popqa prompt $KM --out $FC --arms $ES --first-batches 2 || fail "fact check generation"
  for task in popqa triviaqa; do
    FO=$NS/gen_prompt/keepfix/$task/$MODEL/tune; FM=$NS/gen_prompt/keepfix_main/$task/$MODEL/tune
    gf $task prompt $KM --out $FO --arms $ES $EA || fail "fact E_k $task"
    gf $task prompt $M --out $FM --arms $ES $EH || fail "fact main E $task"
    gf $task none $M --out $FM --arms x_E2 || fail "fact P $task"
    for s in $S $SH; do
      "${DC[@]}" $NC parts $FM/$(tok x_E2:E$EL.w.p$s).json $NS/gen_prompt/$task/$MODEL/tune/x--E$EL-w-p$(echo $S | tr . -).json $s $S E >> $log 2>&1 || fail "fact main E part $task $s"
    done
  done
  "${DC[@]}" $NH compare $(first $FC $(tok $ES)) $NS/gen_prompt/keepfix/popqa/$MODEL/tune/$(tok $ES).json >> $log 2>&1 || fail "fact twice"
  "${DC[@]}" $KF/keepfix_table.py check-fact --run $ROOT/$R1 --arms $ES $EA >> $log 2>&1 || fail "fact write records"
  say "fact sets generated and checked"
  flock -u 9; exec 9>&-
  say "card released; waiting for the four judges"
  wait
  say "balance after: $(bash $L/balance.sh)"
  for t in kf kfx qaqa qaqah; do [ -e ${MK}_${t}_DONE ] || fail "judge $t not done"; done
  touch ${MK}_DONE; say "judged"
  exit 0
fi
# ============================================================================ judge
if [ $PH = judge ]; then
  say "balance before: $(bash $L/balance.sh)"
  for t in kf kfx qaqa qaqah; do jstart $t; done
  wait
  say "balance after: $(bash $L/balance.sh)"
  if [ -e ${MK}_kf_DONE ] && [ -e ${MK}_kfx_DONE ] && [ -e ${MK}_qaqa_DONE ] && [ -e ${MK}_qaqah_DONE ]; then touch ${MK}_DONE; else touch ${MK}_FAILED; fi
fi

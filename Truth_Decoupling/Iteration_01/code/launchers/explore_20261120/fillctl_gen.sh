#!/bin/bash
# 2026-11-23 23:4x, exploration, queue L section 9 + 10.1 (discussion/work_queue/2026-11-23_queue_l.md), executor A:
# the new pass TE = must-answer prompt + honest sentence (conf_E2) + E at the model's s_sel, written at "prompt", TruthfulQA.
# The SE command of n1s_model.sh (queue I) with ONE change: the arm prefix x_E2 -> conf_E2 (same entry
# scripts/review_checks/e44_positions_h22.py, same engine arguments: batch 8, max 48 new tokens, max 24 target tokens,
# seed 42, the h22 prompt file; layer from diag/n1_on_sentence.py plan, s_sel from newsets/h22/n1s_choice_<model>.json).
# PHASE pilot: the first batch (8 questions) of the tune split, n1s_check.py parts against SE's tune file (the write is
#   identical), diag/fill_only2.py gencheck. No export, no judging.
# PHASE tune|test: the whole split, the same two checks, export fillctl_<split>, judge cells
#   (eval_extra/judge_<split>_fillctl_<split>); the card is released. No judging here (fillctl_judge.sh).
# One model, one card, only through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock); cards 1 or 2 only; ours plus
# others at most 66560 MiB on the card. Markers FILLCTL_<MODEL>_<PHASE>_DONE / _FAILED; log fillctl_<model>_<phase>.log.
# usage (host): fillctl_gen.sh KEY GPU pilot|tune|test      KEY = llama q9 q27 g12 g27
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2; PH=$3
case "$PH" in pilot) SPL=tune ;; tune|test) SPL=$PH ;; *) echo "usage: fillctl_gen.sh KEY GPU pilot|tune|test"; exit 2 ;; esac
case "$GPU" in 1|2) ;; *) echo "cards 1 or 2 only"; exit 2 ;; esac
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
MK=$L/FILLCTL_${MODEL}_$PH; log=$L/fillctl_${MODEL}_$PH.log
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') fillctl $MODEL $PH: $*" | tee -a $L/fillctl_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_DONE
NEWP=$ROOT/Iteration_01/code/configs/prompts/e44_eval_extra_h22_v1.json
WR=$ROOT/Iteration_01/code/scripts/review_checks/e44_positions_h22.py
NC=$ROOT/Iteration_01/code/scripts/newsets/n1s_check.py
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B Iteration_01/diag/n1_on_sentence.py)
F2=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B Iteration_01/diag/fill_only2.py)
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
read -r EL S SH CL <<< "$("${RD[@]}" plan $MODEL 2>> $log)"
[ -n "$CL" ] || fail "plan"
SSEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["strength_dot"])' $HOST/Iteration_01/newsets/h22/n1s_choice_$MODEL.json) || fail "no tune choice"
[ -n "$SSEL" ] || fail "empty s_sel"
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
ARM=conf_E2:E$EL.w.p$SSEL; SEARM=x_E2:E$EL.w.p$SSEL
say "plan E$EL s_sel $SSEL (main $S), arm $ARM (SE arm $SEARM), split $SPL"
# ---- the card
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$GPU.lock
exec 9>"$LOCK" || fail "cannot open $LOCK"
until flock -n 9; do sleep 60; done
while true; do
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1 == g {print $2}')
  [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $GPU"
  [ $((USED + NEED)) -le 66560 ] && break
  say "GPU $GPU holds $USED MiB, $NEED more would exceed 66560; waiting"; sleep 60
done
say "start on GPU $GPU ($USED MiB in use)"
DT=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
COMMON=(--source-run $SRC --run-dir $ROOT/$run --e-on $E_ON --c-run-dir $ROOT/$crun --c-e-on $C_E_ON
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $NEWP)
G=$ROOT/$run/eval_extra/gen
F=$(tok $ARM)-at-prompt_$SPL.json; FSE=$(tok $SEARM)-at-prompt_$SPL.json
[ -s $HOST/$run/eval_extra/gen/$FSE ] || fail "no SE file $FSE"
if [ $PH = pilot ]; then
  "${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split tune --first-batches 1 --arms $ARM >> $log 2>&1 || fail "pilot generation"
  F=t8.$F
else
  "${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split $SPL --arms $ARM >> $log 2>&1 || fail "generation"
fi
[ -s $HOST/$run/eval_extra/gen/$F ] || fail "no output $F"
"${DC[@]}" $NC parts $G/$F $G/$FSE $SSEL $SSEL E >> $log 2>&1 || fail "write differs from SE's"
"${F2[@]}" gencheck $MODEL $SPL $G/$F >> $log 2>&1 || fail "gencheck"
say "generated $F; write identical to SE's; gencheck passed"
if [ $PH != pilot ]; then
  NAME=fillctl_$PH; W=$run/eval_extra/judge_${PH}_$NAME
  [ -e $HOST/$W ] && fail "work dir $W exists"
  "${DT[@]}" $PYM $WR prompt export "${COMMON[@]}" --split $PH --gen $F --name $NAME >> $log 2>&1 || fail "export"
  mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$PH/$NAME.json $HOST/$W/judge_input.json || fail "judge input"
  "${DT[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "cells"
  say "exported to $W"
fi
flock -u 9; exec 9>&-
say "card $GPU released"
touch ${MK}_DONE

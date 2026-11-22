#!/bin/bash
# 2026-11-22 21:4x, exploration, queue I (discussion/work_queue/2026-11-22_queue_i.md): the main table's E / E + C arms on
# the prompt x_E2 (P = plain prompt + honest sentence), written at "prompt", TruthfulQA. One model, one card (its lock).
# Layer, main strength, half strength and C layer from the main table's records (diag/n1_on_sentence.py plan).
# PHASE tune: checks (queue section 4 items 1-4; item 5 is the read program's step 0), the engine's two-write selfcheck
#   on the whole tune split (record *-h22-at-prompt.json, scripts/review_checks/e44_positions_h22.py), both strengths:
#   x_E2:E<L>.w.p<s> and x_E2:E<L>.w.p<s>+C<CL>.w.m<c>, c = 0.25 0.5 1 2; export n1s_tune.
# PHASE test: the strength chosen on tune (newsets/h22/n1s_choice_<model>.json, diag/n1_on_sentence.py choose), selfcheck
#   on the whole test split, its five arms, export n1s_test.
# Then the card is released and DeepSeek Pro judges in the background (reply caches of the h22c4 and posprompt2 work
# dirs merged first, off peak only). Markers N1S_<MODEL>_<PHASE>_GPU_DONE / _FAILED, N1S_<MODEL>_<PHASE>_JUDGE_DONE / _FAILED.
# usage: n1s_model.sh KEY GPU tune|test
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2; PH=$3
[ "$PH" = tune ] || [ "$PH" = test ] || { echo "usage: n1s_model.sh KEY GPU tune|test"; exit 2; }
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
MK=$L/N1S_${MODEL}_$PH; log=$L/n1s_${MODEL}_$PH.log
say() { echo "$(date '+%F %T') n1s $MODEL $PH: $*" >> $L/n1s_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_GPU_DONE
NEWP=$ROOT/Iteration_01/code/configs/prompts/e44_eval_extra_h22_v1.json
WR=$ROOT/Iteration_01/code/scripts/review_checks/e44_positions_h22.py
NC=$ROOT/Iteration_01/code/scripts/newsets/n1s_check.py
NH=$ROOT/Iteration_01/code/scripts/newsets/ns_h22.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B Iteration_01/diag/n1_on_sentence.py)
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
read -r EL S SH CL <<< "$("${RD[@]}" plan $MODEL 2>> $log)"
[ -n "$CL" ] || fail "plan"
if [ $PH = tune ]; then STRS="$S $SH"; else
  STRS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["strength_dot"])' $HOST/Iteration_01/newsets/h22/n1s_choice_$MODEL.json) || fail "no tune choice"
fi
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
ARMS=""; for s in $STRS; do ARMS="$ARMS x_E2:E$EL.w.p$s"; for c in 0.25 0.5 1 2; do ARMS="$ARMS x_E2:E$EL.w.p$s+C$CL.w.m$c"; done; done
say "plan E$EL main $S half $SH, C$CL; strengths $STRS"
# ---- the card
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$GPU.lock
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
DT=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
COMMON=(--source-run $SRC --run-dir $ROOT/$run --e-on $E_ON --c-run-dir $ROOT/$crun --c-e-on $C_E_ON
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $NEWP)
G=$ROOT/$run/eval_extra/gen
STK=$(echo $S | tr . -)
if [ $PH = tune ]; then
  # 1. the entry changes nothing: x_E2:C<CL>.w.m0.25 on the first 2 batches = the existing file
  "${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split tune --first-batches 2 --arms x_E2:C$CL.w.m0.25 >> $log 2>&1 || fail "check 1 generation"
  "${DC[@]}" $NH compare $G/t16.x_E2--C$CL-w-m0-25-at-prompt_tune.json $G/x_E2--C$CL-w-m0-25-at-prompt_tune.json >> $log 2>&1 || fail "check 1: differs"
  # 2-3. E at 0 = P; the same setting twice
  CK=$ROOT/Iteration_01/newsets/h22/n1s_check_$MODEL.json
  [ -s $HOST/${CK#$ROOT/} ] || "${DQ[@]}" $PYM -I $NC gpu --models $M --model $MODEL --el $EL --cl $CL --s $S \
      --base-file $G/../base_x_E2_tune.json --out $CK >> $log 2>&1 || fail "checks 2-3"
  say "checks 1-3 passed"
fi
SC=x_E2:E$EL.w.p$(echo $STRS | cut -d" " -f1)+C$CL.w.m0.25
"${DT[@]}" $PYM $WR prompt selfcheck "${COMMON[@]}" --split $PH --arms $SC >> $log 2>&1 || fail "selfcheck"
"${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split $PH --arms $ARMS >> $log 2>&1 || fail "generation"
GEN=""
for a in $ARMS; do
  f=$(tok $a)-at-prompt_$PH.json; GEN="$GEN $f"
  s=$(echo $a | sed -e 's/.*w\.p//' -e 's/+.*//')
  if [[ $a == *+C* ]]; then
    c=$(echo $a | sed 's/.*w\.m//')
    MAINF=$G/x--E$EL-w-p$STK-and-C$CL-w-m$(echo $c | tr . -)-at-prompt_$PH.json
    "${DC[@]}" $NC parts $G/$f $MAINF $s $S E >> $log 2>&1 && "${DC[@]}" $NC parts $G/$f $MAINF $s $S C >> $log 2>&1 || fail "check 4 $a"
  else
    "${DC[@]}" $NC parts $G/$f $G/x--E$EL-w-p$STK-at-prompt_$PH.json $s $S E >> $log 2>&1 || fail "check 4 $a"
  fi
done
if [ $PH = tune ]; then
  "${DC[@]}" $NC cross $CK $G/$(tok x_E2:E$EL.w.p$S)-at-prompt_tune.json $G/$(tok x_E2:E$EL.w.p$S+C$CL.w.m0.25)-at-prompt_tune.json >> $log 2>&1 || fail "check 3 across runs"
fi
say "generated and checked"
NAME=n1s_$PH; W=$run/eval_extra/judge_${PH}_$NAME
if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then
  "${DT[@]}" $PYM $WR prompt export "${COMMON[@]}" --split $PH --gen $GEN --name $NAME >> $log 2>&1 || fail "export"
  mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$PH/$NAME.json $HOST/$W/judge_input.json || fail "judge input"
  "${DT[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "cells"
fi
flock -u 9; exec 9>&-
touch ${MK}_GPU_DONE
say "exported; card released"
(
  rm -f ${MK}_JUDGE_DONE ${MK}_JUDGE_FAILED
  JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
      -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  for old in judge_${PH}_h22c4_$PH judge_${PH}_posprompt2l${EL}_$PH; do
    [ -d $HOST/$run/eval_extra/$old/judge2/cache ] || continue
    "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $ROOT/$run/eval_extra/$old/judge2/cache >> $log 2>&1 || { touch ${MK}_JUDGE_FAILED; exit 1; }
  done
  "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json >> $log 2>&1 || { touch ${MK}_JUDGE_FAILED; exit 1; }
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
  if "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro >> $log 2>&1 &&
     "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro > $HOST/$W/single_pro_stdout.md 2>> $log; then
    say "judged"; touch ${MK}_JUDGE_DONE
  else
    say "FAILED: judge"; touch ${MK}_JUDGE_FAILED; exit 1
  fi
) > /dev/null 2>&1 < /dev/null &

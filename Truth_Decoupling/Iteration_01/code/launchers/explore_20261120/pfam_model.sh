#!/bin/bash
# 2026-11-22 23:3x, exploration, queue I section 7.6: three more no-write prompts on TruthfulQA (x_H, x_E2_H, x_E1_E2;
# configs/prompts/e44_eval_extra_pfam_v1.json, H added in memory by scripts/review_checks/e44_pfam.py), tune and test,
# one model, one card. Check: x_E2 regenerated through this flow on the first 2 batches of tune = the existing
# base_x_E2_tune.json (16 rows). Export pfam_<split>, DeepSeek Pro off peak (reply cache of the h22c4 work dirs merged).
# Markers PFAM_<MODEL>_GPU_DONE / _FAILED, PFAM_<MODEL>_<tune|test>_JUDGE_DONE / _FAILED.  usage: pfam_model.sh KEY GPU
# (the model block below is the one of n1s2_model.sh)
#
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
[ -n "$GPU" ] || { echo "usage: pfam_model.sh KEY GPU"; exit 2; }
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
MK=$L/PFAM_$MODEL; log=$L/pfam_$MODEL.log
say() { echo "$(date '+%F %T') pfam $MODEL: $*" >> $L/n1s_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_GPU_DONE
PF=$ROOT/Iteration_01/code/configs/prompts/e44_eval_extra_pfam_v1.json
WR=$ROOT/Iteration_01/code/scripts/review_checks/e44_pfam.py
NH=$ROOT/Iteration_01/code/scripts/newsets/ns_h22.py
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
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
COMMON=(--source-run $SRC --run-dir $ROOT/$run --e-on $E_ON --c-run-dir $ROOT/$crun --c-e-on $C_E_ON
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $PF)
E=$ROOT/$run/eval_extra
"${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split tune --first-batches 2 --bases x_E2 >> $log 2>&1 || fail "check generation"
"${DC[@]}" $NH compare $E/t16.base_x_E2_tune.json $E/base_x_E2_tune.json >> $log 2>&1 || fail "check: x_E2 differs"
for sp in tune test; do
  "${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split $sp --bases x_H x_E2_H x_E1_E2 >> $log 2>&1 || fail "generation $sp"
  NAME=pfam_$sp; W=$run/eval_extra/judge_${sp}_$NAME
  if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then
    "${DT[@]}" $PYM $WR prompt export "${COMMON[@]}" --split $sp --gen --prompts-out x_H x_E2_H x_E1_E2 --name $NAME >> $log 2>&1 || fail "export $sp"
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$sp/$NAME.json $HOST/$W/judge_input.json || fail "judge input $sp"
    "${DT[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "cells $sp"
  fi
done
flock -u 9; exec 9>&-
touch ${MK}_GPU_DONE
say "generated and exported; card released"
for sp in tune test; do
  ( w=$ROOT/$run/eval_extra/judge_${sp}_pfam_$sp; hw=$HOST/$run/eval_extra/judge_${sp}_pfam_$sp; old=$ROOT/$run/eval_extra/judge_${sp}_h22c4_$sp
    JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
        -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
    rm -f ${MK}_${sp}_JUDGE_DONE ${MK}_${sp}_JUDGE_FAILED
    "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $w/judge2/cache $old/judge2/cache >> $log 2>&1 || { touch ${MK}_${sp}_JUDGE_FAILED; exit 1; }
    "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $w/judge2/cache/conflicts.json >> $log 2>&1 || { touch ${MK}_${sp}_JUDGE_FAILED; exit 1; }
    secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
    if [ "$secs" -lt 900 ]; then say "judge $sp not started: peak hours or too close"; touch ${MK}_${sp}_JUDGE_FAILED; exit 1; fi
    if "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $w --judge pro >> $log 2>&1 &&
       "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $w --judge pro > $hw/single_pro_stdout.md 2>> $log; then
      say "judged $sp"; touch ${MK}_${sp}_JUDGE_DONE
    else
      say "FAILED: judge $sp"; touch ${MK}_${sp}_JUDGE_FAILED; exit 1
    fi ) > /dev/null 2>&1 < /dev/null &
done

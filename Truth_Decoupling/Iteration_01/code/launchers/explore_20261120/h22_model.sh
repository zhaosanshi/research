#!/bin/bash
# 2026-11-22 20:0x, exploration, queue H (discussion/work_queue/2026-11-22_queue_h.md): the missing cell of the 2x2,
# cell 4 = plain prompt + honest sentence (x_E2, configs/prompts/e44_eval_extra_h22_v1.json) + C only at 0.25, written
# at the core rule "prompt", C at the layer of cell 3. One model, one card (held through the lock file of that card).
# Steps (any failure stops the model, marker H22_<MODEL>_FAILED):
#   1. checks on 16 questions (scripts/newsets/ns_h22.py check): C at amount 0 = no write (TruthfulQA tune and (QA)^2),
#      = the promptdiag no-write x_E2 answers where they exist, the same C 0.25 twice identical;
#   2. the entries change nothing else: x:C<CL>.w.m0.25 regenerated through ns_h22.py on (QA)^2 and through
#      e44_positions.py with the new prompt file on TruthfulQA tune (first 2 batches) = the existing files;
#   3. cell 4 on (QA)^2 (570 questions, gen dir newsets/gen_prompt/h22/qaqafull/<model>/test), export, cells
#      (judge work dir newsets/judge/qaqafull/<model>/test_full570_h22);
#   4. cell 4 on TruthfulQA tune and test (eval_extra/gen/x_E2--C<CL>-w-m0-25-at-prompt_<split>.json, the no-write x_E2
#      answers eval_extra/base_x_E2_<split>.json), export h22c4_<split>, work dir eval_extra/judge_<split>_h22c4_<split>;
#   5. the C write of every cell-4 file = the C part of cell 3's file (ns_h22.py parts);
#   then the card is released (marker H22_<MODEL>_GPU_DONE) and the three judges run in the background: reply cache
#   of an existing work dir merged first, conflicts stop, DeepSeek Pro (non-thinking, the flow's own config), only off
#   peak (the clock guard of qaqa_run.sh); markers H22_<MODEL>_JUDGE_<qaqa|tune|test>_DONE / _FAILED.
# usage: h22_model.sh KEY GPU      KEY = llama | q9 | q27 | g12 | g27
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2
[ -n "$GPU" ] || { echo "usage: h22_model.sh KEY GPU"; exit 2; }
TF=$ROOT/envs/tf519/bin/python
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3; PDG=""
case $KEY in
  llama) MODEL=llama3.1-8b; CODE=$ROOT/code_e45_20261112; PYM=python3; run=results_e44_factorial_das/llama3.1-8b_side_E2
         crun=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; EL=16; S=3.5; CL=19; SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
         PDG=results_newmodels_ff_20261112/promptdiag/llama3.1-8b/frozen_tune.json; NEED=20000 ;;
  q9)    MODEL=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.5-9b_E2
         crun=$E44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; EL=15; S=2; CL=16; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b
         PDG=results_newmodels_ff_20261112/promptdiag/qwen3.5-9b/frozen_tune.json; NEED=22000 ;;
  q27)   MODEL=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
         crun=$run; E_ON=E2; C_E_ON=E2; EL=28; S=1.75; CL=33; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm; NEED=62000 ;;
  g12)   MODEL=gemma3-12b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/12b/e44/gemma3-12b
         crun=$run; E_ON=E2; C_E_ON=E2; EL=24; S=2.5; CL=25; SRC=$ROOT/$G3/12b/source/gemma3-12b; NEED=30000 ;;
  g27)   MODEL=gemma3-27b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
         crun=$run; E_ON=E2; C_E_ON=E2; EL=28; S=1; CL=31; SRC=$ROOT/$G3/27b/source/gemma3-27b; NEED=62000 ;;
  *) echo "unknown key $KEY"; exit 2 ;;
esac
MK=$L/H22_$MODEL; log=$L/h22_$MODEL.log
say() { echo "$(date '+%F %T') h22 $MODEL: $*" >> $L/h22_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_GPU_DONE
NEWP=$ROOT/Iteration_01/code/configs/prompts/e44_eval_extra_h22_v1.json
NH=$ROOT/Iteration_01/code/scripts/newsets/ns_h22.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
FULLB=$ROOT/Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
CHK=$NS/h22; HCHK=$HNS/h22
[ -s $HOST/Iteration_01/code/configs/prompts/e44_eval_extra_h22_v1.json ] && [ -s $HOST/Iteration_01/code/scripts/newsets/ns_h22.py ] || fail "new files missing"
# ---- the card: lock (retry every minute), then the memory rule
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
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
DT=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I $NH)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
   -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
ARM=x_E2:C$CL.w.m0.25; TOK=x_E2--C$CL-w-m0-25
OLDC=x:C$CL.w.m0.25; OTOK=x--C$CL-w-m0-25
THREE=x--E$EL-w-p$(echo $S | tr . -)-and-C$CL-w-m0-25
COMMON=(--source-run $SRC --run-dir $ROOT/$run --e-on $E_ON --c-run-dir $ROOT/$crun --c-e-on $C_E_ON
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $NEWP)
# ---- 1. checks on 16 questions
mkdir -p $HCHK
if [ ! -s $HCHK/check_$MODEL.json ]; then
  "${DQ[@]}" $PYM -I $NH check --models $M --model $MODEL --bank $FULLB --cl $CL ${PDG:+--promptdiag $ROOT/$PDG} \
      --out $CHK/check_$MODEL.json >> $log 2>&1 || fail "check (zero amount / twice / promptdiag)"
fi
say "check passed"
# ---- 2. the entries change nothing else
QCHK=$NS/gen_prompt/h22chk/qaqafull/$MODEL/test
"${DQ[@]}" $PYM -I $NH qaqafull prompt gen --models $M --model $MODEL --task qaqafull --items $FULLB --split test \
    --out $QCHK --arms $OLDC --first-batches 2 >> $log 2>&1 || fail "entry check generation, qaqa"
sub=$(ls $HOST/Iteration_01/newsets/gen_prompt/h22chk/qaqafull/$MODEL/test/s*.$OTOK.json | head -1)
[ -n "$sub" ] || fail "entry check file, qaqa"
"${DC[@]}" compare $ROOT/${sub#$HOST/} $NS/gen_prompt/qaqafull/$MODEL/test/$OTOK.json >> $log 2>&1 || fail "entry check, qaqa: differs"
"${DT[@]}" $PYM scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --split tune --first-batches 2 \
    --arms $OLDC >> $log 2>&1 || fail "entry check generation, TruthfulQA"
"${DC[@]}" compare $ROOT/$run/eval_extra/gen/t16.$OTOK-at-prompt_tune.json $ROOT/$run/eval_extra/gen/$OTOK-at-prompt_tune.json \
    >> $log 2>&1 || fail "entry check, TruthfulQA: differs"
say "entry checks passed"
# ---- 3. cell 4 on (QA)^2
QOUT=$NS/gen_prompt/h22/qaqafull/$MODEL/test; QW=$NS/judge/qaqafull/$MODEL/test_full570_h22; HQW=$HNS/judge/qaqafull/$MODEL/test_full570_h22
"${DQ[@]}" $PYM -I $NH qaqafull prompt gen --models $M --model $MODEL --task qaqafull --items $FULLB --split test \
    --out $QOUT --arms $ARM >> $log 2>&1 || fail "generation, qaqa"
"${DC[@]}" parts $QOUT/$TOK.json $NS/gen_prompt/qaqafull/$MODEL/test/$THREE.json >> $log 2>&1 || fail "C write of cell 4 differs from cell 3, qaqa"
if [ ! -e $HQW/judge_input.json ]; then
  "${J[@]}" python3 -I $NH qaqafull none export --task qaqafull --model $MODEL --items $FULLB --split test --gen-dir $QOUT \
      --only-arms $ARM --out $QW/judge_input.json >> $log 2>&1 || fail "export, qaqa"
fi
[ -e $HQW/judge2/cells.jsonl ] || "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $QW --generation judge_input.json >> $log 2>&1 || fail "cells, qaqa"
say "qaqa generated and exported"
# ---- 4. cell 4 on TruthfulQA tune and test
for sp in tune test; do
  "${DT[@]}" $PYM scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --split $sp --arms $ARM >> $log 2>&1 || fail "generation, TruthfulQA $sp"
  "${DC[@]}" parts $ROOT/$run/eval_extra/gen/$TOK-at-prompt_$sp.json $ROOT/$run/eval_extra/gen/$THREE-at-prompt_$sp.json >> $log 2>&1 || fail "C write of cell 4 differs from cell 3, TruthfulQA $sp"
  NAME=h22c4_$sp; W=$run/eval_extra/judge_${sp}_$NAME
  if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then
    "${DT[@]}" $PYM scripts/review_checks/e44_positions.py prompt export "${COMMON[@]}" --split $sp --gen $TOK-at-prompt_$sp.json --name $NAME >> $log 2>&1 || fail "export, TruthfulQA $sp"
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$sp/$NAME.json $HOST/$W/judge_input.json || fail "judge input, TruthfulQA $sp"
    "${DT[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "cells, TruthfulQA $sp"
  fi
done
say "TruthfulQA generated and exported; card released"
flock -u 9; exec 9>&-
touch ${MK}_GPU_DONE
# ---- judges in the background
judge_one() {  # judge_one TAG WORK_IN_CONTAINER HOST_WORK OLD_HOST_WORK CODE_COPY (whose judge scripts the flow used)
  local tag=$1 w=$2 hw=$3 old=$4 code=$5 secs
  local J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
           -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$code
           -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $code truth_dec)
  rm -f ${MK}_JUDGE_${tag}_DONE ${MK}_JUDGE_${tag}_FAILED
  [ -s $hw/judge2/single_pro.jsonl ] && { touch ${MK}_JUDGE_${tag}_DONE; return 0; }
  mkdir -p $hw/judge2/cache
  if [ -d $old/judge2/cache ]; then
    "${J[@]}" python3 scripts/review_checks/judge2_cache_merge.py $w/judge2/cache ${old/#$HOST/$ROOT}/judge2/cache >> $log 2>&1 || { touch ${MK}_JUDGE_${tag}_FAILED; return 1; }
    "${J[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $w/judge2/cache/conflicts.json >> $log 2>&1 || { touch ${MK}_JUDGE_${tag}_FAILED; return 1; }
  fi
  secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
  if [ "$secs" -lt 900 ]; then say "judge $tag not started: peak hours or too close"; touch ${MK}_JUDGE_${tag}_FAILED; return 1; fi
  if "${J[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $w --judge pro >> $log 2>&1 &&
     "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $w --judge pro > $hw/single_pro_stdout.md 2>> $log; then
    say "judged $tag"; touch ${MK}_JUDGE_${tag}_DONE
  else
    say "FAILED: judge $tag"; touch ${MK}_JUDGE_${tag}_FAILED; return 1
  fi
}
( judge_one qaqa $QW $HQW $HNS/judge/qaqafull/$MODEL/test_full570 $ROOT/Iteration_01/code ) > /dev/null 2>&1 < /dev/null &
for sp in tune test; do
  ( judge_one $sp $ROOT/$run/eval_extra/judge_${sp}_h22c4_$sp $HOST/$run/eval_extra/judge_${sp}_h22c4_$sp \
      $HOST/$run/eval_extra/judge_${sp}_pospromptc_$sp $CODE ) > /dev/null 2>&1 < /dev/null &
done
say "judges started"

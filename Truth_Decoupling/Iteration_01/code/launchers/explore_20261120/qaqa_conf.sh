#!/bin/bash
# 2026-11-22 02:4x, exploration: on (QA)^2, the model's confidence in its own prompt-only answers (conf_E2: must
# answer + honest sentence, no write), for the comparator "no write, keep the most confident N answers" (the P1 of
# summary section 8.2, now on this set). Mean token log-probability by newsets_gen.py conf (unchanged scorer;
# the 98% greedy-reproduction check is not relaxed: scoring batch 8 first, batch 1 only if that misses the check).
# Prediction written before this ran: discussion/work_queue/2026-11-22_queue.md section 2.8.
# No judge call (the answers are already judged in the frozen run). Output newsets/conf/<TASK>/<MODEL>/test/conf_E2.json.
# usage: qaqa_conf.sh TASK BANK_FILE_NAME MODEL GPUS     e.g. qaqa_conf.sh qaqafull qaqa_full_bank_20261122.jsonl llama3.1-8b 2
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
NT=$CODE/scripts/newsets/ns_task.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
TASK=$1; BANKF=$2; MODEL=$3; GPUS=$4
[ -n "$GPUS" ] || { echo "usage: qaqa_conf.sh TASK BANK_FILE_NAME MODEL GPUS"; exit 2; }
GEN=$NS/gen_prompt/$TASK/$MODEL/test/conf_E2.json
OUT=$NS/conf/$TASK/$MODEL/test/conf_E2.json
MK=$L/QAQA_CONF_${TASK}_${MODEL}
log=$L/qaqaconf_${TASK}_$MODEL.log
say() { echo "$(date '+%F %T') qaqa conf $TASK $MODEL: $*" >> $L/qaqa_all.log; }
rm -f ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
[ -s $HNS/gen_prompt/$TASK/$MODEL/test/conf_E2.json ] || fail "no prompt-only generation"
if [ -e $HNS/conf/$TASK/$MODEL/test/conf_E2.json ]; then say "already there"; touch ${MK}_DONE; exit 0; fi
plan=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || fail "model not in the config"
PY=$plan
[ -n "$PY" ] || fail "model not in the config"
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
say "start on GPU $GPUS"
if "${DX[@]}" $PY -I $NT $TASK none conf --task $TASK --models $M --model $MODEL --items $NS/bank/$BANKF \
     --gen-file $GEN --out $OUT >> $log 2>&1; then
  say "done (scoring batch 8)"
else
  say "missed a check at scoring batch 8: once more at scoring batch 1"
  "${DX[@]}" $PY -I $NT $TASK none conf --task $TASK --models $M --model $MODEL --items $NS/bank/$BANKF \
     --gen-file $GEN --out $OUT --score-batch 1 >> $log 2>&1 || fail "confidence scoring"
  say "done (scoring batch 1)"
fi
touch ${MK}_DONE

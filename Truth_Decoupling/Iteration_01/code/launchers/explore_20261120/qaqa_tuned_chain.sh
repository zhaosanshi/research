#!/bin/bash
# 2026-11-22 02:1x, exploration: (QA)^2 with the strengths chosen on its own tune split, from the tune verdicts to the
# table (rule: discussion/work_queue/2026-11-22_queue.md section 2.8, written before any tune verdict of the sweep).
#   1. wait for the tune judges of the five models (qaqa_tune.sh);
#   2. choose every setting on the 100 tune questions (diag/qaqa_pilot_table.py --tune --write-choice);
#   3. on the full bank generate and judge the chosen arms that the frozen run does not have (work dir
#      test_full570_tuned; GPU 0: Gemma-3-12B, Gemma-3-27B, Qwen3.8-27B; GPU 1: Llama-3.1-8B, Qwen3.5-9B);
#   4. wait for those judges and for the frozen run's judges, then write the tables:
#      out_qaqa_tune_20261122.txt (the choice), out_qaqa_tuned_470_20261122.txt (chosen settings on the 470 questions
#      outside the tune split), out_qaqa_frozen_470_20261122.txt and out_qaqa_frozen_570_20261122.txt (TruthfulQA settings).
# Marker QAQA_TUNED_CHAIN_DONE or _FAILED.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
NT=$CODE/scripts/newsets/ns_task.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
MODELS="llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b"
FULLB=Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl
CHOICE=Iteration_01/newsets/splits/qaqa_tune_choice_20261122.json
TUNEIDS=Iteration_01/newsets/splits/qaqa_pilot_20261122.json
D=Iteration_01/diag
TB=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -w $ROOT truth_dec python3 -B Iteration_01/diag/qaqa_pilot_table.py)
say() { echo "$(date '+%F %T') qaqa tuned chain: $*" >> $L/qaqa_all.log; }
rm -f $L/QAQA_TUNED_CHAIN_DONE $L/QAQA_TUNED_CHAIN_FAILED
fail() { say "FAILED: $*"; touch $L/QAQA_TUNED_CHAIN_FAILED; exit 1; }
for m in $MODELS; do
  until [ -e $L/QAQA_TUNE_${m}_DONE ]; do [ ! -e $L/QAQA_TUNE_${m}_FAILED ] || fail "tune of $m"; sleep 30; done
done
until [ -e $L/QAQA_TUNE_1122_GEN_DONE ]; do [ ! -e $L/QAQA_TUNE_1122_GEN_FAILED ] || fail "tune generation"; sleep 30; done
if [ ! -e $CHOICE ]; then
  "${TB[@]}" --tune --work test_tune100 --write-choice $CHOICE $MODELS > $D/out_qaqa_tune_20261122.txt 2>&1 || fail "choice on tune"
fi
say "chosen on tune"
"${TB[@]}" --choice $CHOICE --list-arms --task qaqafull --bank $FULLB $MODELS > $L/qaqa_tuned_arms.txt 2>> $L/qaqa_tuned.log || fail "list of missing arms"
[ "$(wc -l < $L/qaqa_tuned_arms.txt)" = 5 ] || fail "list of missing arms is not five lines"
tuned() {  # tuned MODEL GPUS: generate and judge this model's missing arms on the full bank
  local MODEL=$1 GPUS=$2 arms plan PY CL
  arms=$(grep "^$MODEL " $L/qaqa_tuned_arms.txt | cut -d" " -f2-)
  rm -f $L/QAQA_TUNED_${MODEL}_DONE $L/QAQA_TUNED_${MODEL}_FAILED
  if [ -z "$arms" ]; then say "$MODEL: the chosen arms are all there already"; touch $L/QAQA_TUNED_${MODEL}_DONE; return 0; fi
  local W=$NS/judge/qaqafull/$MODEL/test_full570_tuned HW=$HNS/judge/qaqafull/$MODEL/test_full570_tuned
  local OUT=$NS/gen_prompt/qaqafull/$MODEL/test log=$L/qaqatuned_$MODEL.log
  plan=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || { touch $L/QAQA_TUNED_${MODEL}_FAILED; return 1; }
  read -r PY CL <<< "$plan"
  [ -n "$PY" ] || { touch $L/QAQA_TUNED_${MODEL}_FAILED; return 1; }
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
      -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
  local J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
      -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  say "$MODEL on GPU $GPUS: $arms"
  "${DX[@]}" $PY -I $NT qaqafull prompt gen --models $M --model $MODEL --task qaqafull --items $ROOT/$FULLB --split test --out $OUT \
      --arms $arms >> $log 2>&1 || { say "FAILED: generation of $MODEL"; touch $L/QAQA_TUNED_${MODEL}_FAILED; return 1; }
  if [ ! -e $HW/judge_input.json ]; then
    "${J[@]}" python3 -I $NT qaqafull none export --task qaqafull --model $MODEL --items $ROOT/$FULLB --split test --gen-dir $OUT \
        --only-arms $arms --out $W/judge_input.json >> $log 2>&1 || { say "FAILED: export of $MODEL"; touch $L/QAQA_TUNED_${MODEL}_FAILED; return 1; }
  fi
  if [ ! -e $HW/judge2/cells.jsonl ]; then
    "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation judge_input.json >> $log 2>&1 || { say "FAILED: cells of $MODEL"; touch $L/QAQA_TUNED_${MODEL}_FAILED; return 1; }
  fi
  ( if "${J[@]}" timeout -k 60 --signal=INT 21600 python3 scripts/review_checks/judge_two_axis_run.py llm $W --judge pro >> $log 2>&1 &&
       "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W --judge pro > $HW/single_pro_stdout.md 2>> $log; then
      touch $L/QAQA_TUNED_${MODEL}_DONE
    else
      touch $L/QAQA_TUNED_${MODEL}_FAILED
    fi ) > /dev/null 2>&1 < /dev/null &
}
if python3 -c '
import time, sys
lt = time.localtime(); m = lt.tm_hour * 60 + lt.tm_min
sys.exit(1 if lt.tm_wday < 5 and (8 * 60 + 30 <= m < 12 * 60 or 13 * 60 + 30 <= m < 18 * 60) else 0)'; then :; else fail "peak hours: the judge of the chosen arms is not started"; fi
( tuned gemma3-12b 0; tuned gemma3-27b 0; tuned qwen3.8-27b 0 ) &
( tuned llama3.1-8b 1; tuned qwen3.5-9b 1 ) &
wait
say "generation of the chosen arms done, waiting for the judges"
for m in $MODELS; do
  until [ -e $L/QAQA_TUNED_${m}_DONE ]; do [ ! -e $L/QAQA_TUNED_${m}_FAILED ] || fail "chosen arms of $m"; sleep 30; done
  until [ -e $L/QAQA_qaqafull_${m}_DONE ]; do [ ! -e $L/QAQA_qaqafull_${m}_FAILED ] || fail "frozen run of $m"; sleep 30; done
done
"${TB[@]}" --choice $CHOICE --skip-ids $TUNEIDS --task qaqafull --bank $FULLB --work test_full570,test_full570_tuned $MODELS > $D/out_qaqa_tuned_470_20261122.txt 2>&1 || fail "table of the chosen settings"
"${TB[@]}" --skip-ids $TUNEIDS --task qaqafull --bank $FULLB --work test_full570 $MODELS > $D/out_qaqa_frozen_470_20261122.txt 2>&1 || fail "table of the frozen settings, 470"
"${TB[@]}" --task qaqafull --bank $FULLB --work test_full570 $MODELS > $D/out_qaqa_frozen_570_20261122.txt 2>&1 || fail "table of the frozen settings, 570"
say "tables written"
touch $L/QAQA_TUNED_CHAIN_DONE

#!/bin/bash
# 2026-11-22 02:4x: the final (QA)^2 tables, after qaqa_tuned_chain.sh (verdicts of the frozen run and of the chosen
# arms) and qaqa_conf_1122.sh (confidence of the prompt-only answers) have both ended. All tables are printed again so
# that every one of them carries the confidence comparator for all five models:
#   out_qaqa_frozen_570_20261122.txt   every setting from TruthfulQA, all 570 questions
#   out_qaqa_frozen_470_20261122.txt   the same on the 470 questions outside the tune split
#   out_qaqa_fixedE_470_20261122.txt   E at the main-table strength, caps and single C chosen on the tune split
#   out_qaqa_tuned_470_20261122.txt    E strength, caps and single C all chosen on the tune split
# Rules and predictions: discussion/work_queue/2026-11-22_queue.md section 2.8. Marker QAQA_TABLES_1122_DONE / _FAILED.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
D=Iteration_01/diag
MODELS="llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b"
FULLB=Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl
TUNEIDS=Iteration_01/newsets/splits/qaqa_pilot_20261122.json
CH=Iteration_01/newsets/splits/qaqa_tune_choice_20261122.json
CHF=Iteration_01/newsets/splits/qaqa_tune_choice_fixedE_20261122.json
TB=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -w $ROOT truth_dec python3 -B Iteration_01/diag/qaqa_pilot_table.py)
rm -f $L/QAQA_TABLES_1122_DONE $L/QAQA_TABLES_1122_FAILED
fail() { echo "$(date '+%F %T') qaqa tables: FAILED: $*" >> $L/qaqa_all.log; touch $L/QAQA_TABLES_1122_FAILED; exit 1; }
until [ -e $L/QAQA_TUNED_CHAIN_DONE ]; do [ ! -e $L/QAQA_TUNED_CHAIN_FAILED ] || fail "the chain failed"; sleep 30; done
until [ -e $L/QAQA_CONF_1122_DONE ]; do [ ! -e $L/QAQA_CONF_1122_FAILED ] || fail "the confidence scoring failed"; sleep 30; done
W=test_full570,test_full570_tuned
"${TB[@]}" --task qaqafull --bank $FULLB --work test_full570 $MODELS > $D/out_qaqa_frozen_570_20261122.txt 2>&1 || fail "frozen 570"
"${TB[@]}" --skip-ids $TUNEIDS --task qaqafull --bank $FULLB --work test_full570 $MODELS > $D/out_qaqa_frozen_470_20261122.txt 2>&1 || fail "frozen 470"
"${TB[@]}" --choice $CHF --skip-ids $TUNEIDS --task qaqafull --bank $FULLB --work $W $MODELS > $D/out_qaqa_fixedE_470_20261122.txt 2>&1 || fail "fixed E 470"
"${TB[@]}" --choice $CH --skip-ids $TUNEIDS --task qaqafull --bank $FULLB --work $W $MODELS > $D/out_qaqa_tuned_470_20261122.txt 2>&1 || fail "tuned 470"
echo "$(date '+%F %T') qaqa tables: written" >> $L/qaqa_all.log
touch $L/QAQA_TABLES_1122_DONE

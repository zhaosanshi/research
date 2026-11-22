#!/bin/bash
# 2026-11-22 09:4x (Opus 5.5), exploration: the second reading of the ITI baseline for one model, after its main
# reading (iti_chain.sh) has ended: the best tune point of the whole grid whatever its MMLU change (the reading most
# favourable to ITI; summary section 14, rules in discussion/work_queue/2026-11-22_queue.md section 4), and GSM8K at
# both choices. Unattended:
#   1. wait for ITI_CHAIN_<MODEL>_<FIT>_<SIGMA>_DONE (stop on _FAILED);
#   2. tune answers for every grid point (iti_grid.sh with GEN_ONLY=1; existing files are kept);
#   3. every tune answer file that is in no judged work dir yet is judged (iti_table.py unjudged), one work dir per K
#      (judge_tune_iti_<FIT><_sp>_k<K>z_tune), in parallel;
#   4. iti_table.py tune --all --write-choice (choice_all[_sp].json; an existing choice file is kept);
#   5. the chosen point's MMLU pass if it has none (the grid's early stop may have skipped it);
#   6. its test answers, judged (work dir judge_test_iti_<FIT><_sp>_all_test), and the test table;
#   7. GSM8K under ITI at the main choice and at this one (iti_baseline.py gsm8k, on two cards in parallel), the
#      GSM8K table (diag/iti_gsm8k.py, in the container).
# Tables: Iteration_01/diag/out_iti_<MODEL>_<FIT><_sp>_{budget,tune_all,test_all,gsm8k}_20261122.txt.
# usage: iti_free.sh MODEL FIT SIGMA GPU "K ..." "ALPHA ..." LIKE_NAME GSM_GPU_A GSM_GPU_B [GSM8K_TOKENS]
#   e.g. iti_free.sh gemma3-12b train_seed42 endq 0 "16 32 48 64" "2.5 5 7.5 10 15 20 25 30" pospromptl24 0 2
# Markers: ITI_FREE_<MODEL>_<FIT>_<SIGMA>_DONE / _FAILED. Log lines in iti_all.log.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
D=Iteration_01/diag
IT=$ROOT/Iteration_01/code/scripts/newsets/iti_baseline.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
MODEL=$1; FIT=$2; SIGMA=$3; GPU=$4; KS=$5; ALPHAS=$6; LIKE=$7; GA=$8; GB=$9; TOK=${10:-512}
[ -n "$GB" ] || { echo "usage: iti_free.sh MODEL FIT SIGMA GPU \"K ...\" \"ALPHA ...\" LIKE_NAME GSM_GPU_A GSM_GPU_B [GSM8K_TOKENS]"; exit 2; }
case "$SIGMA" in endq) SUF="" ;; pairs) SUF="_sp" ;; *) echo "SIGMA is endq or pairs"; exit 2 ;; esac
B=Iteration_01/baselines/iti/$MODEL
MK=$L/ITI_FREE_${MODEL}_${FIT}_${SIGMA}
log=$L/iti_free_${MODEL}_${FIT}_$SIGMA.log
say() { echo "$(date '+%F %T') iti free $MODEL $FIT $SIGMA: $*" >> $L/iti_all.log; }
rm -f ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
CHM=$L/ITI_CHAIN_${MODEL}_${FIT}_${SIGMA}
until [ -e ${CHM}_DONE ]; do [ ! -e ${CHM}_FAILED ] || fail "the main reading failed"; sleep 60; done
PY=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)
[ -n "$PY" ] || fail "model not in the config"
DX() { local g=$1; shift; docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$g -w $ROOT truth_dec "$@"; }
TB=(timeout 900 python3 -B $D/iti_table.py)
say "start: tune answers for the whole grid on GPU $GPU"
GEN_ONLY=1 bash $L/iti_grid.sh $MODEL $GPU $FIT $SIGMA "$KS" "$ALPHAS" gall >> $log 2>&1 || fail "the generation-only grid"
todo=$("${TB[@]}" unjudged $MODEL $FIT --sigma $SIGMA 2>> $log) || fail "no list of unjudged tune answers"
pids=()
for K in $KS; do
  files=""
  while read -r k stem; do
    [ -n "$stem" ] && [ "$k" = "$K" ] || continue
    files="$files $B/$FIT/gen/tune_$stem.json"
  done <<< "$todo"
  [ -n "$files" ] || continue
  bash $L/iti_judge.sh $MODEL tune iti_${FIT}${SUF}_k${K}z $LIKE $files >> $log 2>&1 &
  pids+=("$!")
done
bad=0
for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
[ "$bad" -eq 0 ] || fail "a tune judging failed"
say "every tune point judged"
CH=$HOST/$B/$FIT/choice_all$SUF.json
[ -e $CH ] || "${TB[@]}" tune $MODEL $FIT --sigma $SIGMA --all --write-choice > $D/out_iti_${MODEL}_${FIT}${SUF}_tune_all_20261122.txt 2>&1 || fail "tune table (all points)"
read -r K A STEM <<< "$(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c["k"], c["alpha"], c["stem"])' $CH)"
[ -n "$STEM" ] || fail "no choice"
say "chosen among all points: K $K alpha $A"
if [ ! -e $HOST/$B/$FIT/mmlu/$STEM.json ]; then
  DX $GPU $PY -I $IT mmlu --models $M --model $MODEL --fit $ROOT/$B/fit_$FIT.pt --sigma $SIGMA --k $K --alpha $A \
    --items $ROOT/data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids $ROOT/Iteration_01/capability/sets/mmlu_14042_20261120.json \
    --out $ROOT/$B/$FIT/mmlu/$STEM.json >> $log 2>&1 || fail "MMLU of the chosen point"
fi
"${TB[@]}" budget $MODEL $FIT --sigma $SIGMA > $D/out_iti_${MODEL}_${FIT}${SUF}_budget_20261122.txt 2>&1 || fail "budget table"
if [ ! -e $HOST/$B/$FIT/gen/test_$STEM.json ]; then
  DX $GPU $PY -I $IT gen --models $M --model $MODEL --fit $ROOT/$B/fit_$FIT.pt --sigma $SIGMA --k $K --alpha $A --split test \
    --out $ROOT/$B/$FIT/gen/test_$STEM.json >> $log 2>&1 || fail "test generation"
fi
bash $L/iti_judge.sh $MODEL test iti_${FIT}${SUF}_all $LIKE $B/$FIT/gen/test_$STEM.json >> $log 2>&1 || fail "test judging"
"${TB[@]}" test $MODEL $FIT --sigma $SIGMA --all > $D/out_iti_${MODEL}_${FIT}${SUF}_test_all_20261122.txt 2>&1 || fail "test table (all points)"
say "test read; GSM8K at both choices on GPUs $GA and $GB"
MAIN=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["stem"])' $HOST/$B/$FIT/choice$SUF.json)
[ -n "$MAIN" ] || fail "no main choice"
mkdir -p $HOST/$B/$FIT/gsm8k
GS="--items $ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet --ids $ROOT/Iteration_01/capability/sets/gsm8k_1319_20261119.json --max-new-tokens $TOK"
gpids=()
i=0
for st in $MAIN $STEM; do
  [ $i -eq 0 ] && g=$GA || g=$GB
  i=$((i + 1))
  [ ! -e $HOST/$B/$FIT/gsm8k/$st.json ] || continue
  read -r kk aa <<< "$(python3 -c '
import json, sys
s = json.load(open(sys.argv[1]))["setting"]
print(s["k"], s["alpha"])' $HOST/$B/$FIT/mmlu/$st.json)"
  [ -n "$aa" ] || fail "no setting for $st"
  DX $g $PY -I $IT gsm8k --models $M --model $MODEL --fit $ROOT/$B/fit_$FIT.pt --sigma $SIGMA --k $kk --alpha $aa $GS \
    --out $ROOT/$B/$FIT/gsm8k/$st.json >> $log 2>&1 &
  gpids+=("$!")
done
bad=0
for pid in "${gpids[@]}"; do wait "$pid" || bad=1; done
[ "$bad" -eq 0 ] || fail "a GSM8K pass"
docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -w $ROOT truth_dec python3 -B Iteration_01/diag/iti_gsm8k.py $MODEL $FIT $MAIN $STEM \
  > $D/out_iti_${MODEL}_${FIT}${SUF}_gsm8k_20261122.txt 2>&1 || fail "GSM8K table"
say "done"
touch ${MK}_DONE

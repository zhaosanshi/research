#!/bin/bash
# 2026-11-22 08:3x, exploration: one ITI variant from its grid to its test reading without anybody watching (the
# sensitivity checks of summary section 14: the fit on training + tune questions, and the paper's sigma). The rules are
# the main reading's (discussion/work_queue/2026-11-22_queue.md section 4), nothing is chosen by hand:
#   1. wait for the lanes that are still filling the grid (their markers, DONE or FAILED), then run the whole grid once
#      more on one card: existing files are kept, so only the points a lane lost are computed (iti_grid.sh);
#   2. the tune answers of every point within the MMLU budget are judged, one work dir per K, in parallel;
#   3. diag/iti_table.py tune --write-choice;
#   4. the test answers at the choice are generated and judged, and the test table is written to
#      Iteration_01/diag/out_iti_<MODEL>_<FIT><_sp>_test_20261122.txt (budget and tune tables next to it).
# usage: iti_chain.sh MODEL FIT SIGMA GPU "K ..." "ALPHA ..." LIKE_NAME [LANE_TAG ...]
#   e.g. iti_chain.sh llama3.1-8b traintune_seed42 endq 2 "16 32 48 64" "2.5 5 7.5 10 15 20 25 30" pospromptl16 tt1 tt2
# Markers: ITI_CHAIN_<MODEL>_<FIT>_<SIGMA>_DONE / _FAILED. Log lines in iti_all.log.
# After the read-only review of 2026-11-22 10 h: the declared grid is passed to the table, which refuses when a point
# lacks its MMLU pass (unless the early stop of its K has been reached).
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
D=Iteration_01/diag
IT=$ROOT/Iteration_01/code/scripts/newsets/iti_baseline.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
MODEL=$1; FIT=$2; SIGMA=$3; GPU=$4; KS=$5; ALPHAS=$6; LIKE=$7
[ -n "$LIKE" ] || { echo "usage: iti_chain.sh MODEL FIT SIGMA GPU \"K ...\" \"ALPHA ...\" LIKE_NAME [LANE_TAG ...]"; exit 2; }
shift 7
case "$SIGMA" in endq) SUF="" ;; pairs) SUF="_sp" ;; *) echo "SIGMA is endq or pairs"; exit 2 ;; esac
B=Iteration_01/baselines/iti/$MODEL
MK=$L/ITI_CHAIN_${MODEL}_${FIT}_${SIGMA}
say() { echo "$(date '+%F %T') iti chain $MODEL $FIT $SIGMA: $*" >> $L/iti_all.log; }
rm -f ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
for tag in "$@"; do
  g=$L/ITI_GRID_${MODEL}_${FIT}_${SIGMA}_$tag
  until [ -e ${g}_DONE ] || [ -e ${g}_FAILED ]; do sleep 30; done
done
say "the lanes have ended; filling the grid on GPU $GPU"
bash $L/iti_grid.sh $MODEL $GPU $FIT $SIGMA "$KS" "$ALPHAS" fill > $L/iti_chain_${FIT}_${SIGMA}_fill.out 2>&1 || fail "the grid is not complete"
TB=(timeout 900 python3 -B $D/iti_table.py)
"${TB[@]}" budget $MODEL $FIT --sigma $SIGMA > $D/out_iti_${MODEL}_${FIT}${SUF}_budget_20261122.txt 2>&1 || fail "budget table"
stems=$("${TB[@]}" stems $MODEL $FIT --sigma $SIGMA 2>> $L/iti_chain_${FIT}_${SIGMA}_fill.out) || fail "no list of the points within the budget"
[ -n "$stems" ] || fail "no point within the budget"
pids=()
for K in $KS; do
  files=""
  while read -r k stem; do
    [ "$k" = "$K" ] || continue
    [ -s $HOST/$B/$FIT/gen/tune_$stem.json ] || fail "no tune answers for $stem"
    files="$files $B/$FIT/gen/tune_$stem.json"
  done <<< "$stems"
  [ -n "$files" ] || continue
  bash $L/iti_judge.sh $MODEL tune iti_${FIT}${SUF}_k${K}a $LIKE $files > $L/iti_chain_${FIT}_${SIGMA}_judge_k$K.out 2>&1 &
  pids+=("$!")
done
bad=0
for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
[ "$bad" -eq 0 ] || fail "a tune judging failed"
say "tune judged"
CH=$HOST/$B/$FIT/choice$SUF.json
[ -e $CH ] || "${TB[@]}" tune $MODEL $FIT --sigma $SIGMA --ks "$KS" --alphas "$ALPHAS" --write-choice > $D/out_iti_${MODEL}_${FIT}${SUF}_tune_20261122.txt 2>&1 || fail "tune table"
read -r K A STEM <<< "$(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c["k"], c["alpha"], c["stem"])' $CH)"
[ -n "$STEM" ] || fail "no choice"
say "chosen K $K alpha $A"
PY=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)
[ -n "$PY" ] || fail "model not in the config"
if [ ! -e $HOST/$B/$FIT/gen/test_$STEM.json ]; then
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec \
    $PY -I $IT gen --models $M --model $MODEL --fit $ROOT/$B/fit_$FIT.pt --sigma $SIGMA --k $K --alpha $A --split test \
    --out $ROOT/$B/$FIT/gen/test_$STEM.json >> $L/iti_chain_${FIT}_${SIGMA}_fill.out 2>&1 || fail "test generation"
fi
bash $L/iti_judge.sh $MODEL test iti_${FIT}${SUF} $LIKE $B/$FIT/gen/test_$STEM.json > $L/iti_chain_${FIT}_${SIGMA}_judge_test.out 2>&1 || fail "test judging"
"${TB[@]}" test $MODEL $FIT --sigma $SIGMA > $D/out_iti_${MODEL}_${FIT}${SUF}_test_20261122.txt 2>&1 || fail "test table"
say "done"
touch ${MK}_DONE

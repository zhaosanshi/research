#!/bin/bash
# 2026-11-22 05:4x, exploration: the ITI baseline (scripts/newsets/iti_baseline.py; method facts in
# novelty/prior_work/2026-11-22_iti_method_spec.md) for one model on one card: for every K of the list and every alpha
# of the list, MMLU by option probability on the full 14,042 items (the budget of the project's own direction is
# read on its selection half) and the answers of the TruthfulQA tune split (192 questions). K = 0 scores the model
# itself (once, alpha ignored). Nothing here calls a judge. Existing files are kept.
# Early stop within a K (alphas must be given in increasing order): when a point's MMLU change on the selection half
# is below -STOP points (default 3; the budget is 1), the larger alphas of that K are not run, and the tune answers
# are generated only for points whose selection-half change is not below -STOP.
# GEN_ONLY=1 (added 2026-11-22 06 h, after the first three tune readings at alpha 5 were seen: -39, -39, -29 against
# -34 without a write): only the tune answers, for every point of the lists, no MMLU pass and no early stop. For the
# reading "the best ITI point on tune whatever its MMLU cost", which is the most favourable one for the baseline.
# The fit file must exist: baselines/iti/<MODEL>/fit_<FIT>.pt (iti_baseline.py fit). Outputs: baselines/iti/<MODEL>/<FIT>/mmlu/
# and .../gen/tune_*.json; the K = 0 pass is baselines/iti/<MODEL>/mmlu_k0.json.
# usage: iti_grid.sh MODEL GPU FIT SIGMA "K ..." "ALPHA ..." TAG      e.g. iti_grid.sh llama3.1-8b 1 train_seed42 endq "16 32" "5 10 15" a
# Markers: ITI_GRID_<MODEL>_<FIT>_<SIGMA>_<TAG>_DONE / _FAILED. Log: iti_<MODEL>_<TAG>.log, iti_all.log.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
IT=$CODE/scripts/newsets/iti_baseline.py
M=$CODE/configs/newsets/frozen_models_20261119.json
MODEL=$1; GPU=$2; FIT=$3; SIGMA=$4; KS=$5; ALPHAS=$6; TAG=$7
[ -n "$TAG" ] || { echo "usage: iti_grid.sh MODEL GPU FIT SIGMA \"K ...\" \"ALPHA ...\" TAG"; exit 2; }
B=Iteration_01/baselines/iti/$MODEL
MK=$L/ITI_GRID_${MODEL}_${FIT}_${SIGMA}_${TAG}
log=$L/iti_${MODEL}_$TAG.log
say() { echo "$(date '+%F %T') iti grid $MODEL $FIT $SIGMA $TAG: $*" >> $L/iti_all.log; }
rm -f ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
[ -s $HOST/$B/fit_$FIT.pt ] || fail "no fit file fit_$FIT.pt"
case "$SIGMA" in endq) SUF="" ;; pairs) SUF="_sp" ;; *) fail "SIGMA is endq or pairs" ;; esac
plan=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || fail "model not in the config"
PY=$plan
[ -n "$PY" ] || fail "model not in the config"
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
ITEMS=$ROOT/data_external/mmlu_20261119/all/test-00000-of-00001.parquet
IDS=$ROOT/Iteration_01/capability/sets/mmlu_14042_20261120.json
D=$B/$FIT                                  # one directory per fit; the K = 0 pass is shared by all fits
mkdir -p $HOST/$D/mmlu $HOST/$D/gen
say "start on GPU $GPU: K $KS, alpha $ALPHAS"
FAILED=0
for K in $KS; do
  if [ "$K" = 0 ]; then
    if [ ! -e $HOST/$B/mmlu_k0.json ]; then
      "${DX[@]}" $PY -I $IT mmlu --models $M --model $MODEL --k 0 --items $ITEMS --ids $IDS --out $ROOT/$B/mmlu_k0.json >> $log 2>&1
      rc=$?; [ "$rc" -eq 0 ] || { say "mmlu K 0 exit=$rc"; FAILED=1; }
    fi
    continue
  fi
  for A in $ALPHAS; do
    name=k${K}_a$(echo "$A" | tr . -)$SUF
    if [ "${GEN_ONLY:-0}" != 1 ]; then
      if [ ! -e $HOST/$D/mmlu/$name.json ]; then
        "${DX[@]}" $PY -I $IT mmlu --models $M --model $MODEL --fit $ROOT/$B/fit_$FIT.pt --sigma $SIGMA --k $K --alpha $A \
            --items $ITEMS --ids $IDS --out $ROOT/$D/mmlu/$name.json >> $log 2>&1
        rc=$?; [ "$rc" -eq 0 ] || { say "mmlu $name exit=$rc"; FAILED=1; }
      fi
      chg=$(python3 -B Iteration_01/diag/iti_table.py point $MODEL $FIT $name 2>> $log) || { say "no MMLU reading for $name"; FAILED=1; continue; }
      if awk -v c="$chg" -v s="${STOP:-3}" 'BEGIN { exit !(c < -s) }'; then
        say "K $K alpha $A: selection-half MMLU change $chg, below -${STOP:-3}: larger alphas of this K are skipped"
        break
      fi
    fi
    if [ ! -e $HOST/$D/gen/tune_$name.json ]; then
      "${DX[@]}" $PY -I $IT gen --models $M --model $MODEL --fit $ROOT/$B/fit_$FIT.pt --sigma $SIGMA --k $K --alpha $A \
          --split tune --out $ROOT/$D/gen/tune_$name.json >> $log 2>&1
      rc=$?; [ "$rc" -eq 0 ] || { say "gen tune $name exit=$rc"; FAILED=1; }
    fi
  done
  say "K $K done"
done
if [ "$FAILED" -ne 0 ]; then touch ${MK}_FAILED; exit 1; fi
say "done"
touch ${MK}_DONE

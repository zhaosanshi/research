#!/usr/bin/env bash
# big_chain.sh <gpu(s)> <model> <download-dir-name>: wait for the ModelScope download, run the four-regime chain,
# then E14 question-level MC (TruthfulQA + BBQ) with the layers the chain picked.
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
GPU="$1"; M="$2"; DL="$3"
until [ -f logs/download-$DL.done ]; do sleep 60; done
stamp "chain $M start on gpu $GPU"
bash code/experiments/scripts/run_model_chain.sh $GPU $M tqa bbq halu knowledge > logs/chain_$M.log 2>&1; stamp "chain $M rc=$?"
if [ -f results_tqa/E13/$M/clean_gap.json ]; then
  read EL1 EL2 EL3 PL WL <<< "$(bash logs/pick_layers.sh results_tqa $M)"
  bash code/experiments/scripts/run_e14_model.sh $GPU $M $EL1 $EL2 $EL3 $PL $WL > logs/e14_$M.log 2>&1; stamp "e14 tqa $M rc=$?"
fi
if [ -f results_bbq/E13/$M/clean_gap.json ]; then
  read EL1 EL2 EL3 PL WL <<< "$(bash logs/pick_layers.sh results_bbq $M)"
  bash code/experiments/scripts/run_e14_model.sh $GPU $M $EL1 $EL2 $EL3 $PL $WL bbq_mc bbq_tf E2b_bbq E1_directions_bbq results_bbq > logs/e14bbq_$M.log 2>&1; stamp "e14 bbq $M rc=$?"
fi
stamp "big $M DONE"

#!/usr/bin/env bash
# C row for gemma3-4b with C orthogonalised to the MEAN-DIFF E of the same layer (its E row holds with mean-diff, not DAS)
set -u
ROOT_HOST=/data/sanshizhao/Truth_Decoupling; ROOT=/workspace/Truth_Decoupling; RES=$ROOT/results_das_tf
LOG=$ROOT_HOST/logs/c_meandiff_gemma3-4b.log; rm -f $LOG.DONE $LOG.FAIL
echo "== $(date '+%F %T') C chain (mean-diff E) start gpu=1 model=gemma3-4b" >> $LOG
if docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=1 -w $ROOT/code/experiments truth_dec python3 scripts/e23_knob_sweep.py --exp E2b_tqa --model gemma3-4b --all-layers --tune-pairs 192 --skip-e --e-estimator meandiff --tag meandiff_C >> $LOG 2>&1; then echo "== $(date '+%F %T') done" >> $LOG; touch $LOG.DONE; else echo "== $(date '+%F %T') FAILED" >> $LOG; touch $LOG.FAIL; fi

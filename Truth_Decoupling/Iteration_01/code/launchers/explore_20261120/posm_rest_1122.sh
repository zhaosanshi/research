#!/bin/bash
# 2026-11-22 01:2x: the mean-difference comparison at the write rule "prompt" (posm.sh; TruthfulQA test, DeepSeek Pro)
# for the four models that night2b_1121.sh left out. It was cut to Llama alone on my reading of a remark of the lead;
# he corrected that reading at 01:15 ("I did not say that, you can do them all").
#   lane A, GPU 0 (free): Gemma-3-12B, then Gemma-3-27B, then Qwen3.8-27B on this one card (its usual two cards are
#     busy until the night queues end; if one card is not enough the step fails loudly and is rerun on 2,3 later);
#   lane B, GPU 1: Qwen3.5-9B, after its nsc_prompt.sh run on that card has ended.
# A failed model does not stop the next one; the failure is in POSM_<KEY>_<TAG>_GPU_FAILED / _JUDGE_FAILED and in the
# exit status of this script. Table: python3 -B Iteration_01/diag/posm_table.py
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/posm.sh
pids=()
(
  rc=0
  bash $P g12 24 2.5 l24 0 judge_test_l24mdg_test > $L/posm_g12.out 2>&1 || rc=1
  bash $P g27 28 1 l28 0 judge_test_l28mdg_test > $L/posm_g27.out 2>&1 || rc=1
  bash $P q27 28 1.75 l28 0 judge_test_mdg_test > $L/posm_q27.out 2>&1 || rc=1
  echo "posm rest lane A end rc=$rc $(date '+%T')" >> $L/posp_all.log
  exit $rc
) &
pids+=("$!")
(
  while ps -eo args | grep -q "[n]sc_prompt.sh qwen3.5-9b "; do sleep 20; done
  bash $P q9 15 2 l15 1 judge_test_mdg_test > $L/posm_q9.out 2>&1; rc=$?
  echo "posm rest lane B end rc=$rc $(date '+%T')" >> $L/posp_all.log
  exit $rc
) &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then touch $L/POSM_REST_1122_FAILED; exit 1; fi
touch $L/POSM_REST_1122_GPU_DONE

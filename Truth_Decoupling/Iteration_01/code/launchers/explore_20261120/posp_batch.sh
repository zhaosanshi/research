#!/bin/bash
# 2026-11-21 12:3x, posp_batch.sh, exploration: the main table at the write rule "prompt" for all five models, budget 1
# MMLU point (the strength of each candidate layer is the main-table one; the MMLU read is the same under both rules,
# so the candidates do not change). pospall.sh per (model, layer, strength); the layer is then chosen on tune by the
# main-table rule (N1's tune score), in diag/main_table_prompt.py.
# Reason for running all five now: on tune, N1 at "prompt" equals N1 at the training rule on the two models done
# (Llama-3.1-8B +27 / +26, Gemma-3-12B +74 / +75), and on Llama the GSM8K damage of the training rule (-19.0) is gone
# at "prompt" (+1.1).
#   lane A, GPU 0 (after the Gemma-3-12B N1 run at "prompt" there): Llama L11 / 0.75 (+ single C), Gemma-3-12B
#     L22 / 1.25 (+ single C), Qwen3.5-9B L15 / 2;
#   lane B, GPUs 2,3 (after the Qwen3.8-27B GSM8K run): Qwen3.8-27B L28 / 1.75 on both cards; then Gemma-3-27B
#     L28 / 1 on GPU 2 and L26 / 0.75 on GPU 3 side by side; then the Gemma-3-27B MMLU of g27_mm28.sh on GPU 2.
# The single C at "prompt" of Llama-3.1-8B and Gemma-3-12B also serves their second layers (one work dir per run);
# the two Gemma-3-27B lanes run side by side, so the second one skips single C (SKIPC=1) to avoid writing the same files.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
echo "$(date '+%F %T') posp_batch start" >> $L/posp_all.log
(
  until [ -e $L/POSP2_l24_GPU_DONE ]; do sleep 30; done
  bash $L/pospall.sh llama 11 0.75 l11 0 judge_tune_ecll19_tune judge_test_tll19_test > $L/pospall_llama_l11.out 2>&1
  bash $L/pospall.sh g12 22 1.25 l22 0 judge_tune_ec25_tune judge_test_t25_test > $L/pospall_g12_l22.out 2>&1
  bash $L/pospall.sh q9 15 2 l15 0 judge_tune_ec9b_tune judge_test_t9b16_test > $L/pospall_q9_l15.out 2>&1
  echo "posp_batch lane A done $(date '+%T')" >> $L/posp_all.log
) &
(
  until grep -q -E "^gsm8k qwen3.8-27b (cap [0-9]+ )?exit=" $L/gsm8k_all.log; do sleep 60; done
  bash $L/pospall.sh q27 28 1.75 l28 2,3 judge_tune_ec2733_tune judge_test_t2733_test > $L/pospall_q27_l28.out 2>&1
  bash $L/pospall.sh g27 28 1 l28 2 judge_tune_l28_tune judge_test_l28_test > $L/pospall_g27_l28.out 2>&1 &
  SKIPC=1 bash $L/pospall.sh g27 26 0.75 l26 3 judge_tune_ec31_tune judge_test_t31_test > $L/pospall_g27_l26.out 2>&1 &
  wait
  echo "posp_batch lane B done $(date '+%T')" >> $L/posp_all.log
  bash $L/g27_mm28.sh 2 > $L/g27_mm28.out 2>&1
) &
wait
touch $L/POSP_BATCH_DONE

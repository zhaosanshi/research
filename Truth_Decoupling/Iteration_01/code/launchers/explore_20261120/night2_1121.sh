#!/bin/bash
# 2026-11-21 23:3x, second night queue, behind night_1121.sh on the cards it frees. Two small things the prompt-rule
# write-up needs at the main-table settings:
#   a. GSM8K, the re-asks of the full N1 (gsm8k_n1_pos.sh; a handful of items per model);
#   b. TruthfulQA test, the mean-difference direction at the same layer and write amount, written at "prompt"
#      (posm.sh), so that "learned direction against mean difference" is also known at the rule the method would use.
#   lane 1, GPU 1, after the Llama / Qwen3.5-9B lane of night_1121.sh: both things for these two models;
#   lane 2, GPUs 2,3, after the Qwen3.8-27B jobs of night_1121.sh (its TriviaQA / PopQA lane and its GSM8K passes at
#     1,024): Qwen3.8-27B on both cards, then Gemma-3-12B on GPU 2 and Gemma-3-27B on GPU 3 side by side.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
(
  until [ -e $L/NSP_qwen3.5-9b_DONE ]; do sleep 60; done
  bash $L/gsm8k_n1_pos.sh 1 llama3.1-8b 1 qwen3.5-9b > $L/gsm8k_n1_small.out 2>&1
  bash $L/posm.sh llama 16 3.5 l16 1 judge_test_l16mdg_test > $L/posm_llama.out 2>&1
  bash $L/posm.sh q9 15 2 l15 1 judge_test_mdg_test > $L/posm_q9.out 2>&1
  echo "night2 lane 1 done $(date '+%T')" >> $L/posp_all.log
) &
(
  until [ -e $L/NSP_qwen3.8-27b_DONE ] && grep -q "^gsm8k qwen3.8-27b cap 1024 at prompt exit=" $L/gsm8k_all.log; do sleep 60; done
  bash $L/gsm8k_n1_pos.sh 2,3 qwen3.8-27b > $L/gsm8k_n1_q27.out 2>&1
  bash $L/posm.sh q27 28 1.75 l28 2,3 judge_test_mdg_test > $L/posm_q27.out 2>&1
  ( bash $L/posm.sh g12 24 2.5 l24 2 judge_test_l24mdg_test > $L/posm_g12.out 2>&1 ) &
  ( bash $L/gsm8k_n1_pos.sh 3 gemma3-27b > $L/gsm8k_n1_g27.out 2>&1; bash $L/posm.sh g27 28 1 l28 3 judge_test_l28mdg_test > $L/posm_g27.out 2>&1 ) &
  wait
  echo "night2 lane 2 done $(date '+%T')" >> $L/posp_all.log
) &
wait
touch $L/NIGHT2_1121_DONE

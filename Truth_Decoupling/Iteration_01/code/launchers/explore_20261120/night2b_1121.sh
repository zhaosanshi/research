#!/bin/bash
# 2026-11-21 23:5x, replaces night2_1121.sh (stopped while it was still waiting; nothing of it had started). The lead:
# a judged comparison does not need every model at once, one run shows the rough result. So the mean-difference
# comparison at the write rule "prompt" (posm.sh, DeepSeek Pro, 529 questions) is run for ONE model, Llama-3.1-8B; the
# other four follow only if the prompt rule is kept for the main table. The GSM8K re-asks of the full N1 stay for all
# models (a handful of items, no judge call; Gemma-3-12B has none).
#   lane 1, GPU 1, after the Llama / Qwen3.5-9B lane of night_1121.sh: GSM8K re-asks of these two, then posm for Llama;
#   lane 2, GPUs 2,3, after the Qwen3.8-27B jobs of night_1121.sh: GSM8K re-asks of Qwen3.8-27B (cap 1,024), then of
#     Gemma-3-27B on GPU 3.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
(
  until [ -e $L/NSP_qwen3.5-9b_DONE ]; do sleep 60; done
  bash $L/gsm8k_n1_pos.sh 1 llama3.1-8b 1 qwen3.5-9b > $L/gsm8k_n1_small.out 2>&1
  bash $L/posm.sh llama 16 3.5 l16 1 judge_test_l16mdg_test > $L/posm_llama.out 2>&1
  echo "night2b lane 1 done $(date '+%T')" >> $L/posp_all.log
) &
(
  until [ -e $L/NSP_qwen3.8-27b_DONE ] && grep -q "^gsm8k qwen3.8-27b cap 1024 at prompt exit=" $L/gsm8k_all.log; do sleep 60; done
  bash $L/gsm8k_n1_pos.sh 2,3 qwen3.8-27b > $L/gsm8k_n1_q27.out 2>&1
  bash $L/gsm8k_n1_pos.sh 3 gemma3-27b > $L/gsm8k_n1_g27.out 2>&1
  echo "night2b lane 2 done $(date '+%T')" >> $L/posp_all.log
) &
wait
touch $L/NIGHT2_1121_DONE

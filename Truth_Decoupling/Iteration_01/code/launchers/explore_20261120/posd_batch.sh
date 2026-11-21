#!/bin/bash
# 2026-11-21 13:2x, posd_batch.sh, exploration: the strong comparator at the write rule "prompt" (posd.sh) for every
# candidate layer of the prompt-rule main table, budget 1 (the layer is chosen later on tune, so all candidates):
#   lane A, GPU 0, after lane A of posp_batch.sh: Llama 16 / 3.5 and 11 / 0.75, Gemma-3-12B 24 / 2.5 and 22 / 1.25,
#     Qwen3.5-9B 15 / 2;
#   lane B, after posp_batch.sh has ended (its last step is the Gemma-3-27B MMLU on GPU 2): Qwen3.8-27B 28 / 1.75 on
#     GPUs 2,3, then Gemma-3-27B 28 / 1 on GPU 2 and 26 / 0.75 on GPU 3 side by side (different export names).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
echo "$(date '+%F %T') posd_batch start" >> $L/posp_all.log
(
  until grep -q "posp_batch lane A done" $L/posp_all.log; do sleep 60; done
  bash $L/posd.sh llama 16 3.5 l16 0 judge_test_testD_E_L11 > $L/posd_llama_l16.out 2>&1
  bash $L/posd.sh llama 11 0.75 l11 0 judge_test_testD_E_L11 > $L/posd_llama_l11.out 2>&1
  bash $L/posd.sh g12 24 2.5 l24 0 judge_test_testD_E_L22 > $L/posd_g12_l24.out 2>&1
  bash $L/posd.sh g12 22 1.25 l22 0 judge_test_testD_E_L22 > $L/posd_g12_l22.out 2>&1
  bash $L/posd.sh q9 15 2 l15 0 judge_test_testD_E_L15 > $L/posd_q9_l15.out 2>&1
  echo "posd_batch lane A done $(date '+%T')" >> $L/posp_all.log
) &
(
  until [ -e $L/POSP_BATCH_DONE ]; do sleep 60; done
  bash $L/posd.sh q27 28 1.75 l28 2,3 judge_test_testD_E_L28 > $L/posd_q27_l28.out 2>&1
  bash $L/posd.sh g27 28 1 l28 2 judge_test_testD_E_L26 > $L/posd_g27_l28.out 2>&1 &
  bash $L/posd.sh g27 26 0.75 l26 3 judge_test_testD_E_L26 > $L/posd_g27_l26.out 2>&1 &
  wait
  echo "posd_batch lane B done $(date '+%T')" >> $L/posp_all.log
) &
wait
touch $L/POSD_BATCH_DONE

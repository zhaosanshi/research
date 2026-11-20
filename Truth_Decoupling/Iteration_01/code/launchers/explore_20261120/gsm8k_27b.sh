#!/bin/bash
# 2026-11-21 07:1x, exploration: GSM8K under the main-table settings for the two 27B models (gsm8k_main.sh), queued
# behind the jobs that hold their cards so that no card gets two large jobs:
#   Gemma-3-27B (E at layer 28, strength 1; about 59 GB) on GPU 0, after BOTH small-model GSM8K lanes there have ended
#     (Llama-3.1-8B and Gemma-3-12B; their exit lines in gsm8k_all.log - the per-card marker of gsm8k_main.sh is touched
#     by whichever lane ends first, so it is not used here);
#   Qwen3.8-27B (E at layer 28, strength 1.75; two cards) on GPUs 2,3, after the 27B lane has ended (NS2C_DONE).
# usage: gsm8k_27b.sh g27|q27
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
case "$1" in
  g27)
    until [ "$(grep -c -E "^gsm8k (llama3.1-8b|gemma3-12b) exit=" $L/gsm8k_all.log)" -ge 2 ]; do sleep 60; done
    echo "gsm8k_27b: GPU 0 is free of the small models $(date '+%T')" >> $L/gsm8k_all.log
    bash $L/gsm8k_main.sh 0 gemma3-27b neutral:E28.w.p1 ;;
  q27)
    until [ -e $L/NS2C_DONE ]; do sleep 60; done
    echo "gsm8k_27b: the 27B lane has ended $(date '+%T')" >> $L/gsm8k_all.log
    bash $L/gsm8k_main.sh 2,3 qwen3.8-27b neutral:E28.w.p1.75 ;;
  *) echo "usage: gsm8k_27b.sh g27|q27"; exit 2 ;;
esac

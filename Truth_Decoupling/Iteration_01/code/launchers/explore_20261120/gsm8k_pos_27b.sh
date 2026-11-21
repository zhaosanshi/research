#!/bin/bash
# 2026-11-21 14:2x, gsm8k_pos_27b.sh: GSM8K for the two 27B models with E at their main-table settings written at the
# core rule "prompt" (gsm8k_pos.sh), cap 512 as their training-rule passes, so that all five models have GSM8K at the
# prompt rule (at the training rule these two lose nothing: -0.3 and -0.5).
#   Gemma-3-27B (E28 at 1) on GPU 0 now (about 59 GB; GPU 0 holds 3.7 GB of another user's process);
#   Qwen3.8-27B (E28 at 1.75) on GPUs 2,3 after g27_mm28.sh has finished there (marker ONCAUT_MM_L28B_DONE).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
bash $L/gsm8k_pos.sh 0 prompt gemma3-27b neutral:E28.w.p1 &
( until [ -e $L/ONCAUT_MM_L28B_DONE ]; do sleep 60; done; bash $L/gsm8k_pos.sh 2,3 prompt qwen3.8-27b neutral:E28.w.p1.75 ) &
wait

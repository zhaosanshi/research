#!/bin/bash
# 2026-11-21 12:1x, gsm8k_orig.sh, exploration: GSM8K with E at the layer chosen for TruthfulQA (before the layer
# change of the main table), at the strength the main-table rule picks there within 1 point of MMLU (Llama-3.1-8B
# layer 11 at 0.75, Gemma-3-12B layer 22 at 1.25), training write rule. Question: did the move to a later layer with a
# larger strength (16 / 3.5 and 24 / 2.5; GSM8K -19 and -14) cause the GSM8K damage, which MMLU does not show?
# Waits for the Qwen3.8-27B GSM8K run on GPUs 2,3 to end, then runs on GPU 3 (gsm8k_main.sh; the no-write passes are
# already there). usage: gsm8k_orig.sh
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
until grep -q -E "^gsm8k qwen3.8-27b (cap [0-9]+ )?exit=" $L/gsm8k_all.log; do sleep 60; done
echo "gsm8k_orig: original layers start on GPU 3 $(date '+%T')" >> $L/gsm8k_all.log
bash $L/gsm8k_main.sh 3 llama3.1-8b neutral:E11.w.p0.75 gemma3-12b neutral:E22.w.p1.25

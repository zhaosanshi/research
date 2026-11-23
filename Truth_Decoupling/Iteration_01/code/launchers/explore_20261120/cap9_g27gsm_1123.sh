#!/bin/bash
# 2026-11-23 08:2x, queue I section 9: Gemma-3-27B's GSM8K pass on GPU 0 (free; CAA_GRID_gemma3-12b_DONE; about 3.7 GB of
# another process there, the 27B model fits as the program is), in parallel with its MMLU pass still running on GPU 2.
# The cap9_1123.sh lane that would have run this GSM8K pass on GPU 2 after the MMLU pass was stopped first (it only
# waited for the MMLU pass; that pass keeps running).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
bash $L/cap9_model.sh g27 0 gsm > $L/cap9_g27_gsm.out 2>&1
echo "$(date '+%F %T') cap9 lanes: gemma3-27b GSM8K on GPU 0 end" >> $L/n1s_all.log

#!/bin/bash
# 2026-11-21 02:3x: starts l16_strong.sh on GPU 3 once the low-strength strong comparator of Llama has finished its GPU
# part there (both are Llama processes of about 18 GB; the MMLU chain of l16b_all.sh runs on the same card).
cd /data/sanshizhao/Truth_Decoupling/Iteration_01/logs/newsets_20261119
(setsid nohup bash l16_strong.sh 3 'grep -q "lowstrong conf llama" $L/low_all.log' > l16_strong.out 2>&1 < /dev/null &)

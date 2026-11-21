#!/bin/bash
# 2026-11-21 12:2x, gsm8k_pos_q9.sh: GSM8K for Qwen3.5-9B with E at its main-table setting (layer 15, strength 2)
# written at the core rule "prompt", generation cap 1,024 (the cap its no-write pass at 1,024 used; at 512 too many of
# its long solutions are cut off). Runs on GPU 1 once the Llama-3.1-8B prompt-rule run there has ended.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
until grep -q "^gsm8k llama3.1-8b cap 512 at prompt exit=" $L/gsm8k_all.log; do sleep 60; done
NEWTOK=1024 bash $L/gsm8k_pos.sh 1 prompt qwen3.5-9b neutral:E15.w.p2

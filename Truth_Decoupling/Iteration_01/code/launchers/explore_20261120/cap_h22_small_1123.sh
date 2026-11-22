#!/bin/bash
# 2026-11-23 00:1x, queue I section 7.3, the two small models (cap_h22_part.sh): GSM8K on GPU 1 (Llama-3.1-8B, then
# Qwen3.5-9B); their MMLU on whichever of GPU 3, 2, 0 (0 only after CAA_GRID_gemma3-12b_DONE / _FAILED) is free first
# (lock free, memory near 0), because the MMLU run went over the 65 GB limit of GPU 1.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/cap_h22_part.sh
LK=Iteration_01/newsets/selfjudge
say() { echo "$(date '+%F %T') caph22 small lanes: $*" >> $L/n1s_all.log; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
pick() {
  while true; do
    if [ "$(mem 3)" -lt 2000 ] && lockfree 3; then echo 3; return; fi
    if [ "$(mem 2)" -lt 2000 ] && lockfree 2; then echo 2; return; fi
    if { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; } && [ "$(mem 0)" -lt 2000 ] && lockfree 0; then echo 0; return; fi
    sleep 60
  done
}
( bash $P llama 1 gsm > $L/caph22_llama_gsm.out 2>&1; bash $P q9 1 gsm > $L/caph22_q9_gsm.out 2>&1; say "GSM8K end" ) &
( for k in llama q9; do g=$(pick); say "$k MMLU on GPU $g"; bash $P $k $g mmlu > $L/caph22_${k}_mmlu.out 2>&1; done; say "MMLU end" ) &
wait

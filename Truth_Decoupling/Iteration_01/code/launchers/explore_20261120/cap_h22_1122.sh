#!/bin/bash
# 2026-11-22 23:4x, queue I section 7.3: cap_h22_model.sh for the five models, each after its pfam_model.sh (7.6) has
# left the card (PFAM_<MODEL>_GPU_DONE or _FAILED). Small models on GPU 1; the three large ones on whichever of
# GPU 3, 2, 0 (0 only after CAA_GRID_gemma3-12b_DONE / _FAILED) is free first (lock free, memory near 0).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/cap_h22_model.sh
LK=Iteration_01/newsets/selfjudge
say() { echo "$(date '+%F %T') caph22 lanes: $*" >> $L/n1s_all.log; }
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
after() { until [ -e $L/PFAM_$1_GPU_DONE ] || [ -e $L/PFAM_$1_FAILED ]; do sleep 30; done; }
( after llama3.1-8b; bash $P llama 1 > $L/caph22_llama.out 2>&1; after qwen3.5-9b; bash $P q9 1 > $L/caph22_q9.out 2>&1; say "small models end" ) &
for k in g27:gemma3-27b q27:qwen3.8-27b g12:gemma3-12b; do
  ( after ${k#*:}; sleep $((RANDOM % 20)); g=$(pick); say "${k#*:} on GPU $g"; bash $P ${k%%:*} $g > $L/caph22_${k%%:*}.out 2>&1 ) &
done
wait
say "all lanes ended"

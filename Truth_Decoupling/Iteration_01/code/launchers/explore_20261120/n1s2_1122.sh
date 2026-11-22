#!/bin/bash
# 2026-11-22 23:2x, queue I section 7 (7.1, 7.2, 7.4): n1s2_model.sh for the five models.
#   GPU 1: Llama-3.1-8B, then Qwen3.5-9B;  GPU 3: Gemma-3-27B;  GPU 2: Qwen3.8-27B (its lock, queued with the queue-J
#   self-judge runs);  Gemma-3-12B: whichever of GPU 0 (after CAA_GRID_gemma3-12b_DONE / _FAILED), 3, 2 is free first.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/n1s2_model.sh
LK=Iteration_01/newsets/selfjudge
say() { echo "$(date '+%F %T') n1s2 lanes: $*" >> $L/n1s_all.log; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
pick() {
  while true; do
    if { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; } && [ "$(mem 0)" -lt 2000 ] && lockfree 0; then echo 0; return; fi
    if [ "$(mem 3)" -lt 2000 ] && lockfree 3; then echo 3; return; fi
    if [ "$(mem 2)" -lt 2000 ] && lockfree 2; then echo 2; return; fi
    sleep 60
  done
}
( bash $P llama 1 > $L/n1s2_llama.out 2>&1; bash $P q9 1 > $L/n1s2_q9.out 2>&1; say "lane GPU 1 end" ) &
( bash $P g27 3 > $L/n1s2_g27.out 2>&1; say "lane GPU 3 end" ) &
( bash $P q27 2 > $L/n1s2_q27.out 2>&1; say "lane GPU 2 end" ) &
( sleep 120; g=$(pick); say "Gemma-3-12B on GPU $g"; bash $P g12 $g > $L/n1s2_g12.out 2>&1; say "lane Gemma-3-12B end" ) &
wait
say "all lanes ended"

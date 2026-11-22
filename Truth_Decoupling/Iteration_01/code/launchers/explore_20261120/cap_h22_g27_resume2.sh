#!/bin/bash
# 2026-11-23 02:4x, queue I: at 02:33 GPU 2 went back to Gemma-3-27B (give_card_1123.sh), but the Gemma-3-12B 7.3 lane took
# the GPU 2 lock 14 s earlier, so Gemma-3-27B's resumed cap_h22_model.sh only waited for that lock. That waiting script was
# stopped (it had started nothing); this one runs Gemma-3-27B's remaining 7.3 passes on whichever of GPU 3, 2 is free
# first (lock free, memory near 0). Outputs already written are skipped by the programs.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
LK=Iteration_01/newsets/selfjudge
say() { echo "$(date '+%F %T') caph22 lanes: $*" >> $L/n1s_all.log; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
while true; do
  if [ "$(mem 3)" -lt 2000 ] && lockfree 3; then g=3; break; fi
  if [ "$(mem 2)" -lt 2000 ] && lockfree 2; then g=2; break; fi
  sleep 60
done
say "gemma3-27b 7.3 resumed on GPU $g"
bash $L/cap_h22_model.sh g27 $g > $L/caph22_g27_resume2.out 2>&1

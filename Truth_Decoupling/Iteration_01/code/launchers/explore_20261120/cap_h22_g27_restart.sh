#!/bin/bash
# 2026-11-23 00:5x, queue I: Gemma-3-27B's 7.3 run (cap_h22_model.sh) was stopped at 00:5x so that the 7.6 pass of
# Gemma-3-12B, which waited for a card, runs first (7.6 comes before 7.3). Restart after PFAM_gemma3-12b_GPU_DONE (or
# _FAILED) on whichever of GPU 3, 2, 0 (0 only after CAA_GRID_gemma3-12b_DONE / _FAILED) is free first.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
LK=Iteration_01/newsets/selfjudge
say() { echo "$(date '+%F %T') caph22 lanes: $*" >> $L/n1s_all.log; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
until [ -e $L/PFAM_gemma3-12b_GPU_DONE ] || [ -e $L/PFAM_gemma3-12b_FAILED ]; do sleep 30; done
while true; do
  if [ "$(mem 3)" -lt 2000 ] && lockfree 3; then g=3; break; fi
  if [ "$(mem 2)" -lt 2000 ] && lockfree 2; then g=2; break; fi
  if { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; } && [ "$(mem 0)" -lt 2000 ] && lockfree 0; then g=0; break; fi
  sleep 60
done
say "gemma3-27b 7.3 restarted on GPU $g"
bash $L/cap_h22_model.sh g27 $g > $L/caph22_g27_restart.out 2>&1

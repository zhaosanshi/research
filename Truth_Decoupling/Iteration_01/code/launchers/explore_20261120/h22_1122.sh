#!/bin/bash
# 2026-11-22 20:0x, exploration, queue H: cell 4 of the 2x2 for the five models (h22_model.sh, one model per card).
#   lane B, GPU 1: Llama-3.1-8B, then Qwen3.5-9B; then the marker H22_CARD1_DONE (another task waits for it);
#   lane C, GPU 3: Gemma-3-27B, then Gemma-3-12B on GPU 3 only if it can start before 21:30, otherwise after the marker
#           ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE on GPU 3;
#   lane D: Qwen3.8-27B on GPU 2 after that ITI marker.
# Lanes passed as arguments (default: B C D). A failed model does not stop the next one of its lane.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/h22_model.sh
ITI=$L/ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE
LANES=${*:-B C D}
say() { echo "$(date '+%F %T') h22 lanes: $*" >> $L/h22_all.log; }
pids=()
for lane in $LANES; do
  case $lane in
    B) ( rc=0
         bash $P llama 1 > $L/h22_llama.out 2>&1 || rc=1
         bash $P q9 1 > $L/h22_q9.out 2>&1 || rc=1
         touch $L/H22_CARD1_DONE; say "lane B end rc=$rc, H22_CARD1_DONE"; exit $rc ) & pids+=("$!") ;;
    C) ( rc=0
         if [ $((10#$(TZ=Asia/Shanghai date +%H%M))) -lt 2130 ]; then bash $P g27 3 > $L/h22_g27.out 2>&1 || rc=1
         else until [ -e $ITI ]; do sleep 60; done; bash $P g27 3 > $L/h22_g27.out 2>&1 || rc=1; fi
         if [ $((10#$(TZ=Asia/Shanghai date +%H%M))) -ge 2130 ]; then say "g12 waits for the ITI marker"; until [ -e $ITI ]; do sleep 60; done; fi
         bash $P g12 3 > $L/h22_g12.out 2>&1 || rc=1
         say "lane C end rc=$rc"; exit $rc ) & pids+=("$!") ;;
    D) ( until [ -e $ITI ]; do sleep 60; done
         bash $P q27 2 > $L/h22_q27.out 2>&1; rc=$?
         say "lane D end rc=$rc"; exit $rc ) & pids+=("$!") ;;
    *) say "unknown lane $lane"; exit 2 ;;
  esac
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
say "lanes $LANES ended, failed=$failed"
exit $failed

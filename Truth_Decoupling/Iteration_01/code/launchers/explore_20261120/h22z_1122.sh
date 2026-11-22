#!/bin/bash
# 2026-11-22 20:4x, queue H section 5: cell 0 (x_E2, no write) for the five models (h22z_model.sh).
#   lane B, GPU 1: Llama-3.1-8B, then Qwen3.5-9B;
#   lane C: Gemma-3-27B, Gemma-3-12B, Qwen3.8-27B on GPU 3, each only if it starts before 21:30; the ones left wait for
#           ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE and then run on GPU 2 (its lock; queued with other tasks).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/h22z_model.sh
ITI=$L/ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE
say() { echo "$(date '+%F %T') h22z lanes: $*" >> $L/h22_all.log; }
( rc=0
  bash $P llama 1 > $L/h22z_llama.out 2>&1 || rc=1
  bash $P q9 1 > $L/h22z_q9.out 2>&1 || rc=1
  say "lane B end rc=$rc"; exit $rc ) &
pB=$!
( rc=0; left=""
  for k in g27 g12 q27; do
    if [ -z "$left" ] && [ $((10#$(TZ=Asia/Shanghai date +%H%M))) -lt 2130 ]; then
      bash $P $k 3 > $L/h22z_$k.out 2>&1 || rc=1
    else
      left="$left $k"
    fi
  done
  if [ -n "$left" ]; then
    say "after 21:30, waiting for the ITI marker:$left"
    until [ -e $ITI ]; do sleep 60; done
    for k in $left; do bash $P $k 2 > $L/h22z_$k.out 2>&1 || rc=1; done
  fi
  say "lane C end rc=$rc"; exit $rc ) &
pC=$!
failed=0
wait $pB || failed=1
wait $pC || failed=1
say "lanes ended, failed=$failed"
exit $failed

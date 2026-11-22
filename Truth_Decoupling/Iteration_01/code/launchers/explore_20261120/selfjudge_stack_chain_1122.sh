#!/bin/bash
# queue J section 6: the batch-1 self-judge scores of SE / SN1, one model at a time, through selfjudge_stack_1122.sh.
# Llama-3.1-8B and Qwen3.5-9B on card 1; the three larger models on card 2 or 3, whichever lock is free first (card 0 only
# after CAA_GRID_gemma3-12b_DONE / _FAILED). The launcher then waits for the lock and the memory rule. Stops at a failure.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119; LK=Iteration_01/newsets/selfjudge
P=$L/selfjudge_stack_1122.sh
say() { echo "$(TZ=Asia/Shanghai date "+%F %T") st chain: $*" >> $L/st_chain.log; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
ok0() { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; }
say "start"
for m in llama3.1-8b qwen3.5-9b; do
  say "$m on card 1"
  bash $P $m 1 full > $L/st_${m}_full.out 2>&1 || { say "$m failed"; touch $L/ST_CHAIN_FAILED; exit 1; }
  say "$m done"
done
for m in qwen3.8-27b gemma3-27b gemma3-12b; do
  while true; do
    g=""
    if lockfree 2; then g=2; elif lockfree 3; then g=3; elif ok0 && lockfree 0; then g=0; fi
    [ -n "$g" ] && break
    sleep 60
  done
  say "$m on card $g"
  bash $P $m $g full > $L/st_${m}_full.out 2>&1 || { say "$m failed"; touch $L/ST_CHAIN_FAILED; exit 1; }
  say "$m done"
done
say "end"
touch $L/ST_CHAIN_DONE

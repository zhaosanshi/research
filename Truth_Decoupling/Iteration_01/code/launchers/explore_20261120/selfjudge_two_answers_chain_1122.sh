#!/bin/bash
# queue J: the remaining self-judge scores, one model at a time. Qwen3.5-9B on card 1; then the three larger models,
# each on the first eligible card: 3 after queue I has finished its card-3 work (N1S_qwen3.8-27b_test_GPU_DONE or
# _FAILED), 2 after ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE, 0 after CAA_GRID_gemma3-12b_DONE / _FAILED; the
# launcher then waits for the lock and the memory rule. Stops at the first failure.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119; LK=Iteration_01/newsets/selfjudge
P=$L/selfjudge_two_answers_1122.sh; SRC=tj_n1chk,tj_n1ctx,tj_s0,tj_conf
say() { echo "$(TZ=Asia/Shanghai date "+%F %T") tj chain: $*" >> $L/tj_chain.log; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
ok3() { [ -e $L/N1S_qwen3.8-27b_test_GPU_DONE ] || [ -e $L/N1S_qwen3.8-27b_test_FAILED ]; }
ok2() { [ -e $L/ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE ]; }
ok0() { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; }
say "start"
bash $P qwen3.5-9b 1 $SRC full > $L/tj_qwen3.5-9b_full.out 2>&1 || { say "qwen3.5-9b failed"; exit 1; }
say "qwen3.5-9b done"
for m in qwen3.8-27b gemma3-27b gemma3-12b; do
  while true; do
    g=""
    if ok3 && lockfree 3; then g=3; elif ok2 && lockfree 2; then g=2; elif ok0 && lockfree 0; then g=0; fi
    [ -n "$g" ] && break
    sleep 60
  done
  say "$m on card $g"
  bash $P $m $g $SRC full > $L/tj_${m}_full.out 2>&1 || { say "$m failed"; exit 1; }
  say "$m done"
done
say "end"
touch $L/TJ_CHAIN_DONE

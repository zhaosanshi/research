#!/bin/bash
# queue J section 7: after the section-6 chain (ST_CHAIN_DONE), per model: export the statements that have no batch-1
# score yet (container, no GPU), then score them through selfjudge_stack_pick_1122.sh, one model at a time. Llama-3.1-8B
# and Qwen3.5-9B on card 1; the three larger models on card 2 or 3, whichever lock is free first (card 0 only after
# CAA_GRID_gemma3-12b_DONE / _FAILED). Stops at the first failure.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119; LK=Iteration_01/newsets/selfjudge
P=$L/selfjudge_stack_pick_1122.sh
say() { echo "$(TZ=Asia/Shanghai date "+%F %T") sp chain: $*" >> $L/sp_chain.log; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
ok0() { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; }
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w /workspace/Truth_Decoupling truth_dec python3 -B)
say "start; waiting for ST_CHAIN_DONE"
until [ -e $L/ST_CHAIN_DONE ]; do [ -e $L/ST_CHAIN_FAILED ] && { say "section-6 chain failed"; touch $L/SP_CHAIN_FAILED; exit 1; }; sleep 60; done
for m in llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-27b gemma3-12b; do
  if [ ! -s $LK/$m/sp_items_items.json ]; then
    "${DC[@]}" Iteration_01/diag/selfjudge_stack_pick.py export $m >> $L/sp_chain.log 2>&1 || { say "$m export failed"; touch $L/SP_CHAIN_FAILED; exit 1; }
  fi
  case $m in
    llama3.1-8b|qwen3.5-9b) g=1 ;;
    *) while true; do
         g=""
         if lockfree 2; then g=2; elif lockfree 3; then g=3; elif ok0 && lockfree 0; then g=0; fi
         [ -n "$g" ] && break
         sleep 60
       done ;;
  esac
  say "$m on card $g"
  bash $P $m $g full > $L/sp_${m}_full.out 2>&1 || { say "$m failed"; touch $L/SP_CHAIN_FAILED; exit 1; }
  say "$m done"
done
say "end"
touch $L/SP_CHAIN_DONE

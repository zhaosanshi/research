#!/bin/bash
# queue J section 5 (batch size 1): the batch-1 self-judge scores, one model at a time, through
# selfjudge_two_answers_1122.sh. Llama-3.1-8B and Qwen3.5-9B on card 1; Qwen3.8-27B, Gemma-3-27B, Gemma-3-12B on
# card 2 (card 0 is not used). The launcher waits for the lock and the memory rule. Stops at the first failure.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/selfjudge_two_answers_1122.sh; SRC=tj1_chk,tj1_n1,tj1_s0,tj1_conf
say() { echo "$(TZ=Asia/Shanghai date "+%F %T") tj b1 chain: $*" >> $L/tj_chain.log; }
say "start"
for mg in llama3.1-8b:1 qwen3.5-9b:1 qwen3.8-27b:2 gemma3-27b:2 gemma3-12b:2; do
  m=${mg%:*}; g=${mg#*:}
  say "$m on card $g"
  bash $P $m $g $SRC b1 > $L/tj_${m}_b1.out 2>&1 || { say "$m failed"; touch $L/TJ_B1_CHAIN_FAILED; exit 1; }
  say "$m done"
done
say "end"
touch $L/TJ_B1_CHAIN_DONE

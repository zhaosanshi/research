#!/bin/bash
# queue J section 7, the two small models (coordinator 02:37): each on the first card among 1, 2, 3 whose lock is free
# (card 1 only while it has room under 65 GB; card 0 not used); selfjudge_stack_pick_1122.sh then takes that lock.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119; LK=Iteration_01/newsets/selfjudge
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
used() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " -v g=$1 "\$1 == g {print \$2}"; }
for m in llama3.1-8b qwen3.5-9b; do
  while true; do
    g=""
    if lockfree 1 && [ $(( $(used 1) + 20000 )) -le 66560 ]; then g=1; elif lockfree 2; then g=2; elif lockfree 3; then g=3; fi
    [ -n "$g" ] && break
    sleep 20
  done
  echo "$(TZ=Asia/Shanghai date "+%F %T") sp small: $m on card $g" >> $L/sp_chain.log
  bash $L/selfjudge_stack_pick_1122.sh $m $g full4 > $L/sp_${m}_full4.out 2>&1 || { touch $L/SP_SMALL_FAILED; exit 1; }
done
touch $L/SP_SMALL_DONE

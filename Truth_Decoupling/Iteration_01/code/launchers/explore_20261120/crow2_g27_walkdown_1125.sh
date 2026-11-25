#!/bin/bash
# Iteration_01 (main session, Claude Fable 5.2, 2026-11-25 17 h; plan discussion/work_queue/2026-11-24_queue_m.md,
# progress entry of 2026-11-25 17 h, whose rule was committed before this script ran): which of the layers 0-18 of
# gemma3-27b can one 72 GB card train? Until now only layer 0 had been tried (2026-11-24 21:17, out of memory); that
# layers 1-18 do not fit was an inference. This script measures it with the unchanged program, batch size and card type:
# from layer TOP down to layer BOTTOM, one layer per call of crow2_gen_from_1125.sh (CROW2_MINL=<layer>: trains that
# layer, the same acceptance as every other layer, its four tune arms). The memory in use on the card is polled every
# 5 seconds and the highest value of each layer is written to the status line. The walk stops at the first layer that is
# not done:
#   - an out-of-memory error in that layer's log: the layer and all earlier ones are recorded as not fitting (the memory
#     needed grows as the write layer moves toward the input, because the gradient runs from the write layer to the
#     output); marker CROW2_gemma3-27b_WALK_STOP with "out of memory at layer <l>";
#   - any other failure: marker CROW2_gemma3-27b_WALK_STOP with "failure at layer <l>" (a fault, not a memory reading).
# All layers down to BOTTOM done: marker CROW2_gemma3-27b_WALK_DONE. Both markers hold one line with the lowest layer
# done. Resumable: layers that already have their marker CROW2_gemma3-27b_FROM<l>_GEN_DONE are skipped.
# Card: WALK_CARD (default 0), through the lock file of crow2_gen_from_1125.sh, which starts only when ours plus others
# stay within 66560 MiB at the start (an empty card for this model).
# Status lines "crow2 walk gemma3-27b: ..." in crow_all.log; output of the calls in crow2_walk_gemma3-27b.out.
# usage (host): ( nohup setsid bash Iteration_01/logs/newsets_20261119/crow2_g27_walkdown_1125.sh \
#                 > Iteration_01/logs/newsets_20261119/crow2_g27_walkdown_1125.out 2>&1 < /dev/null & )
# WALK_LISTONLY=1: print, for each layer of the walk, what crow2_gen_from_1125.sh would do (no lock, no card) and exit.
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
L=Iteration_01/logs/newsets_20261119
MODEL=gemma3-27b
RUN=Iteration_01/results/ff/gemma3/27b/e44/$MODEL
CARD=${WALK_CARD:-0}
TOP=${WALK_TOP:-18}
BOTTOM=${WALK_BOTTOM:-1}
LISTONLY=${WALK_LISTONLY:-0}
GEN=$L/crow2_gen_from_1125.sh
MK=$L/CROW2_${MODEL}_WALK
say() { echo "$(date '+%F %T') crow2 walk $MODEL: $*" >> $L/crow_all.log; }
case "$CARD" in 0|1|2|3) ;; *) echo "WALK_CARD must be one card"; exit 2 ;; esac
case "$TOP$BOTTOM" in *[!0-9]*) echo "WALK_TOP / WALK_BOTTOM must be layer numbers"; exit 2 ;; esac
[ "$TOP" -ge "$BOTTOM" ] && [ "$TOP" -le 18 ] || { echo "need 18 >= WALK_TOP >= WALK_BOTTOM"; exit 2; }
[ -s $GEN ] || { echo "missing $GEN"; exit 2; }
if [ "$LISTONLY" = 1 ]; then
  for m in $(seq $TOP -1 $BOTTOM); do
    echo "== layer $m"; CROW2_CARD=$CARD CROW2_MINL=$m CROW2_LISTONLY=1 bash $GEN g27 || exit 1
  done
  exit 0
fi
mkdir $L/.crow2_walk_running 2>/dev/null || { echo "already running ($L/.crow2_walk_running)"; exit 3; }
MEMPID=""
cleanup() { [ -n "$MEMPID" ] && kill $MEMPID 2>/dev/null; rmdir $L/.crow2_walk_running 2>/dev/null; }
trap cleanup EXIT
rm -f ${MK}_DONE ${MK}_STOP
low=none
say "start (pid $$): layers $TOP down to $BOTTOM on card $CARD, one layer per call of $(basename $GEN) ($(sha256sum $GEN | cut -c1-16))"
for m in $(seq $TOP -1 $BOTTOM); do
  if [ -e $L/CROW2_${MODEL}_FROM${m}_GEN_DONE ] && [ -s $RUN/dirs/C_L$m.pt ]; then
    low=$m; say "layer $m: already done"; continue
  fi
  [ -s $RUN/dirs/C_L$((m + 1)).pt ] || { say "FAILED: layer $((m + 1)) has no direction file; the walk goes one layer at a time"
                                         echo "failure before layer $m; lowest layer done $low" > ${MK}_STOP; exit 1; }
  mem=$L/.crow2_walk_mem_L$m.txt
  : > $mem
  ( while :; do nvidia-smi -i $CARD --query-gpu=memory.used --format=csv,noheader,nounits >> $mem 2>/dev/null; sleep 5; done ) &
  MEMPID=$!
  say "layer $m: start"
  t0=$(date +%s)
  CROW2_CARD=$CARD CROW2_MINL=$m bash $GEN g27 >> $L/crow2_walk_$MODEL.out 2>&1
  rc=$?
  kill $MEMPID 2>/dev/null; wait $MEMPID 2>/dev/null; MEMPID=""
  peak=$(sort -n $mem | tail -n 1)
  secs=$(( $(date +%s) - t0 ))
  glog=$L/crow2_gen_from${m}_$MODEL.log
  if [ $rc -eq 0 ] && [ -e $L/CROW2_${MODEL}_FROM${m}_GEN_DONE ] && [ -s $RUN/dirs/C_L$m.pt ]; then
    low=$m
    say "layer $m: direction and four arms done in $secs s; highest memory in use on card $CARD ${peak:-?} MiB"
    continue
  fi
  if grep -q "CUDA out of memory" $glog 2>/dev/null; then
    say "layer $m: out of memory (rc $rc, $secs s; highest memory in use ${peak:-?} MiB; $(grep -m 1 -o 'this process has [0-9.]* GiB memory in use' $glog)); lowest layer done: $low"
    echo "out of memory at layer $m; lowest layer done $low" > ${MK}_STOP
    exit 0
  fi
  say "FAILED: layer $m not done and no out-of-memory line (rc $rc, $secs s; see $glog); lowest layer done: $low"
  echo "failure at layer $m; lowest layer done $low" > ${MK}_STOP
  exit 1
done
say "done: every layer from $TOP down to $BOTTOM has a direction and the four arms"
echo "lowest layer done $low" > ${MK}_DONE

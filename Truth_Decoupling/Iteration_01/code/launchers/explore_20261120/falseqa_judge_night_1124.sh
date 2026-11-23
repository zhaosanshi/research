#!/bin/bash
# 2026-11-24 01 h, exploration (main session, Claude Fable 5.2): DeepSeek Pro judging of the FalseQA answers of the five
# models, development set first, then the held-out set (plan discussion/work_queue/2026-11-23_queue_l.md section 11).
# It starts only falseqa_judge.sh, unchanged (its own clock guard keeps judging out of peak hours). Judging is not a
# reading: no table is made here; the held-out reading (diag/falseqa_read.py table) waits for the external review.
# One lane per model, in parallel: dev, wait for its marker, heldout, wait for its marker.
# Status lines: "<date> falseqa judge night: ..." in Iteration_01/logs/newsets_20261119/falseqa_all.log; marker
# FALSEQA_JUDGE_NIGHT_1124_DONE there at the end.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/falseqa_judge_night_1124.sh > Iteration_01/logs/newsets_20261119/falseqa_judge_night_1124.out 2>&1 < /dev/null &
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
L=Iteration_01/logs/newsets_20261119
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') falseqa judge night: $*" >> $L/falseqa_all.log; }
[ -s $L/falseqa_judge.sh ] || { echo "missing $L/falseqa_judge.sh" >&2; exit 2; }
wait_marker() {  # wait_marker MODEL SET : until DONE or FAILED exists (at most 8 hours); prints DONE, FAILED or TIMEOUT
  local mk=$L/FALSEQA_JUDGE_${1}_$2 i
  for i in $(seq 1 480); do
    [ -e ${mk}_DONE ] && { echo DONE; return 0; }
    [ -e ${mk}_FAILED ] && { echo FAILED; return 0; }
    sleep 60
  done
  echo TIMEOUT
}
lane() {  # lane MODEL
  local m=$1 s r
  for s in dev heldout; do
    say "$m: $s judging starts"
    bash $L/falseqa_judge.sh $m $s >> $L/falseqa_judge_night_1124_$m.out 2>&1
    r=$(wait_marker $m $s)
    say "$m: $s judging ended: $r"
    [ "$r" = DONE ] || { say "$m: lane stops after $s ($r)"; return 1; }
  done
}
rm -f $L/FALSEQA_JUDGE_NIGHT_1124_DONE
say "start (pid $$): five lanes, dev then heldout"
for m in llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b; do
  lane $m &
done
wait
say "all lanes ended"
touch $L/FALSEQA_JUDGE_NIGHT_1124_DONE

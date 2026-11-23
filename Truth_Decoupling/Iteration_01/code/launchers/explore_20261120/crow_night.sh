#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor F, 2026-11-24 02 h; plan discussion/work_queue/2026-11-24_queue_m.md section 1):
# queues crow_gen.sh (unchanged per model) on cards 1 and 2. Lane card 1: llama, q9. Lane card 2: g27, g12. After both
# lanes: q27 on cards 1,2 (both locks). crow_gen.sh waits for each card's lock and for room itself; a model that fails
# is skipped and the lane goes on. Status lines in Iteration_01/logs/newsets_20261119/crow_all.log; marker
# CROW_NIGHT_DONE there at the end. No judging here.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/crow_night.sh > Iteration_01/logs/newsets_20261119/crow_night.out 2>&1 < /dev/null &
cd /data/sanshizhao/Truth_Decoupling || exit 2
L=Iteration_01/logs/newsets_20261119
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') crow night: $*" >> $L/crow_all.log; }
[ -s $L/crow_gen.sh ] || { echo "missing $L/crow_gen.sh" >&2; exit 2; }
rm -f $L/CROW_NIGHT_DONE
lane() {  # lane CARD KEY...
  local card=$1 k
  shift
  for k in "$@"; do
    say "$k: card $card"
    CROW_CARD=$card bash $L/crow_gen.sh $k
    say "$k: exit $?"
  done
}
say "start (pid $$)"
lane 1 llama q9 &
lane 2 g27 g12 &
wait
lane 1,2 q27
say "end"
touch $L/CROW_NIGHT_DONE

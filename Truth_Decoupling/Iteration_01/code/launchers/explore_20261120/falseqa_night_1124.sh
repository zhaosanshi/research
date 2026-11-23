#!/bin/bash
# 2026-11-24 00 h, exploration (main session, Claude Fable 5.2): the FalseQA generation and the batch-1 margins of the
# five models, queued on cards 1 and 2 (plan discussion/work_queue/2026-11-23_queue_l.md section 11). It starts only
# falseqa_model.sh (MODE full) and falseqa_score.sh (PHASE full), unchanged. No judge call and no table here: judging
# and the held-out reading wait for the external review of the pipeline.
# Lane card 2: g27, q27. Lane card 1: llama, q9, g12. Both launchers wait for the card's lock and for room themselves.
# A model whose generation fails is skipped (no margins) and the lane goes on.
# Status lines: "<date> falseqa night: ..." in Iteration_01/logs/newsets_20261119/falseqa_all.log; marker
# FALSEQA_NIGHT_1124_DONE there at the end.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/falseqa_night_1124.sh > Iteration_01/logs/newsets_20261119/falseqa_night_1124.out 2>&1 < /dev/null &
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
L=Iteration_01/logs/newsets_20261119
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') falseqa night: $*" >> $L/falseqa_all.log; }
name() {
  case $1 in
    llama) echo llama3.1-8b ;; q9) echo qwen3.5-9b ;; q27) echo qwen3.8-27b ;;
    g12) echo gemma3-12b ;; g27) echo gemma3-27b ;; *) return 1 ;;
  esac
}
for f in falseqa_model.sh falseqa_score.sh; do
  [ -s $L/$f ] || { echo "missing $L/$f" >&2; exit 2; }
done
lane() {  # lane CARD KEY...
  local card=$1 k m rc
  shift
  for k in "$@"; do
    m=$(name $k) || { say "unknown key $k"; continue; }
    say "$m: generation on card $card"
    bash $L/falseqa_model.sh $k $card full >> $L/falseqa_night_1124_$m.out 2>&1
    rc=$?
    if [ ! -e $L/FALSEQA_${m}_GPU_DONE ] || [ -e $L/FALSEQA_${m}_FAILED ]; then
      say "$m: generation FAILED (exit $rc); margins skipped"
      continue
    fi
    say "$m: generation done; margins on card $card"
    bash $L/falseqa_score.sh $m $card full >> $L/falseqa_night_1124_$m.out 2>&1
    say "$m: margins ended (exit $?)"
  done
  say "lane card $card ended"
}
rm -f $L/FALSEQA_NIGHT_1124_DONE
say "start (pid $$): card 2: g27 q27; card 1: llama q9 g12"
lane 2 g27 q27 &
lane 1 llama q9 g12 &
wait
say "all lanes ended"
touch $L/FALSEQA_NIGHT_1124_DONE

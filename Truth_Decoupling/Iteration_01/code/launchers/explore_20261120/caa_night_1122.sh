#!/bin/bash
# 2026-11-22 20 h, exploration (Claude Opus 5.5): tonight's part of the CAA grid that costs no judge money
# (discussion/work_queue/2026-11-22_queue_f.md section 9 item 3): the acceptance of each model (caa_accept.sh) and, for
# every model whose acceptance holds (GPT's guard, scripts/newsets/caa_run_guard.py grid), caa_grid.sh MODE=grid: MMLU
# per layer, the early stop, tune answers, judge inputs and cells. No judge is called, nothing is chosen, no test answers.
# Unattended; every wait is written here:
#   card 1: Llama-3.1-8B and Qwen3.5-9B, one at a time. The two acceptances (minutes each) run as soon as .gpu1.lock is
#           free; the grids start only after Iteration_01/logs/newsets_20261119/H22_CARD1_DONE exists or Beijing time
#           2026-11-22 22:30 has passed, Llama first, then Qwen3.5-9B;
#   card 0: Gemma-3-12B (acceptance, then grid) once .gpu0.lock is free and the card has room;
#   cards 2 and 3: only after ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE (or _FAILED) exists; Gemma-3-27B on card 2,
#           Qwen3.8-27B on card 3 (acceptance, then grid).
# caa_accept.sh and caa_grid.sh wait for their card's lock themselves (once a minute) and exit 3 when the card has no
# room; exit 3 is retried here once a minute. Nothing is stopped. Before every start: nvidia-smi memory and the running
# .sh processes go to the log.
# Layers, template layer: Iteration_01/code/configs/newsets/caa_grid_20261122.json; multipliers 0.5 1 2 4; tag g1.
# Markers (Iteration_01/logs/newsets_20261119/): CAA_GRID_<MODEL>_DONE / _FAILED per model (_FAILED also when the
# acceptance fails), CAA_NIGHT_1122_DONE when every lane has ended. Log caa_night_1122.log; status lines in caa_all.log;
# the scripts' own output caa_accept_<MODEL>.out, caa_grid_<MODEL>.out; caa_grid.sh's log
# Iteration_01/baselines/caa/<MODEL>/caa_grid_g1.log.
# usage (host): (setsid nohup bash Iteration_01/logs/newsets_20261119/caa_night_1122.sh > Iteration_01/logs/newsets_20261119/caa_night_1122.out 2>&1 < /dev/null &)
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CFGJ=Iteration_01/code/configs/newsets/caa_grid_20261122.json
GUARD=Iteration_01/code/scripts/newsets/caa_run_guard.py
TAG=g1
MULTS="0.5 1 2 4"
LOG=$HOST/$L/caa_night_1122.log
CARD1_AFTER=$(date -d "2026-11-22 22:30 +0800" +%s) || exit 2
for f in $L/caa_accept.sh $L/caa_grid.sh $CFGJ $GUARD; do [ -s $f ] || { echo "missing $f" >&2; exit 2; }; done
say() { echo "$(date '+%F %T') caa night: $*" | tee -a $LOG >> $HOST/$L/caa_all.log; }
look() {
  { echo "--- $(date '+%F %T') before: $*"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    ps -eo pid,args | grep -E "\.sh " | grep -v grep; } >> $LOG 2>&1
}
field() {  # field MODEL KEY: one value (a list is printed space-separated)
  python3 -c '
import json, sys
v = json.load(open(sys.argv[1]))["models"][sys.argv[2]][sys.argv[3]]
print(" ".join(map(str, v)) if isinstance(v, list) else v)' $CFGJ "$1" "$2"
}
accepted() { python3 -B $GUARD grid "$1" "$(field $1 candidate_layers)" "$MULTS" >> $LOG 2>&1; }
mark() { rm -f $HOST/$L/CAA_GRID_$1_DONE $HOST/$L/CAA_GRID_$1_FAILED; touch $HOST/$L/CAA_GRID_$1_$2; }
accept() {  # accept MODEL GPU: caa_accept.sh until it does not exit 3; true when the guard accepts the result
  local m=$1 g=$2 tl rc
  if accepted $m; then say "$m: acceptance already valid (guard)"; return 0; fi
  tl=$(field $m template_layer) || return 1
  while :; do
    look "acceptance $m on GPU $g"
    say "$m: acceptance on GPU $g (E and template layer $tl, 2 batches)"
    bash $HOST/$L/caa_accept.sh $m $g $tl 2 $tl >> $HOST/$L/caa_accept_$m.out 2>&1
    rc=$?
    [ $rc -eq 3 ] || break
    sleep 60
  done
  if [ $rc -eq 0 ] && accepted $m; then say "$m: acceptance passed"; return 0; fi
  say "$m: acceptance FAILED (exit $rc); no grid"; mark $m FAILED; return 1
}
grid() {  # grid MODEL GPU
  local m=$1 g=$2 layers rc
  layers=$(field $m candidate_layers) || { mark $m FAILED; return 1; }
  rm -f $HOST/$L/CAA_GRID_${m}_DONE $HOST/$L/CAA_GRID_${m}_FAILED
  while :; do
    look "grid $m on GPU $g"
    say "$m: grid on GPU $g, layers $layers, multipliers $MULTS, tag $TAG"
    MODE=grid bash $HOST/$L/caa_grid.sh $m $g "$layers" "$MULTS" $TAG >> $HOST/$L/caa_grid_$m.out 2>&1
    rc=$?
    [ $rc -eq 3 ] || break
    sleep 60
  done
  if [ $rc -eq 0 ]; then say "$m: grid done"; mark $m DONE; else say "$m: grid FAILED (exit $rc)"; mark $m FAILED; fi
}
lane1() {
  local ok_l=0 ok_q=0
  accept llama3.1-8b 1 && ok_l=1
  accept qwen3.5-9b 1 && ok_q=1
  until [ -e $HOST/$L/H22_CARD1_DONE ] || [ "$(date +%s)" -ge "$CARD1_AFTER" ]; do sleep 60; done
  say "card 1: H22_CARD1_DONE $([ -e $HOST/$L/H22_CARD1_DONE ] && echo present || echo absent at 22:30)"
  [ $ok_l -eq 1 ] && grid llama3.1-8b 1
  [ $ok_q -eq 1 ] && grid qwen3.5-9b 1
  say "lane card 1 ended"
}
lane0() {
  accept gemma3-12b 0 && grid gemma3-12b 0
  say "lane card 0 ended"
}
lane23() {
  local P=$HOST/$L/ITI_FREE_gemma3-27b_faithful_seed42_endq
  until [ -e ${P}_DONE ] || [ -e ${P}_FAILED ]; do sleep 60; done
  say "cards 2 and 3: $(basename $(ls ${P}_DONE ${P}_FAILED 2>/dev/null | head -1)) present"
  ( accept gemma3-27b 2 && grid gemma3-27b 2; say "lane card 2 ended" ) &
  ( accept qwen3.8-27b 3 && grid qwen3.8-27b 3; say "lane card 3 ended" ) &
  wait
}
rm -f $HOST/$L/CAA_NIGHT_1122_DONE
say "start (pid $$)"
lane1 &
lane0 &
lane23 &
wait
say "all lanes ended"
touch $HOST/$L/CAA_NIGHT_1122_DONE

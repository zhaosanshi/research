#!/bin/bash
# 2026-11-22 20 h, exploration (Claude Opus 5.5): tonight's card plan for the CAA grid, second version (the coordinator's
# instruction of 20:1x, discussion/work_queue/2026-11-22_queue_f.md progress). Replaces the card-1 lane and the card-2/3
# lane of caa_night_1122.sh (those two waiting lanes were stopped; its card-0 lane, Gemma-3-12B, keeps running there).
# Reason: the Gemma-3-12B grid holds 43.5 GB on card 0 against an estimate of 26 GB; card 1 carries another user's vllm
# server (about 43.4 GB, never stopped), so a CAA grid no longer fits there. No judge is called; only acceptance and
# caa_grid.sh MODE=grid (MMLU per layer, early stop, tune answers, judge inputs and cells).
#   card 3: Llama-3.1-8B, then Qwen3.5-9B. Start when .gpu3.lock is free and Gemma-3-27B's ITI no longer needs card 3:
#           baselines/iti/gemma3-27b/faithful_seed42/choice.json and choice_all.json both exist with the same "stem"
#           (ITI then runs one GSM8K pass, on card 2), or ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE / _FAILED exists.
#           After Llama's first MMLU pass has run for a while its memory is read; the rule "Llama's use + Qwen3.5-9B's
#           estimate scaled by Gemma-3-12B's actual / estimate (43.5 / 26) <= 65 GB" is evaluated and logged. Both
#           caa_grid.sh runs take .gpu3.lock for their whole run, so a second grid on the same card waits for the
#           first one in any case: Qwen3.5-9B starts after Llama ends (see the log line of the rule).
#   card 2: Qwen3.8-27B (acceptance, then grid) after ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE / _FAILED.
#   Gemma-3-27B: on card 0 once CAA_GRID_gemma3-12b_DONE / _FAILED exists, or on card 3 once this script's Llama and
#           Qwen3.5-9B lane has ended, whichever comes first (acceptance, then grid).
# Every grid: the memory of its first MMLU pass is read (process and card) and logged; a CUDA out-of-memory error in the
# grid's log stops that grid (its own processes only) with CAA_GRID_<MODEL>_FAILED, no retry, no other batch size.
# Exit 3 of caa_accept.sh / caa_grid.sh (no room at the start) is retried once a minute. Nothing else is stopped.
# Before every start: nvidia-smi memory and the running .sh processes go to caa_night_1122.log.
# Markers (Iteration_01/logs/newsets_20261119/): CAA_GRID_<MODEL>_DONE / _FAILED, CAA_NIGHT2_1122_DONE at the end.
# usage (host): (setsid nohup bash Iteration_01/logs/newsets_20261119/caa_night2_1122.sh > Iteration_01/logs/newsets_20261119/caa_night2_1122.out 2>&1 < /dev/null &)
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CFGJ=Iteration_01/code/configs/newsets/caa_grid_20261122.json
GUARD=Iteration_01/code/scripts/newsets/caa_run_guard.py
SJ=Iteration_01/newsets/selfjudge
ITI=Iteration_01/baselines/iti/gemma3-27b/faithful_seed42
ITI_MK=$L/ITI_FREE_gemma3-27b_faithful_seed42_endq
TAG=g1
MULTS="0.5 1 2 4"
LIMIT=66560                      # 65 GB in MiB
SCALE_NUM=43500; SCALE_DEN=26000 # Gemma-3-12B grid: actual / estimate
LOG=$HOST/$L/caa_night_1122.log
for f in $L/caa_accept.sh $L/caa_grid.sh $CFGJ $GUARD; do [ -s $f ] || { echo "missing $f" >&2; exit 2; }; done
say() { echo "$(date '+%F %T') caa night2: $*" | tee -a $LOG >> $HOST/$L/caa_all.log; }
look() {
  { echo "--- $(date '+%F %T') before: $*"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    ps -eo pid,args | grep -E "\.sh " | grep -v grep; } >> $LOG 2>&1
}
field() {
  python3 -c '
import json, sys
v = json.load(open(sys.argv[1]))["models"][sys.argv[2]][sys.argv[3]]
print(" ".join(map(str, v)) if isinstance(v, list) else v)' $CFGJ "$1" "$2"
}
card_used() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$1" '$1 == g {print $2}'; }
lock_free() { flock -n $SJ/.gpu$1.lock true; }
done_mk() { [ -e $HOST/$L/CAA_GRID_$1_DONE ] || [ -e $HOST/$L/CAA_GRID_$1_FAILED ]; }
accepted() { python3 -B $GUARD grid "$1" "$(field $1 candidate_layers)" "$MULTS" >> $LOG 2>&1; }
mark() { rm -f $HOST/$L/CAA_GRID_$1_DONE $HOST/$L/CAA_GRID_$1_FAILED; touch $HOST/$L/CAA_GRID_$1_$2; }
accept() {  # accept MODEL GPU
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
watch_grid() {  # watch_grid MODEL GPU PID OFFSET: memory of the first MMLU pass; stop the grid on a CUDA OOM
  local m=$1 g=$2 pid=$3 off=$4 glog=$HOST/Iteration_01/baselines/caa/$1/caa_grid_$TAG.log mp="" t0="" peak=0 cpeak=0 n u c
  while kill -0 $pid 2>/dev/null; do
    if [ -f $glog ] && tail -c +$((off + 1)) $glog | grep -q -E "OutOfMemoryError|CUDA out of memory"; then
      say "$m: CUDA out of memory in $glog on GPU $g; stopping this grid (pid $pid and its children), no retry: $(tail -c +$((off + 1)) $glog | grep -m1 -E 'OutOfMemoryError|CUDA out of memory' | cut -c1-200)"
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null
      pkill -TERM -f "baselines/caa/$m/" 2>/dev/null
      touch $HOST/$L/CAA_GRID_${m}_OOM
      return 0
    fi
    if [ -z "$t0" ]; then
      mp=$(pgrep -f "^[^ ]*python[^ ]* .*cap_mcq_sweep.py run .*baselines/caa/$m/" | head -1)
      [ -n "$mp" ] && { t0=$(date +%s); say "$m: first MMLU pass started (pid $mp) on GPU $g"; }
    elif [ $(( $(date +%s) - t0 )) -le 900 ]; then
      u=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$mp" '$1 == p {print $2}')
      c=$(card_used $g)
      [[ $u =~ ^[0-9]+$ ]] && [ $u -gt $peak ] && peak=$u
      [[ $c =~ ^[0-9]+$ ]] && [ $c -gt $cpeak ] && cpeak=$c
      if [ $(( $(date +%s) - t0 )) -gt 840 ]; then
        say "$m: first MMLU pass, 15 min sampled every 30 s: process peak $peak MiB, card $g peak $cpeak MiB (estimate $(field $m need_mib) MiB)"
        echo "$m $g $peak $cpeak" > $HOST/$L/caa_night2_mem_$m.txt
        t0=done
      fi
    fi
    sleep 30
  done
}
grid() {  # grid MODEL GPU [after-start hook]
  local m=$1 g=$2 hook=${3:-} layers rc pid off glog
  layers=$(field $m candidate_layers) || { mark $m FAILED; return 1; }
  rm -f $HOST/$L/CAA_GRID_${m}_DONE $HOST/$L/CAA_GRID_${m}_FAILED $HOST/$L/CAA_GRID_${m}_OOM
  glog=$HOST/Iteration_01/baselines/caa/$m/caa_grid_$TAG.log
  while :; do
    look "grid $m on GPU $g"
    say "$m: grid on GPU $g, layers $layers, multipliers $MULTS, tag $TAG"
    off=$(stat -c %s $glog 2>/dev/null || echo 0)
    MODE=grid bash $HOST/$L/caa_grid.sh $m $g "$layers" "$MULTS" $TAG >> $HOST/$L/caa_grid_$m.out 2>&1 &
    pid=$!
    watch_grid $m $g $pid $off &
    local wpid=$!
    [ -n "$hook" ] && $hook &
    wait $pid; rc=$?
    wait $wpid
    [ -e $HOST/$L/CAA_GRID_${m}_OOM ] && { mark $m FAILED; say "$m: grid FAILED: CUDA out of memory (see above); waiting for the coordinator"; return 1; }
    [ $rc -eq 3 ] || break
    sleep 60
  done
  if [ $rc -eq 0 ]; then say "$m: grid done"; mark $m DONE; else say "$m: grid FAILED (exit $rc)"; mark $m FAILED; fi
}
iti_off_card3() {  # true when Gemma-3-27B's ITI no longer needs card 3
  [ -e ${ITI_MK}_DONE ] || [ -e ${ITI_MK}_FAILED ] && return 0
  [ -s $ITI/choice.json ] && [ -s $ITI/choice_all.json ] || return 1
  python3 -c '
import json, sys
a, b = (json.load(open(p)).get("stem") for p in sys.argv[1:3])
sys.exit(0 if a is not None and a == b else 1)' $ITI/choice.json $ITI/choice_all.json
}
qwen9_rule() {  # after Llama's first MMLU pass has been sampled: log the parallel rule
  local f=$HOST/$L/caa_night2_mem_llama3.1-8b.txt est
  until [ -s $f ] || done_mk llama3.1-8b; do sleep 60; done
  [ -s $f ] || { say "qwen3.5-9b: no memory reading of Llama (its grid ended first)"; return 0; }
  read -r _ _ peak cpeak < $f
  est=$(( $(field qwen3.5-9b need_mib) * SCALE_NUM / SCALE_DEN ))
  if [ $((cpeak + est)) -le $LIMIT ]; then
    say "qwen3.5-9b: rule: card 3 with Llama $cpeak MiB + Qwen3.5-9B estimate $est MiB = $((cpeak + est)) <= $LIMIT: it would fit, but caa_grid.sh holds .gpu3.lock for a whole run, so Qwen3.5-9B starts after Llama"
  else
    say "qwen3.5-9b: rule: card 3 with Llama $cpeak MiB + Qwen3.5-9B estimate $est MiB = $((cpeak + est)) > $LIMIT: Qwen3.5-9B starts after Llama"
  fi
}
lane3() {
  until iti_off_card3 && lock_free 3; do sleep 60; done
  say "card 3: Gemma-3-27B ITI no longer needs card 3 ($( [ -e ${ITI_MK}_DONE ] || [ -e ${ITI_MK}_FAILED ] && echo 'ITI marker present' || echo 'choice.json and choice_all.json have the same stem')), .gpu3.lock free"
  accepted llama3.1-8b && grid llama3.1-8b 3 qwen9_rule
  accepted qwen3.5-9b && grid qwen3.5-9b 3
  touch $HOST/$L/CAA_NIGHT2_CARD3_ENDED
  say "lane card 3 (Llama, Qwen3.5-9B) ended"
}
lane2() {
  until [ -e ${ITI_MK}_DONE ] || [ -e ${ITI_MK}_FAILED ]; do sleep 60; done
  say "card 2: $(basename $(ls ${ITI_MK}_DONE ${ITI_MK}_FAILED 2>/dev/null | head -1)) present"
  accept qwen3.8-27b 2 && grid qwen3.8-27b 2
  say "lane card 2 (Qwen3.8-27B) ended"
}
lane_g27() {
  local g=""
  until [ -n "$g" ]; do
    if done_mk gemma3-12b; then g=0; elif [ -e $HOST/$L/CAA_NIGHT2_CARD3_ENDED ]; then g=3; else sleep 60; fi
  done
  say "gemma3-27b: card $g is free of CAA work first"
  accept gemma3-27b $g && grid gemma3-27b $g
  say "lane Gemma-3-27B (card $g) ended"
}
rm -f $HOST/$L/CAA_NIGHT2_1122_DONE $HOST/$L/CAA_NIGHT2_CARD3_ENDED $HOST/$L/caa_night2_mem_*.txt
for m in llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-27b; do rm -f $HOST/$L/CAA_GRID_${m}_DONE $HOST/$L/CAA_GRID_${m}_FAILED $HOST/$L/CAA_GRID_${m}_OOM; done
say "start (pid $$)"
lane3 &
lane2 &
lane_g27 &
wait
say "all lanes ended"
touch $HOST/$L/CAA_NIGHT2_1122_DONE

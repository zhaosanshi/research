#!/bin/bash
# 2026-11-23 23 h, exploration (Claude Opus 5.5, executor B): follow-up starts of the CAA baseline after the grids that
# are running now (plan discussion/work_queue/2026-11-23_queue_l.md section 7 item 1 and section 10.3, row B). Only
# reviewed scripts are started, unchanged; nothing here calls a judge or touches a card itself. Two lanes:
#   card 3: wait for Iteration_01/baselines/caa/qwen3.5-9b/CAA_grid_g1_DONE / _FAILED (polled once a minute).
#           DONE: caa_chain.sh qwen3.5-9b 3 "8 10 12 14 16 18 20 22 24" "0.5 1 2 4" g1 pospromptl15 pospromptl15 3
#           (with the peak-hour restarts below). FAILED: its chain is skipped.
#           Then, whatever the chain's outcome: caa_accept.sh qwen3.8-27b 3 28 2 28; if
#           baselines/caa/qwen3.8-27b/acceptance/ACCEPT_DONE exists afterwards:
#           caa_grid.sh qwen3.8-27b 3 "16 20 24 28 32 36 40 44 48" "0.5 1 2 4" g1.  (CAA_AFTER_Q27=0 skips both.)
#   card 0: wait for baselines/caa/gemma3-27b/CAA_grid_g1_DONE / _FAILED. DONE:
#           caa_chain.sh gemma3-27b 0 "16 20 24 28 32 36 40 44" "0.5 1 2 4" g1 pospromptl28 pospromptl28 0.
# Peak-hour restart: a chain that ends with CAA_CHAIN_<MODEL>_FAILED whose own FAILED line in caa_all.log (after its
# start) is a judging step ("a tune judging", "judging multiplier -1", "test judging") and whose every caa_judge.sh
# FAILED line for that model (after its start) is "peak hours or too close to them ..., judge not started" is started
# again, unchanged, in the next allowed window (weekdays 12:00-13:30 or from 18:00 on; any time at weekends; Beijing
# time), at most twice per chain. Any other failure is logged and left.
# Exit 3 of caa_accept.sh / caa_grid.sh (no room on the card at the start) is retried once a minute, as in
# caa_night2_1122.sh (at most 24 h).
# Arguments are checked at the start against code/configs/newsets/caa_grid_20261122.json (candidate_layers, like_tune,
# like_test, template_layer); a mismatch stops the launcher before anything is started.
# Status lines: "<date> caa after: ..." in Iteration_01/logs/newsets_20261119/caa_all.log. Outputs of the started scripts:
# caa_chain_<MODEL>.out, caa_accept_<MODEL>.out, caa_grid_<MODEL>.out there (appended). Marker at the end:
# CAA_AFTER_1123_DONE there.
# Dry run (no script is started): CAA_AFTER_DRY=1 CAA_AFTER_ROOT=<temp dir> [CAA_AFTER_STUB=<script>]
#   [CAA_AFTER_NOW=<epoch>]: markers and the log are read and written under <temp dir> (same relative paths), every
#   start is printed ("DRY: ...") and handed to the stub (which may write markers / log lines), the clock is simulated
#   (starting at CAA_AFTER_NOW, advanced by every wait instead of sleeping).
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/caa_after_1123.sh > Iteration_01/logs/newsets_20261119/caa_after_1123.out 2>&1 < /dev/null &
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling
DRY=${CAA_AFTER_DRY:-0}
Q27=${CAA_AFTER_Q27:-1}
MR=${CAA_AFTER_ROOT:-$HOST}
if [ "$DRY" = 1 ]; then
  [ "$MR" != "$HOST" ] || { echo "a dry run needs CAA_AFTER_ROOT (a temporary directory)" >&2; exit 2; }
  FAKE=${CAA_AFTER_NOW:-$(date +%s)}
else
  [ "$MR" = "$HOST" ] || { echo "CAA_AFTER_ROOT is for dry runs only" >&2; exit 2; }
fi
cd $HOST || exit 2
L=Iteration_01/logs/newsets_20261119
CAA=Iteration_01/baselines/caa
CFGJ=$HOST/Iteration_01/code/configs/newsets/caa_grid_20261122.json
ALL=$MR/$L/caa_all.log
MULTS="0.5 1 2 4"
TAG=g1
Q9_LAYERS="8 10 12 14 16 18 20 22 24"
Q27_LAYERS="16 20 24 28 32 36 40 44 48"
G27_LAYERS="16 20 24 28 32 36 40 44"
say() { echo "$(date '+%F %T') caa after: $*" >> $ALL; }
now() { if [ "$DRY" = 1 ]; then echo $FAKE; else date +%s; fi; }
tick() { if [ "$DRY" = 1 ]; then FAKE=$((FAKE + $1)); else sleep $1; fi; }
stamp() { TZ=Asia/Shanghai date -d "@$1" '+%F %H:%M %a'; }
window_wait() {  # seconds from epoch $1 to the next allowed judging window (0 = inside one)
  TZ=Asia/Shanghai python3 -c '
import sys, time
def allowed(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday >= 5 or 12 * 60 <= m < 13 * 60 + 30 or m >= 18 * 60
t0 = int(sys.argv[1]); t = t0
while not allowed(t) and t - t0 < 8 * 86400:
    t = (t // 60 + 1) * 60
print(t - t0)' "$1"
}
start() {  # start OUTFILE COMMAND... : run one reviewed launcher (dry run: print it and hand it to the stub)
  local out=$1
  shift
  if [ "$DRY" = 1 ]; then
    echo "DRY: $* >> $L/$out"
    say "DRY: $*"
    if [ -n "${CAA_AFTER_STUB:-}" ]; then bash "$CAA_AFTER_STUB" "$MR" "$@"; return $?; fi
    return 0
  fi
  "$@" >> $HOST/$L/$out 2>&1
}
field() {
  python3 -c '
import json, sys
v = json.load(open(sys.argv[1]))["models"][sys.argv[2]][sys.argv[3]]
print(" ".join(map(str, v)) if isinstance(v, list) else v)' $CFGJ "$1" "$2"
}
expect() {  # expect MODEL FIELD VALUE
  local got
  got=$(field $1 $2) || { say "FAILED: cannot read $2 of $1 from the config; nothing started"; exit 2; }
  [ "$got" = "$3" ] || { say "FAILED: $1 $2 is '$got' in the config, not '$3'; nothing started"; exit 2; }
}
for f in caa_chain.sh caa_grid.sh caa_accept.sh caa_judge.sh; do
  [ -s $HOST/$L/$f ] || { echo "missing $L/$f" >&2; exit 2; }
done
[ -e $ALL ] || { echo "missing $ALL" >&2; exit 2; }
expect qwen3.5-9b candidate_layers "$Q9_LAYERS"; expect qwen3.5-9b like_tune pospromptl15; expect qwen3.5-9b like_test pospromptl15
expect gemma3-27b candidate_layers "$G27_LAYERS"; expect gemma3-27b like_tune pospromptl28; expect gemma3-27b like_test pospromptl28
expect qwen3.8-27b candidate_layers "$Q27_LAYERS"; expect qwen3.8-27b template_layer 28
expect qwen3.8-27b like_tune pospromptl28; expect qwen3.8-27b like_test pospromptl28
peak_only() {  # peak_only MODEL OFFSET : the chain's failure after byte OFFSET of caa_all.log is a peak-hour refusal only
  tail -c +$(($2 + 1)) $ALL | python3 -c '
import re, sys
model = sys.argv[1]
lines = sys.stdin.read().splitlines()
chain = [l for l in lines if " caa chain %s: FAILED: " % model in l]
judge = [l for l in lines if re.search(r" caa judge %s \S+: FAILED: " % re.escape(model), l)]
ok = (bool(chain) and chain[-1].split(": FAILED: ", 1)[1] in ("a tune judging", "judging multiplier -1", "test judging")
      and bool(judge) and all(re.search(r": FAILED: peak hours or too close to them \(\d+s left\), judge not started$", l) for l in judge))
sys.exit(0 if ok else 1)' "$1"
}
last_fail() { tail -c +$(($2 + 1)) $ALL | grep -E " caa (chain|judge) $1[: ].*FAILED: " | tail -1 | cut -c1-200; }
chain() {  # chain MODEL GPU LAYERS LIKE_TUNE LIKE_TEST GSM_GPU : run caa_chain.sh, restart after peak-hour refusals only
  local m=$1 attempt off rc w mk=$MR/$L/CAA_CHAIN_$1
  for attempt in 1 2 3; do
    off=$(stat -c %s $ALL)
    say "$m: chain attempt $attempt: caa_chain.sh $m $2 \"$3\" \"$MULTS\" $TAG $4 $5 $6"
    start caa_chain_$m.out bash $HOST/$L/caa_chain.sh $m $2 "$3" "$MULTS" $TAG $4 $5 $6
    rc=$?
    if [ -e ${mk}_DONE ]; then say "$m: chain done (exit $rc)"; return 0; fi
    if [ ! -e ${mk}_FAILED ]; then say "$m: chain ended with exit $rc and no marker; left as it is"; return 1; fi
    if ! peak_only $m $off; then
      say "$m: chain FAILED (exit $rc; $(last_fail $m $off)); not a peak-hour refusal, left as it is"
      return 1
    fi
    if [ $attempt -eq 3 ]; then say "$m: chain FAILED again on a peak-hour refusal; two restarts used, left as it is"; return 1; fi
    w=$(window_wait $(now)) || { say "$m: cannot compute the next window; left as it is"; return 1; }
    say "$m: chain FAILED only because judging refused to start in or near peak hours; restart $attempt of 2 at $(stamp $(( $(now) + w )))"
    tick $w
    until [ "$(window_wait $(now))" = 0 ]; do tick 60; done
  done
}
retry3() {  # retry3 OUTFILE COMMAND... : exit 3 (no room at the start) retried once a minute, at most 24 h
  local i rc
  for i in $(seq 1 1440); do
    start "$@"
    rc=$?
    [ $rc -eq 3 ] || return $rc
    [ $i -eq 1 ] && say "$*: no room on the card at the start (exit 3); retrying once a minute"
    tick 60
  done
  return 3
}
wait_grid() {  # wait_grid MODEL : until the running grid wrote its marker; prints DONE or FAILED
  local mk=$MR/$CAA/$1/CAA_grid_${TAG}
  until [ -e ${mk}_DONE ] || [ -e ${mk}_FAILED ]; do tick 60; done
  if [ -e ${mk}_DONE ]; then echo DONE; else echo FAILED; fi
}
lane3() {
  local r rc
  r=$(wait_grid qwen3.5-9b)
  say "qwen3.5-9b: grid marker CAA_grid_${TAG}_$r"
  if [ "$r" = DONE ]; then
    chain qwen3.5-9b 3 "$Q9_LAYERS" pospromptl15 pospromptl15 3
  else
    say "qwen3.5-9b: grid FAILED, its chain skipped"
  fi
  if [ "$Q27" != 1 ]; then say "qwen3.8-27b: skipped (CAA_AFTER_Q27=$Q27)"; say "lane card 3 ended"; return 0; fi
  say "qwen3.8-27b: acceptance on GPU 3: caa_accept.sh qwen3.8-27b 3 28 2 28"
  retry3 caa_accept_qwen3.8-27b.out bash $HOST/$L/caa_accept.sh qwen3.8-27b 3 28 2 28
  rc=$?
  if [ ! -e $MR/$CAA/qwen3.8-27b/acceptance/ACCEPT_DONE ]; then
    say "qwen3.8-27b: acceptance FAILED (exit $rc, no ACCEPT_DONE); no grid"; say "lane card 3 ended"; return 1
  fi
  say "qwen3.8-27b: acceptance passed (exit $rc, ACCEPT_DONE '$(cat $MR/$CAA/qwen3.8-27b/acceptance/ACCEPT_DONE)'); grid on GPU 3"
  retry3 caa_grid_qwen3.8-27b.out env MODE=grid bash $HOST/$L/caa_grid.sh qwen3.8-27b 3 "$Q27_LAYERS" "$MULTS" $TAG
  rc=$?
  if [ -e $MR/$CAA/qwen3.8-27b/CAA_grid_${TAG}_DONE ]; then say "qwen3.8-27b: grid done (exit $rc)"
  else say "qwen3.8-27b: grid FAILED (exit $rc); left as it is"; fi
  say "lane card 3 ended"
}
lane0() {
  local r
  r=$(wait_grid gemma3-27b)
  say "gemma3-27b: grid marker CAA_grid_${TAG}_$r"
  if [ "$r" = DONE ]; then
    chain gemma3-27b 0 "$G27_LAYERS" pospromptl28 pospromptl28 0
  else
    say "gemma3-27b: grid FAILED, its chain not started"
  fi
  say "lane card 0 ended"
}
rm -f $MR/$L/CAA_AFTER_1123_DONE
say "start (pid $$, dry $DRY, qwen3.8-27b steps $Q27); waiting for the grid markers of qwen3.5-9b (card 3) and gemma3-27b (card 0)"
lane3 &
lane0 &
wait
say "all lanes ended"
touch $MR/$L/CAA_AFTER_1123_DONE

#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-18): DeepSeek Pro pass on the test questions of Gemma-3-12B / Gemma-3-27B that Pro
# has not judged yet (赵磊 2026-11-18: approved, ONLY in DeepSeek off-peak hours = half price). Thin wrapper: it only calls
# launchers/ff_tune_pro_20261117.sh (one lane per work dir, 4 lanes in parallel), seeded with the Pro reply cache of the
# same-named "dr100" dir (the Dr's 100 questions, already judged by Pro), so cells judged there are cache hits, not new calls.
# Peak (Beijing time): Mon-Fri 09:00-12:00 and 14:00-18:00; everything else (weekends all day) is off-peak.
# Lanes may run only from 30 s after an off-peak window opens until 5 min before the next peak (e.g. 12:00:30-13:55:00,
# 18:00:30-08:55:00 next weekday). At the stop time the lanes still running are stopped by their own process group
# (host side) and by their own work-dir command line (inside the container), then started again with the same command in
# the next window (resume: pro.jsonl + reply cache). A lane that fails by itself is not retried.
# Balance: queried in the container before the start, at every resumption and at the end; only the balance_infos JSON is
# logged, never the key. Start refused below 20 CNY (ALL.FAIL); a resumption refused below 5 CNY (a DeepSeek 402 stalls
# silently). After all 4 LANE.DONE: Iteration_01/diag/confirm_fresh.py, then ALL.DONE.
# usage (host): cd /data/sanshizhao/Truth_Decoupling/Iteration_01/launchers;
#   setsid nohup bash ff_g3_fresh_pro_offpeak_20261118.sh > ../results/fresh_pro_20261118/driver.log 2>&1 < /dev/null & disown
# Dry run (no judge call, no balance call, no stats; fake lanes, explicit windows, log dir elsewhere):
#   FRESH_DRYRUN=1 FRESH_DRY_DIR=/tmp/x FRESH_FAKE_INNER=/tmp/x/fake.sh FRESH_TEST_WINDOWS="s1-e1 s2-e2" (epoch seconds)
set -uo pipefail
export TZ=Asia/Shanghai
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
DRY=${FRESH_DRYRUN:-0}
if [ "$DRY" = 1 ]; then
  LH=${FRESH_DRY_DIR:?dry run needs FRESH_DRY_DIR}; INNER=${FRESH_FAKE_INNER:?dry run needs FRESH_FAKE_INNER}
  : "${FRESH_TEST_WINDOWS:?dry run needs FRESH_TEST_WINDOWS}"
  lanearg() { echo "$LH/lane_$1"; }
else
  LH=$HOST/Iteration_01/results/fresh_pro_20261118
  INNER=$HOST/Iteration_01/launchers/ff_tune_pro_20261117.sh
  lanearg() { echo "$ROOT/Iteration_01/results/fresh_pro_20261118/lane_$1"; }
fi
POLL=${FRESH_POLL:-20}
MIN_START=20; MIN_RESUME=5
say() { echo "$(date '+%F %T %Z') $*"; }
hm() { date -d "@$1" '+%F %T %a'; }
mkdir -p "$LH" || exit 2
exec 9> "$LH/driver.lock"
flock -n 9 || { say "another driver holds $LH/driver.lock: exit"; exit 2; }
[ -e "$LH/ALL.DONE" ] || [ -e "$LH/ALL.FAIL" ] && { say "ALL.DONE / ALL.FAIL exists in $LH already: refusing to run again"; exit 2; }
allfail() { say "ALL.FAIL: $*"; printf '%s\n%s\n' "$(date '+%F %T %Z')" "$*" > "$LH/ALL.FAIL"; exit 1; }

C=$ROOT/Iteration_01/results/ff/gemma3
NAMES=(g12_t25 g12_L22 g27_t31 g27_L26)
WORKS=("$C/12b/e44/gemma3-12b/eval_extra/judge_test_t25_test" "$C/12b/e44/gemma3-12b/judge_test_testD_E_L22"
       "$C/27b/e44/gemma3-27b/eval_extra/judge_test_t31_test" "$C/27b/e44/gemma3-27b/judge_test_testD_E_L26")
host() { echo "$HOST/${1#$ROOT/}"; }

# window NOW -> "RUN <stop epoch>" or "WAIT <start epoch>"
window() {
  python3 - "$1" "$DRY" "${FRESH_TEST_WINDOWS:-}" <<'PY'
import sys, time
now, dry, spec = int(sys.argv[1]), sys.argv[2] == "1", sys.argv[3]
if dry:
    for w in spec.split():
        s, e = map(int, w.split("-"))
        if now < s: print("WAIT", s); sys.exit()
        if now < e: print("RUN", e); sys.exit()
    print("NONE 0"); sys.exit()
def peak(t):
    lt = time.localtime(t)                      # TZ=Asia/Shanghai from the environment
    m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (540 <= m < 720 or 840 <= m < 1080)
def allowed(t):                                 # off-peak, >= 30 s after the peak ended, >= 5 min before the next one
    return not peak(t) and not peak(t - 30) and not peak(t + 300)
assert time.strftime("%z") == "+0800", "TZ is not Asia/Shanghai"
g = now - now % 30                              # all boundaries lie on the 30 s grid (UTC+8 is whole hours)
if allowed(now):
    for k in range(1, 4 * 24 * 120):
        if not allowed(g + 30 * k): print("RUN", g + 30 * k); sys.exit()
else:
    for k in range(1, 4 * 24 * 120):
        if allowed(g + 30 * k): print("WAIT", g + 30 * k); sys.exit()
sys.exit("no window boundary within 4 days")
PY
}

balance() {  # prints the balance_infos JSON only; the key is read inside the container and never printed
  docker exec -u 1000:1001 truth_dec python3 -c '
import json, urllib.request
k = open("/workspace/Truth_Decoupling/logs/.deepseek_key", encoding="utf-8").read().strip()
r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + k, "Accept": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=30))
print(json.dumps(d.get("balance_infos"), ensure_ascii=False, separators=(",", ":")))' 2>/dev/null
}
cny() { python3 -c 'import json,sys; print([b for b in json.loads(sys.argv[1]) if b.get("currency")=="CNY"][0]["total_balance"])' "$1" 2>/dev/null; }
check_balance() {  # check_balance FLOOR WHAT -> sets BAL_JSON / BAL; exits with ALL.FAIL below FLOOR or on error
  if [ "$DRY" = 1 ]; then BAL_JSON='[dry run: not queried]'; BAL=99; say "balance ($2): $BAL_JSON"; return; fi
  BAL_JSON=$(balance) || allfail "balance query failed ($2)"
  BAL=$(cny "$BAL_JSON") || allfail "balance reply has no CNY total ($2): $BAL_JSON"
  say "balance ($2): $BAL_JSON"
  python3 -c 'import sys; sys.exit(float(sys.argv[1]) < float(sys.argv[2]))' "$BAL" "$1" \
    || allfail "balance $BAL CNY < $1 CNY ($2): not started"
}

sleep_until() { local t=$1 n; while n=$(date +%s); [ "$n" -lt "$t" ]; do sleep $(( t - n < 60 ? t - n : 60 )); done; }

declare -a STATE PID RUNS REASON
for i in 0 1 2 3; do STATE[$i]=pending; PID[$i]=; RUNS[$i]=0; REASON[$i]=; done
lanedir() { echo "$LH/lane_${NAMES[$1]}"; }

start_lane() {
  local i=$1 d; d=$(lanedir "$i"); mkdir -p "$d"; RUNS[$i]=$(( RUNS[$i] + 1 ))
  setsid bash "$INNER" "$(lanearg "${NAMES[$i]}")" "${WORKS[$i]}@${WORKS[$i]}dr100/judge2/cache" \
    > "$d/driver_run${RUNS[$i]}.out" 2>&1 < /dev/null &
  PID[$i]=$!; STATE[$i]=running
  say "start lane ${NAMES[$i]} (run ${RUNS[$i]}, pid/pgid ${PID[$i]}): ${WORKS[$i]} seed ${WORKS[$i]}dr100/judge2/cache"
}

reap_lane() {  # the lane process exited by itself
  local i=$1 d rc; d=$(lanedir "$i"); wait "${PID[$i]}"; rc=$?
  if [ -e "$d/LANE.DONE" ]; then STATE[$i]=done; say "lane ${NAMES[$i]} LANE.DONE (rc $rc)"
  else STATE[$i]=failed; REASON[$i]="rc $rc, LANE.FAIL: $(cat "$d/LANE.FAIL" 2>/dev/null || echo '<none>')"
       say "lane ${NAMES[$i]} FAILED (not retried): ${REASON[$i]}"; fi
}

stop_lane() {  # end of the window: stop only this lane's processes, keep it pending for the next window
  local i=$1 W=${WORKS[$1]} d k WH; d=$(lanedir "$i"); WH=$(host "$W")
  say "stop lane ${NAMES[$i]} (pgid ${PID[$i]}): end of the off-peak window"
  kill -TERM -- "-${PID[$i]}" 2>/dev/null
  local pat="review_checks/(judge_two_axis_run\.py (llm|single) $W --judge pro|judge2_cache_merge\.py $W/judge2/cache )"
  for k in $(seq 1 30); do
    [ "$DRY" = 1 ] || docker exec truth_dec pkill -TERM -f "$pat" > /dev/null 2>&1
    kill -0 -- "-${PID[$i]}" 2>/dev/null && { sleep 2; continue; }
    [ "$DRY" = 1 ] || ! docker exec truth_dec pgrep -f "$pat" > /dev/null 2>&1 || { sleep 2; continue; }
    break
  done
  if kill -0 -- "-${PID[$i]}" 2>/dev/null || { [ "$DRY" != 1 ] && docker exec truth_dec pgrep -f "$pat" > /dev/null 2>&1; }; then
    say "lane ${NAMES[$i]} still alive after 60 s: SIGKILL"
    kill -KILL -- "-${PID[$i]}" 2>/dev/null; [ "$DRY" = 1 ] || docker exec truth_dec pkill -KILL -f "$pat" > /dev/null 2>&1
    sleep 2
  fi
  wait "${PID[$i]}" 2>/dev/null
  if [ -e "$d/LANE.DONE" ]; then STATE[$i]=done; say "lane ${NAMES[$i]} had finished just before the stop: LANE.DONE"; return; fi
  rm -f "$d/LANE.FAIL"        # a marker written while being stopped is not a failure of the lane (it clears it itself anyway)
  if [ "$DRY" != 1 ]; then
    if [ -e "$WH/judge2/single_pro.jsonl" ]; then
      STATE[$i]=failed; REASON[$i]="stopped during 'single': $WH/judge2/single_pro.jsonl exists, the lane would refuse to restart"
      say "lane ${NAMES[$i]} FAILED: ${REASON[$i]}"; return; fi
    if ! python3 -c '
import json, os, sys
p = sys.argv[1]
if os.path.exists(p):
    for n, l in enumerate(open(p, encoding="utf-8"), 1):
        if l.strip(): json.loads(l)' "$WH/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl" 2>/dev/null; then
      STATE[$i]=failed; REASON[$i]="reply cache unreadable after the stop (not repaired): $WH/judge2/cache"
      say "lane ${NAMES[$i]} FAILED: ${REASON[$i]}"; return; fi
  fi
  STATE[$i]=pending; say "lane ${NAMES[$i]} stopped, resumes in the next window"
}

count() { local s=$1 n=0 i; for i in 0 1 2 3; do [ "${STATE[$i]}" = "$s" ] && n=$((n+1)); done; echo $n; }

say "driver pid $$ (dry=$DRY), log dir $LH, inner $INNER"
for i in 0 1 2 3; do
  WH=$(host "${WORKS[$i]}"); SH=$(host "${WORKS[$i]}dr100")
  [ -f "$WH/judge2/cells.jsonl" ] && [ -f "$WH/judge2/single_allenai.jsonl" ] || allfail "${NAMES[$i]}: no cells / AllenAI pass in $WH"
  [ -e "$WH/judge2/single_pro.jsonl" ] && allfail "${NAMES[$i]}: a Pro pass exists already in $WH"
  [ -f "$SH/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl" ] || allfail "${NAMES[$i]}: no Pro reply cache in the seed $SH"
done
say "pre-checks passed (4 work dirs without a Pro pass, 4 seed caches present)"

FIRST=1
while [ "$(count pending)" -gt 0 ]; do
  read -r MODE T <<< "$(window "$(date +%s)")" || allfail "window computation failed"
  case $MODE in
    WAIT) say "等到 $(hm "$T") (off-peak) 再$([ $FIRST = 1 ] && echo 启动 || echo 续判) lanes: $(for i in 0 1 2 3; do [ "${STATE[$i]}" = pending ] && printf '%s ' "${NAMES[$i]}"; done)"
          sleep_until "$T"; continue ;;
    RUN)  ;;
    *)    allfail "window computation gave '$MODE $T' (dry-run windows exhausted, or no boundary found) with $(count pending) lanes pending" ;;
  esac
  if [ $FIRST = 1 ]; then check_balance $MIN_START "before the start"; BAL0=$BAL_JSON
  else check_balance $MIN_RESUME "before resuming"; fi
  FIRST=0
  say "off-peak window open, lanes may run until $(hm "$T")"
  for i in 0 1 2 3; do [ "${STATE[$i]}" = pending ] && start_lane "$i"; done
  while [ "$(count running)" -gt 0 ]; do
    for i in 0 1 2 3; do
      [ "${STATE[$i]}" = running ] && ! kill -0 "${PID[$i]}" 2>/dev/null && reap_lane "$i"
    done
    [ "$(count running)" -gt 0 ] || break
    if [ "$(date +%s)" -ge "$T" ]; then
      for i in 0 1 2 3; do [ "${STATE[$i]}" = running ] && stop_lane "$i"; done
      break
    fi
    sleep "$POLL"
  done
  say "window over: done $(count done), failed $(count failed), pending $(count pending)"
done

check_balance 0 "after the end"; BAL1=$BAL_JSON
if [ "$(count failed)" -gt 0 ]; then
  allfail "$(for i in 0 1 2 3; do [ "${STATE[$i]}" = failed ] && printf 'lane %s (%s): %s; ' "${NAMES[$i]}" "${WORKS[$i]}" "${REASON[$i]}"; done) balance before $BAL0, after $BAL1"
fi
if [ "$DRY" = 1 ]; then RC=dry-run-skipped; else
  say "all 4 lanes LANE.DONE: run Iteration_01/diag/confirm_fresh.py"
  (cd "$HOST" && python3 -B Iteration_01/diag/confirm_fresh.py > Iteration_01/diag/out/confirm_fresh_20261118.txt 2>&1); RC=$?
  say "confirm_fresh.py rc $RC -> $HOST/Iteration_01/diag/out/confirm_fresh_20261118.txt"
  [ "$RC" = 0 ] || say "STATS SCRIPT FAILED (rc $RC); the judging itself is complete"
fi
printf 'end %s\nlanes: 4/4 LANE.DONE (%s)\nconfirm_fresh.py rc: %s\nbalance before: %s\nbalance after: %s\n' \
  "$(date '+%F %T %Z')" "$(for i in 0 1 2 3; do printf '%s runs=%s ' "${NAMES[$i]}" "${RUNS[$i]}"; done)" "$RC" "$BAL0" "$BAL1" > "$LH/ALL.DONE"
say "ALL.DONE"

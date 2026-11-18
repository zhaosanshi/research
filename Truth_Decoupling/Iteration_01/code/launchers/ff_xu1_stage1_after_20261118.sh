#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-18): the stronger-caution runs of 11-18 (results/ff_xu1: Gemma-3-12B, Gemma-3-27B,
# Gemma-4-31B, started by ff_xu1_stage1_20261118.sh), once each run's test chain v3 has ended: a DeepSeek Pro pass on
#   (1) the tune E+C grid  e44/<model>/eval_extra/judge_tune_ec<C>_tune   (launchers/ff_tune_pro_20261117.sh, no seed), and
#   (2) the Dr's 100 test questions of the test E+C grid e44/<model>/eval_extra/judge_test_t<C>_test, as in
#       ff_gemma3_27b_after_v3.sh: a 3-question trial (suffix tr3), then dr100 seeded with the trial's reply cache
#       (code/scripts/review_checks/ff_test_subset_judge.sh). Its AllenAI half runs first, on the run's own card, any time.
# Only these two kinds: no must-answer (judge_*_D_*, testD) and no C-alone (testS) work dir this time.
# C is the run's own: the dirs are read from disk (exactly one judge_tune_ec<N>_tune and one judge_test_t<N>_test, the same
# N, equal to the "R6 v3 C<N>" line of the chain log); anything else = that run fails and the log says what was seen.
# A run whose v3 chain ends with any .FAIL ("chain end: .. N failed or not tried", N > 0) is not judged (run fails).
# DeepSeek only off-peak (赵磊: half price). Peak (Beijing): Mon-Fri 09:00-12:00, 14:00-18:00. Pro lanes run from 30 s after
# a peak until 5 min before the next one (12:00:30-13:55, 18:00:30-08:55, weekends); still running at the stop time ->
# stopped by their process group (host) and by their own work-dir command line (container), resumed in the next window.
# Window, stop and balance code as ff_g3_fresh_pro_offpeak_20261118.sh (whose lanes are other work dirs: not touched).
# Balance before every window of a run (only the balance_infos JSON is logged, never the key): below 5 CNY -> STOP for all
# runs, ALL.FAIL. End: ALL.DONE (each run's two Pro work dirs, balances) or ALL.FAIL (what failed).
# usage (host): cd /data/sanshizhao/Truth_Decoupling/Iteration_01/launchers;
#   setsid nohup bash ff_xu1_stage1_after_20261118.sh > ../results/xu1_stage1_after_20261118/driver.log 2>&1 < /dev/null & disown
#   bash ff_xu1_stage1_after_20261118.sh find <host run root> <model>     (read-only: prints the two dirs it would judge)
# Dry run (no judge, no balance call, no GPU): XU_DRYRUN=1 XU_DRY_DIR=<log dir> XU_RES=<fake results dir, host = container>
#   XU_RUNS="name|model|subpath|gpu ..." XU_FAKE=<fake lane / allenai script> XU_TEST_WINDOWS="s1-e1 s2-e2" (epoch seconds)
set -uo pipefail
export TZ=Asia/Shanghai
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
SELF=$(readlink -f "$0")
DRY=${XU_DRYRUN:-0}
say() { echo "$(date '+%F %T %Z') $*"; }
hm() { date -d "@$1" '+%F %T %a'; }

# ---------- find the two grid dirs of one run (read-only) ----------
# find_dirs HOST_RUN_ROOT MODEL -> prints "C TUNE_DIR_NAME TEST_DIR_NAME" or the reason on stderr, rc 1
find_dirs() {
  python3 - "$1" "$2" <<'PY'
import os, re, sys
root, model = sys.argv[1:3]
ex = os.path.join(root, "e44", model, "eval_extra")
log = os.path.join(root, "logs", "chain_ec_v3", "chain_jobs.log")
if not os.path.isdir(ex): sys.exit("no dir %s" % ex)
names = sorted(os.listdir(ex))
def grab(pat):
    return [(n, m.group(1)) for n in names for m in [re.fullmatch(pat, n)] if m]
tune, test = grab(r"judge_tune_ec([0-9]+)_tune"), grab(r"judge_test_t([0-9]+)_test")
seen = "seen in %s: tune %s, test %s (all judge_* there: %s)" % (ex, [n for n, _ in tune], [n for n, _ in test],
                                                                 [n for n in names if n.startswith("judge_")])
if len(tune) != 1 or len(test) != 1: sys.exit("not exactly one tune E+C grid and one test E+C grid; " + seen)
if tune[0][1] != test[0][1]: sys.exit("tune and test grids have different C layers; " + seen)
r6 = re.findall(r" R6 v3 C([0-9]+) ", open(log, encoding="utf-8").read()) if os.path.exists(log) else []
if not r6: sys.exit("no 'R6 v3 C<N>' line in %s; " % log + seen)
if r6[-1] != tune[0][1]: sys.exit("the chain chose C%s (last R6 line of %s) but the grid dirs say C%s; " % (r6[-1], log, tune[0][1]) + seen)
print(tune[0][1], tune[0][0], test[0][0])
PY
}

if [ "${1:-}" = find ]; then find_dirs "$2" "$3"; exit $?; fi

# ---------- one Pro lane on the Dr's subset of the test grid (child mode; same steps as ff_gemma3_27b_after_v3.sh) ----------
# testlane LANE_DIR(host) MODEL RUN_EVAL_EXTRA(container) NAME TRIAL_SUBSET(container)
if [ "${1:-}" = testlane ]; then
  LD=$2; M=$3; RUNC=$4; TN=$5; SUBC=$6
  mkdir -p "$LD"; rm -f "$LD/LANE.DONE" "$LD/LANE.FAIL"
  lfail() { say "$*"; echo "$*" > "$LD/LANE.FAIL"; exit 1; }
  cd "$HOST/Iteration_01/code/scripts/review_checks" || lfail "no code dir"
  say "pro: trial (3 questions) $RUNC/judge_test_${TN}tr3"
  env -u CACHE_FROM -u GPU MODEL="$M" CODE_PATH=$ROOT/Iteration_01/code RUN="$RUNC" NAMES="$TN" SUFFIX=tr3 SUBSET="$SUBC" \
    JUDGES=pro bash ff_test_subset_judge.sh < /dev/null || lfail "pro trial failed"
  say "pro: dr100 $RUNC/judge_test_${TN}dr100 (cache seeded from the trial)"
  env -u SUBSET -u GPU MODEL="$M" CODE_PATH=$ROOT/Iteration_01/code RUN="$RUNC" NAMES="$TN" SUFFIX=dr100 JUDGES=pro \
    CACHE_FROM="$RUNC/judge_test_${TN}tr3" bash ff_test_subset_judge.sh < /dev/null || lfail "pro dr100 failed"
  say "lane done"; touch "$LD/LANE.DONE"; exit 0
fi

# ---------- driver ----------
if [ "$DRY" = 1 ]; then
  LH=${XU_DRY_DIR:?dry run needs XU_DRY_DIR}; FAKE=${XU_FAKE:?dry run needs XU_FAKE}; : "${XU_TEST_WINDOWS:?dry run needs XU_TEST_WINDOWS}"
  RES_H=${XU_RES:?dry run needs XU_RES}; RES_C=$RES_H; LC=$LH
  read -r -a RUNS <<< "${XU_RUNS:?dry run needs XU_RUNS}"
else
  LH=$HOST/Iteration_01/results/xu1_stage1_after_20261118; LC=$ROOT/Iteration_01/results/xu1_stage1_after_20261118
  RES_H=$HOST/Iteration_01/results; RES_C=$ROOT/Iteration_01/results
  RUNS=("g3_12b|gemma3-12b|ff_xu1/gemma3/12b|0" "g3_27b|gemma3-27b|ff_xu1/gemma3/27b|1" "g4_31b|gemma4-31b-dm|ff_xu1/gemma4/31b|2")
fi
TRIAL_SRC=$HOST/Iteration_01/results/ff/gemma4/12b/logs/dr100_ec/trial3.json
POLL=${XU_POLL:-20}; CHAIN_POLL=${XU_CHAIN_POLL:-60}; GPU_WAIT=${XU_GPU_WAIT:-240}
MIN_BAL=5
mkdir -p "$LH" || exit 2
exec 9> "$LH/driver.lock"
flock -n 9 || { say "another driver holds $LH/driver.lock: exit"; exit 2; }
{ [ -e "$LH/ALL.DONE" ] || [ -e "$LH/ALL.FAIL" ]; } && { say "ALL.DONE / ALL.FAIL exists in $LH already: refusing to run again"; exit 2; }
rm -f "$LH/STOP"

window() {  # NOW -> "RUN <stop epoch>" | "WAIT <start epoch>" (as ff_g3_fresh_pro_offpeak_20261118.sh)
  python3 - "$1" "$DRY" "${XU_TEST_WINDOWS:-}" <<'PY'
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
g = now - now % 30
if allowed(now):
    for k in range(1, 4 * 24 * 120):
        if not allowed(g + 30 * k): print("RUN", g + 30 * k); sys.exit()
else:
    for k in range(1, 4 * 24 * 120):
        if allowed(g + 30 * k): print("WAIT", g + 30 * k); sys.exit()
sys.exit("no window boundary within 4 days")
PY
}
balance() {  # the balance_infos JSON only; the key is read inside the container and never printed
  docker exec -u 1000:1001 truth_dec python3 -c '
import json, urllib.request
k = open("/workspace/Truth_Decoupling/logs/.deepseek_key", encoding="utf-8").read().strip()
r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + k, "Accept": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=30))
print(json.dumps(d.get("balance_infos"), ensure_ascii=False, separators=(",", ":")))' 2>/dev/null
}
cny() { python3 -c 'import json,sys; print([b for b in json.loads(sys.argv[1]) if b.get("currency")=="CNY"][0]["total_balance"])' "$1" 2>/dev/null; }
sleep_until() { local t=$1 n; while n=$(date +%s); [ "$n" -lt "$t" ]; do [ -e "$LH/STOP" ] && return 1; sleep $(( t - n < 60 ? t - n : 60 )); done; }
gpu_free() { [ "$DRY" = 1 ] && return 0; local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1") || return 1
  [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")" = 0 ]; }

worker() {  # worker INDEX: everything for one run; writes RUN.DONE / RUN.FAIL in its dir
  local NAME M SUB GPU; IFS='|' read -r NAME M SUB GPU <<< "${RUNS[$1]}"
  local RL=$LH/run_$NAME RLC=$LC/run_$NAME RH=$RES_H/$SUB
  local VLOG=$RH/logs/chain_ec_v3 EXH=$RES_H/$SUB/e44/$M/eval_extra EXC=$RES_C/$SUB/e44/$M/eval_extra
  mkdir -p "$RL"; rm -f "$RL/RUN.DONE" "$RL/RUN.FAIL"
  rfail() { say "[$NAME] RUN FAILED: $*"; echo "$*" > "$RL/RUN.FAIL"; exit 1; }
  stopped() { [ -e "$LH/STOP" ]; }
  say "[$NAME] waiting for the v3 chain of $M: $VLOG (lock .lock_chain gone and last line 'chain end:')"
  while ! { [ ! -d "$VLOG/.lock_chain" ] && tail -n 1 "$VLOG/chain_jobs.log" 2>/dev/null | grep -q 'chain end:'; }; do
    stopped && rfail "global STOP before its chain ended: $(cat "$LH/STOP")"; sleep "$CHAIN_POLL"; done
  local END; END=$(tail -n 1 "$VLOG/chain_jobs.log"); say "[$NAME] v3 chain ended: $END"
  local NF; NF=$(sed -n 's/.*chain end: [0-9]* done, \([0-9]*\) failed or not tried.*/\1/p' <<< "$END")
  [ "$NF" = 0 ] && [ -z "$(ls "$VLOG" | grep '\.FAIL$')" ] \
    || rfail "the v3 chain ended with failed or untried steps (not judged): '$END'; FAIL files: $(ls "$VLOG" | grep '\.FAIL$' | tr '\n' ' ')"
  local F; F=$(find_dirs "$RH" "$M" 2>&1) || rfail "grid dirs not found: $F"
  local CL TUNE TEST; read -r CL TUNE TEST <<< "$F"; local TN=${TEST#judge_test_}
  local TUW=$EXC/$TUNE TUH=$EXH/$TUNE TRW=$EXC/${TEST}tr3 DRW=$EXC/${TEST}dr100 TRH=$EXH/${TEST}tr3 DRH=$EXH/${TEST}dr100
  say "[$NAME] C$CL: tune grid $TUH, test grid $EXH/$TEST -> Pro on $TUH and $DRH (trial $TRH)"
  [ -f "$TUH/judge2/cells.jsonl" ] && [ -f "$TUH/judge2/single_allenai.jsonl" ] || rfail "$TUH has no cells / AllenAI pass"
  [ -e "$TUH/judge2/single_pro.jsonl" ] && rfail "$TUH has a Pro pass already"
  [ -f "$EXH/$TEST/judge_input.json" ] || rfail "no $EXH/$TEST/judge_input.json"
  [ -e "$TRH" ] && rfail "$TRH exists already"; [ -e "$DRH" ] && rfail "$DRH exists already"
  [ -e "$EXH/frozen_test.json" ] || ln -s ../frozen_test.json "$EXH/frozen_test.json" || rfail "cannot link frozen_test.json"
  cp "$TRIAL_SRC" "$RL/trial3.json" || rfail "cannot copy $TRIAL_SRC"
  # AllenAI half of the trial and of dr100 (GPU, no DeepSeek; any time)
  local n=0; while ! gpu_free "$GPU" && [ $n -lt "$GPU_WAIT" ]; do sleep 30; n=$((n + 1)); done
  gpu_free "$GPU" || rfail "GPU $GPU still busy after $((GPU_WAIT / 2)) min"
  say "[$NAME] AllenAI on GPU $GPU: trial + dr100 of $TN"
  if [ "$DRY" = 1 ]; then bash "$FAKE" allenai "$RL" > "$RL/allenai.log" 2>&1 < /dev/null || rfail "fake allenai failed"
  else
    ( cd "$HOST/Iteration_01/code/scripts/review_checks" &&
      env -u CACHE_FROM MODEL="$M" CODE_PATH=$ROOT/Iteration_01/code RUN="$EXC" NAMES="$TN" SUFFIX=tr3 SUBSET="$RLC/trial3.json" \
        JUDGES=allenai GPU="$GPU" bash ff_test_subset_judge.sh > "$RL/allenai_tr3.log" 2>&1 < /dev/null ) || rfail "AllenAI trial failed ($RL/allenai_tr3.log)"
    ( cd "$HOST/Iteration_01/code/scripts/review_checks" &&
      env -u CACHE_FROM -u SUBSET MODEL="$M" CODE_PATH=$ROOT/Iteration_01/code RUN="$EXC" NAMES="$TN" SUFFIX=dr100 \
        JUDGES=allenai GPU="$GPU" bash ff_test_subset_judge.sh > "$RL/allenai_dr100.log" 2>&1 < /dev/null ) || rfail "AllenAI dr100 failed ($RL/allenai_dr100.log)"
  fi
  say "[$NAME] AllenAI done"
  # Pro: two lanes (tune grid; trial then dr100), off-peak only
  local LN=(tune test) ST=(pending pending) PID=("" "") RUNS_=(0 0) WHY=("" "") B0="" B1="" FIRST=1 MODE T i
  local PAT=("review_checks/(judge_two_axis_run\.py (llm|single) $TUW --judge pro|judge2_cache_merge\.py $TUW/judge2/cache )"
             "review_checks/judge_two_axis_run\.py (llm|single) ($TRW|$DRW) --judge pro")
  local CACHES=("$TUH/judge2/cache" "$TRH/judge2/cache $DRH/judge2/cache")
  cnt() { local s=$1 k c=0; for k in 0 1; do [ "${ST[$k]}" = "$s" ] && c=$((c+1)); done; echo $c; }
  start_lane() {
    local k=$1 d=$RL/lane_${LN[$1]} dc=$RLC/lane_${LN[$1]}; mkdir -p "$d"; RUNS_[$k]=$(( RUNS_[$k] + 1 ))
    if [ "$DRY" = 1 ]; then setsid bash "$FAKE" lane "$d" > "$d/driver_run${RUNS_[$k]}.out" 2>&1 < /dev/null 9>&- &
    elif [ "$k" = 0 ]; then setsid bash "$HOST/Iteration_01/launchers/ff_tune_pro_20261117.sh" "$dc" "$TUW" > "$d/driver_run${RUNS_[$k]}.out" 2>&1 < /dev/null 9>&- &
    else setsid bash "$SELF" testlane "$d" "$M" "$EXC" "$TN" "$RLC/trial3.json" > "$d/driver_run${RUNS_[$k]}.out" 2>&1 < /dev/null 9>&- &
    fi
    PID[$k]=$!; ST[$k]=running; say "[$NAME] start lane ${LN[$k]} (run ${RUNS_[$k]}, pid/pgid ${PID[$k]})"
  }
  reap_lane() {
    local k=$1 d=$RL/lane_${LN[$1]} rc; wait "${PID[$k]}"; rc=$?
    if [ -e "$d/LANE.DONE" ]; then ST[$k]=done; say "[$NAME] lane ${LN[$k]} LANE.DONE (rc $rc)"
    else ST[$k]=failed; WHY[$k]="rc $rc, LANE.FAIL: $(cat "$d/LANE.FAIL" 2>/dev/null || echo '<none>')"; say "[$NAME] lane ${LN[$k]} FAILED: ${WHY[$k]}"; fi
  }
  stop_lane() {  # as ff_g3_fresh_pro_offpeak_20261118.sh: own process group, then own work-dir command lines in the container
    local k=$1 d=$RL/lane_${LN[$1]} j c pat=${PAT[$1]}
    say "[$NAME] stop lane ${LN[$k]} (pgid ${PID[$k]}): $2"
    kill -TERM -- "-${PID[$k]}" 2>/dev/null
    for j in $(seq 1 30); do
      [ "$DRY" = 1 ] || docker exec truth_dec pkill -TERM -f "$pat" > /dev/null 2>&1
      kill -0 -- "-${PID[$k]}" 2>/dev/null && { sleep 2; continue; }
      [ "$DRY" = 1 ] || ! docker exec truth_dec pgrep -f "$pat" > /dev/null 2>&1 || { sleep 2; continue; }
      break
    done
    if kill -0 -- "-${PID[$k]}" 2>/dev/null || { [ "$DRY" != 1 ] && docker exec truth_dec pgrep -f "$pat" > /dev/null 2>&1; }; then
      say "[$NAME] lane ${LN[$k]} still alive after 60 s: SIGKILL"
      kill -KILL -- "-${PID[$k]}" 2>/dev/null; [ "$DRY" = 1 ] || docker exec truth_dec pkill -KILL -f "$pat" > /dev/null 2>&1; sleep 2
    fi
    wait "${PID[$k]}" 2>/dev/null
    if [ -e "$d/LANE.DONE" ]; then ST[$k]=done; say "[$NAME] lane ${LN[$k]} had finished just before the stop: LANE.DONE"; return; fi
    rm -f "$d/LANE.FAIL"
    if [ "$DRY" != 1 ]; then
      if [ "$k" = 0 ] && [ -e "$TUH/judge2/single_pro.jsonl" ]; then
        ST[$k]=failed; WHY[$k]="stopped during 'single': $TUH/judge2/single_pro.jsonl exists, ff_tune_pro_20261117.sh would refuse to restart"
        say "[$NAME] lane ${LN[$k]} FAILED: ${WHY[$k]}"; return; fi
      for c in ${CACHES[$k]}; do
        python3 -c '
import json, os, sys
p = sys.argv[1]
if os.path.exists(p):
    for l in open(p, encoding="utf-8"):
        if l.strip(): json.loads(l)' "$c/judge2_cache_deepseek-v4-pro.jsonl" 2>/dev/null || {
          ST[$k]=failed; WHY[$k]="reply cache unreadable after the stop (not repaired): $c"; say "[$NAME] lane ${LN[$k]} FAILED: ${WHY[$k]}"; return; }
      done
    fi
    ST[$k]=pending; say "[$NAME] lane ${LN[$k]} stopped, resumes in the next window"
  }
  while [ "$(cnt pending)" -gt 0 ]; do
    stopped && rfail "global STOP: $(cat "$LH/STOP")"
    read -r MODE T <<< "$(window "$(date +%s)")" || rfail "window computation failed"
    case $MODE in
      WAIT) say "[$NAME] 等到 $(hm "$T") (off-peak) 再$([ $FIRST = 1 ] && echo 启动 || echo 续判) Pro lanes: $(for i in 0 1; do [ "${ST[$i]}" = pending ] && printf '%s ' "${LN[$i]}"; done)"
            sleep_until "$T" || rfail "global STOP while waiting: $(cat "$LH/STOP")"; continue ;;
      RUN) ;;
      *) rfail "window computation gave '$MODE $T' with $(cnt pending) lanes pending" ;;
    esac
    local BJ BAL
    if [ "$DRY" = 1 ]; then BJ='[dry run: not queried]'; BAL=99
    else BJ=$(balance) || rfail "balance query failed"; BAL=$(cny "$BJ") || rfail "balance reply has no CNY total: $BJ"; fi
    say "[$NAME] balance ($([ $FIRST = 1 ] && echo 'before the start' || echo 'before resuming')): $BJ"
    [ $FIRST = 1 ] && B0=$BJ; FIRST=0
    if ! python3 -c 'import sys; sys.exit(float(sys.argv[1]) < float(sys.argv[2]))' "$BAL" "$MIN_BAL"; then
      echo "$(date '+%F %T %Z') run $NAME: balance $BAL CNY < $MIN_BAL CNY" > "$LH/STOP"; rfail "balance $BAL CNY < $MIN_BAL CNY: not started (STOP for all runs)"
    fi
    say "[$NAME] off-peak window open, Pro lanes may run until $(hm "$T")"
    for i in 0 1; do [ "${ST[$i]}" = pending ] && start_lane "$i"; done
    while [ "$(cnt running)" -gt 0 ]; do
      for i in 0 1; do [ "${ST[$i]}" = running ] && ! kill -0 "${PID[$i]}" 2>/dev/null && reap_lane "$i"; done
      [ "$(cnt running)" -gt 0 ] || break
      if stopped; then for i in 0 1; do [ "${ST[$i]}" = running ] && stop_lane "$i" "global STOP"; done
        rfail "global STOP: $(cat "$LH/STOP")"; fi
      if [ "$(date +%s)" -ge "$T" ]; then for i in 0 1; do [ "${ST[$i]}" = running ] && stop_lane "$i" "end of the off-peak window"; done; break; fi
      sleep "$POLL"
    done
    say "[$NAME] window over: done $(cnt done), failed $(cnt failed), pending $(cnt pending)"
  done
  if [ "$DRY" = 1 ]; then B1='[dry run: not queried]'; else B1=$(balance) || B1="<balance query failed>"; fi
  say "[$NAME] balance (after): $B1"
  local OUT; OUT=$(printf 'run %s (%s, C%s)\n  tune Pro: %s  [%s, runs %s]\n  test Pro: %s  [%s, runs %s] (trial %s)\n  balance before: %s\n  balance after: %s\n' \
    "$NAME" "$M" "$CL" "$TUH" "${ST[0]}${WHY[0]:+: ${WHY[0]}}" "${RUNS_[0]}" "$DRH" "${ST[1]}${WHY[1]:+: ${WHY[1]}}" "${RUNS_[1]}" "$TRH" "$B0" "$B1")
  if [ "$(cnt failed)" -gt 0 ]; then echo "$OUT" > "$RL/RUN.FAIL"; say "[$NAME] RUN FAILED"; exit 1; fi
  echo "$OUT" > "$RL/RUN.DONE"; say "[$NAME] RUN.DONE"; exit 0
}

say "driver pid $$ (dry=$DRY), log dir $LH, $(sha256sum "$SELF" | cut -c1-16) $SELF"
declare -a WPID
for i in "${!RUNS[@]}"; do
  IFS='|' read -r NAME _ <<< "${RUNS[$i]}"; mkdir -p "$LH/run_$NAME"
  ( worker "$i" ) >> "$LH/run_$NAME/worker.log" 2>&1 < /dev/null &
  WPID[$i]=$!; say "worker $NAME (pid ${WPID[$i]}): ${RUNS[$i]}, log $LH/run_$NAME/worker.log"
done
for i in "${!RUNS[@]}"; do wait "${WPID[$i]}"; done
SUM=""; BAD=0
for i in "${!RUNS[@]}"; do
  IFS='|' read -r NAME _ <<< "${RUNS[$i]}"; d=$LH/run_$NAME
  if [ -e "$d/RUN.DONE" ]; then SUM+="OK   $(cat "$d/RUN.DONE")"$'\n'
  else BAD=1; SUM+="FAIL run $NAME: $(cat "$d/RUN.FAIL" 2>/dev/null || echo '<no RUN.FAIL: the worker died>')"$'\n'; fi
done
[ -e "$LH/STOP" ] && { BAD=1; SUM+="STOP: $(cat "$LH/STOP")"$'\n'; }
if [ $BAD = 1 ]; then printf 'end %s\n%s' "$(date '+%F %T %Z')" "$SUM" > "$LH/ALL.FAIL"; say "ALL.FAIL"; echo "$SUM"; exit 1; fi
printf 'end %s\n%s' "$(date '+%F %T %Z')" "$SUM" > "$LH/ALL.DONE"; say "ALL.DONE"; echo "$SUM"

#!/bin/bash
# 2026-11-24 03 h, exploration (main session, Claude Fable 5.2): a budget guard for the CAA judging (caa_judge.sh).
# The judge balance is read once a minute (balance.sh prints only the balance JSON). While it is below FLOOR yuan
# (default 1), every running CAA judge runner (judge_two_axis_run.py llm|single on a path under baselines/caa/) gets
# SIGINT, again every 20 seconds, because caa_chain.sh starts the next layer's judge as soon as one ends. A judge that
# runs out of balance stalls silently; a chain stopped here ends with its own FAILED line (a judging step), keeps its
# verdict caches and is started again by hand after the balance is topped up (judge_all skips judged work dirs).
# Nothing else is touched: no card, no file of the chain, no FalseQA runner.
# Ends when Iteration_01/logs/newsets_20261119/CAA_BUDGET_GUARD_STOP exists or after HOURS hours (default 40).
# Status lines: "<date> caa budget guard: ..." in Iteration_01/logs/newsets_20261119/caa_budget_guard_1124.log
# (a log of its own: caa_after_1123.sh reads caa_all.log to tell peak-hour refusals from other failures).
# usage (host): FLOOR=1 nohup setsid bash Iteration_01/logs/newsets_20261119/caa_budget_guard_1124.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
L=Iteration_01/logs/newsets_20261119
FLOOR=${FLOOR:-1}
HOURS=${HOURS:-40}
PAT='[j]udge_two_axis_run.py (llm|single) [^ ]*baselines/caa/'
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') caa budget guard: $*" >> $L/caa_budget_guard_1124.log; }
bal() {
  bash $L/balance.sh 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin)[0]["total_balance"])
except Exception:
    print("")'
}
below() { python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)' "$1" "$FLOOR"; }
end=$(( $(date +%s) + HOURS * 3600 ))
low=0
say "start (pid $$), floor $FLOOR yuan, at most $HOURS hours"
while [ ! -e $L/CAA_BUDGET_GUARD_STOP ] && [ "$(date +%s)" -lt "$end" ]; do
  b=$(bal)
  if [ -n "$b" ] && below "$b"; then
    [ $low = 0 ] && say "balance $b is below $FLOOR: CAA judge runners are stopped until it is topped up"
    low=1
    for i in 1 2 3; do
      if pgrep -f "$PAT" > /dev/null; then
        pkill -INT -f "$PAT"
        say "runners signalled (balance $b)"
      fi
      sleep 20
    done
  else
    [ $low = 1 ] && [ -n "$b" ] && say "balance $b is at or above $FLOOR again: runners are left alone"
    [ -n "$b" ] && low=0
    sleep 60
  fi
done
say "guard ends (last balance $(bal))"

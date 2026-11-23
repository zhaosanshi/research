#!/bin/bash
# 2026-11-24 02 h, exploration (main session, Claude Fable 5.2): a budget guard for tonight's FalseQA judging.
# The judge balance is read every two minutes (balance.sh prints only the balance JSON). When it falls below FLOOR
# yuan (default 7), every running FalseQA judge runner (judge_two_axis_run.py on a path under judge/falseqa/) gets
# SIGINT, so that the balance is not used up (a judge that runs out of balance stalls silently) and the CAA chains
# keep a reserve. Nothing else is touched. The verdict cache keeps what was judged; falseqa_judge.sh MODEL heldout
# resumes later. Ends when FALSEQA_JUDGE_NIGHT_1124_DONE exists or after it stopped the runners.
# Status lines: "<date> falseqa budget guard: ..." in Iteration_01/logs/newsets_20261119/falseqa_all.log.
# usage (host): FLOOR=7 nohup setsid bash Iteration_01/logs/newsets_20261119/falseqa_budget_guard_1124.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
L=Iteration_01/logs/newsets_20261119
FLOOR=${FLOOR:-7}
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') falseqa budget guard: $*" >> $L/falseqa_all.log; }
bal() {
  bash $L/balance.sh 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin)[0]["total_balance"])
except Exception:
    print("")'
}
say "start (pid $$), floor $FLOOR yuan"
while [ ! -e $L/FALSEQA_JUDGE_NIGHT_1124_DONE ]; do
  b=$(bal)
  if [ -n "$b" ] && python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)' "$b" "$FLOOR"; then
    say "balance $b is below $FLOOR: stopping the FalseQA judge runners"
    pkill -INT -f "[j]udge_two_axis_run.py (llm|single) [^ ]*judge/falseqa/"
    say "runners signalled (pkill exit $?); guard ends"
    exit 0
  fi
  sleep 120
done
say "judging queue finished; guard ends (last balance $(bal))"

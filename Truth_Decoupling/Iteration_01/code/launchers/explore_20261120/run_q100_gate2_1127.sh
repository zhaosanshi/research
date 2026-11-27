#!/bin/bash
# Iteration_01 exploration (main session, Claude Opus 5.5, 2026-11-27): queue P section 5, the 100-question passes
# (discussion/work_queue/2026-11-26_queue_p.md, progress entry 2026-11-27 08:4x). Zhao Lei 07:40: why judge every question
# when 100 show the rough answer. Not reviewed by GPT.
# Lanes (in parallel, one judge process each), the models whose full-run judging had not finished:
#   A gemma3-27b (FalseQA held-out, the ON questions of the 50 pairs of newsets/gate2_1127/q100_pairs_gate2_1127.json);
#   B qwen3.8-27b gemma3-12b gemma3-27b llama3.1-8b (TruthfulQA test, the Dr's 100 in the main list).
#   A lane is skipped when its full-run judging is done (GATE2_JUDGE_<m>_DONE / GATE2B_JUDGE_<m>_DONE).
# Per lane: diag/q100_gate2_1127.py judge-input-a|b (the full-run judge input restricted to the subset, read only);
#   judge_two_axis_run.py cells; Pro reply cache seeded with the caches the full run used (A: test_full700_heldout and the
#   full-run work dir test_gate2_1127_heldout; B: judge_test_posprompt<tag>_test, judge_test_posprompt2<tag>_test and the
#   full-run work dir judge_test_gate2b_1127_tqa_test when it has a cache): same text -> same verdict, sources untouched;
#   no cell of the work dir may use a request key that the merged caches answer in two ways (diag/q100_conflict_use_1127.py;
#   changed 2026-11-27 08:32 from "no conflicting key at all", after the two TruthfulQA seed caches of gemma3-12b and
#   llama3.1-8b turned out to disagree on 1 and 2 keys that none of their 100-question cells uses); off-peak only (weekdays 08:30-12:00 and 14:00-18:00 Beijing are peak; a pass is cut 2 minutes
#   before a peak); no batch that could take these passes over CAP yuan in all (sum of the upper estimates of
#   crow_pro_requests.py, read and written under a lock) or the balance under FLOOR yuan; llm then single (Pro), the
#   container commands of run_gate2_1127.sh (A) and run_gate2b_1127.sh (B) unchanged; then
#   diag/table_q100_gate2_1127.py a|b q100 q100 <model> > diag/out_q100_<seg>_<model>_20261127.txt.
# At the end diag/out_q100_gate2_1127_20261127.txt: A on the 50 pairs (full-run verdicts for the models judged in full,
#   100-question verdicts for the others) and B on the Dr's 100 (likewise).
# Markers in Iteration_01/logs/newsets_20261119: Q100_JUDGE_<seg>_<model>_DONE, Q100_TABLE_<seg>_<model>_DONE,
#   Q100_DONE / Q100_FAILED / Q100_COST_STOP. Status lines "q100:" in crow_all.log; step logs q100_1127_<seg>_<model>.log.
# usage (host): run_q100_gate2_1127.sh full
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling; ROOT=/workspace/Truth_Decoupling
cd $HOST || exit 2
export TZ=Asia/Shanghai
L=Iteration_01/logs/newsets_20261119
ALL=$L/crow_all.log
NS=$ROOT/Iteration_01/newsets
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
QH=Iteration_01/diag/q100_gate2_1127.py
TQ=Iteration_01/diag/table_q100_gate2_1127.py
OUT=Iteration_01/diag/out_q100_gate2_1127_20261127.txt
CAP=2; FLOOR=10
SPENT=$L/q100_1127_cost_spent.txt
LANES_A="gemma3-27b"
LANES_B="qwen3.8-27b gemma3-12b gemma3-27b llama3.1-8b"
ORDER="llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b"
say() { echo "$(date '+%F %T') q100: $*" >> $ALL; }
failed() { say "FAILED: $*"; touch $L/Q100_FAILED; }
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)

setm() {  # setm SEG MODEL: the judging setup of run_gate2_1127.sh (A) / run_gate2b_1127.sh (B), plus the 100-question work dir
  SEG=$1; M=$2
  MLOG=$L/q100_1127_${SEG}_$M.log
  if [ $SEG = a ]; then
    CODE=$ROOT/Iteration_01/code
    J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
       -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
    JX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
        -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
        -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
    W=$NS/judge/falseqa/$M/test_gate2_1127_q100
    JIN=judge_input_gate2_1127.json
    FULLDONE=$L/GATE2_JUDGE_${M}_DONE
    SEEDS="$NS/judge/falseqa/$M/test_full700_heldout/judge2/cache"
    local fw=$NS/judge/falseqa/$M/test_gate2_1127_heldout/judge2/cache
    [ -f $HOST/${fw#$ROOT/}/judge2_cache_deepseek-v4-pro.jsonl ] && SEEDS="$SEEDS $fw"
    return 0
  fi
  case $M in
    llama3.1-8b) CODE=$ROOT/code_e45_20261112; run=results_e44_factorial_das/llama3.1-8b_side_E2; EL=16 ;;
    qwen3.5-9b)  CODE=$ROOT/code_ff_explore_20261113b; run=$E44/qwen3.5-9b_E2; EL=15 ;;
    qwen3.8-27b) CODE=$ROOT/code_ff_explore_20261113b; run=$E44/qwen3.8-27b-dm; EL=28 ;;
    gemma3-12b)  CODE=$ROOT/Iteration_01/code; run=$G3/12b/e44/gemma3-12b; EL=24 ;;
    gemma3-27b)  CODE=$ROOT/Iteration_01/code; run=$G3/27b/e44/gemma3-27b; EL=28 ;;
    *) echo "unknown model $M" >&2; return 2 ;;
  esac
  TAG=l$EL
  J=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -e CUDA_VISIBLE_DEVICES= -w $CODE truth_dec)
  JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
      -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  W=$ROOT/$run/eval_extra/judge_test_gate2b_1127_q100_test
  JIN=judge_input_gate2b_1127.json
  FULLDONE=$L/GATE2B_JUDGE_${M}_DONE
  SEEDS="$ROOT/$run/eval_extra/judge_test_posprompt${TAG}_test/judge2/cache $ROOT/$run/eval_extra/judge_test_posprompt2${TAG}_test/judge2/cache"
  local fw=$ROOT/$run/eval_extra/judge_test_gate2b_1127_tqa_test/judge2/cache
  [ -f $HOST/${fw#$ROOT/}/judge2_cache_deepseek-v4-pro.jsonl ] && SEEDS="$SEEDS $fw"
  return 0
}
st() {  # st NAME CMD...: one step, output to the lane log, its exit code in the common log
  local name=$1 rc; shift
  echo "=== $(date '+%F %T') $SEG $M $name: $*" >> $MLOG
  "$@" >> $MLOG 2>&1; rc=$?
  say "$SEG $M $name exit=$rc"
  return $rc
}
budget() {  # seconds this lane may judge now (0 = peak: weekdays 08:30-12:00 and 14:00-18:00), cut 120 s before a peak
  local dow=$(date +%u) hm=$((10#$(date +%H%M))) now=$(date +%s) t
  if [ $dow -ge 6 ]; then echo 7200; return; fi
  if [ $hm -lt 830 ]; then t=$(( $(date -d "08:30" +%s) - now - 120 ))
  elif [ $hm -lt 1200 ]; then t=0
  elif [ $hm -lt 1400 ]; then t=$(( $(date -d "14:00" +%s) - now - 120 ))
  elif [ $hm -lt 1800 ]; then t=0
  else t=7200; fi
  [ $t -gt 7200 ] && t=7200
  [ $t -lt 180 ] && t=0
  echo $t
}
bal() {
  bash $L/balance.sh 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin)[0]["total_balance"])
except Exception:
    print("")'
}
reserve() {  # reserve UPPER: under the lock, add UPPER to the spent sum unless it would pass CAP or the balance would fall under FLOOR
  local up=$1 b sp
  b=$(bal); [[ $b =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "$SEG $M: cannot read the balance"; return 2; }
  exec 7>$HOST/$L/.q100_1127_spent.lock
  flock 7
  sp=0; [ -s $SPENT ] && sp=$(cat $SPENT)
  if ! [[ $sp =~ ^[0-9]+(\.[0-9]+)?$ ]]; then flock -u 7; exec 7>&-; say "cannot read $SPENT"; return 2; fi
  if python3 -c 'import sys; sp, b, up, cap, floor = map(float, sys.argv[1:]); sys.exit(0 if (sp + up > cap or b - up < floor) else 1)' $sp $b $up $CAP $FLOOR; then
    flock -u 7; exec 7>&-
    say "COST STOP before $SEG $M: earlier batches at most $sp yuan, this batch at most $up, cap $CAP; balance $b, floor $FLOOR"
    touch $L/Q100_COST_STOP
    return 9
  fi
  python3 -c 'import sys; print("%.4f" % (float(sys.argv[1]) + float(sys.argv[2])))' $sp $up > $SPENT
  flock -u 7; exec 7>&-
  say "$SEG $M: batch reserved (at most $up yuan; earlier at most $sp; balance $b)"
  return 0
}
judge() {  # judge: DeepSeek Pro on $W, off-peak, within the cap and above the floor (judge() of the full-run launchers)
  local cost=$L/q100_1127_cost_${SEG}_$M.txt secs up rc tries=0 left s
  [ -f $HOST/${W#$ROOT/}/judge2/cells.jsonl ] || { say "$SEG $M: no cells in $W"; return 1; }
  for s in $SEEDS; do
    [ -f $HOST/${s#$ROOT/}/judge2_cache_deepseek-v4-pro.jsonl ] || { say "$SEG $M: no Pro reply cache in $s"; return 1; }
  done
  st merge "${J[@]}" python3 scripts/review_checks/judge2_cache_merge.py $W/judge2/cache $SEEDS || { say "$SEG $M: cache merge FAILED"; return 1; }
  # 08:32 change: stop only when a cell of this work dir uses a request key the merged caches answer in two ways
  # (diag/q100_conflict_use_1127.py); single leaves such a cell without a verdict anyway, and a cell without a verdict stops
  st conflicts "${J[@]}" python3 -B $ROOT/Iteration_01/diag/q100_conflict_use_1127.py $W || { say "$SEG $M: a cell uses a request key with conflicting cached replies"; return 1; }
  while true; do
    secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
    [ -e $L/Q100_COST_STOP ] && return 9
    "${J[@]}" python3 -B $ROOT/Iteration_01/diag/crow_pro_requests.py $W > $cost 2>> $MLOG || { say "$SEG $M: request count FAILED"; return 1; }
    up=$(sed -n 's/^expected cost .*, upper \([0-9.]*\) CNY.*/\1/p' $cost)
    [[ $up =~ ^[0-9]+\.[0-9]+$ ]] || { say "$SEG $M: cannot read the cost estimate in $cost"; return 1; }
    reserve $up; rc=$?
    [ $rc = 0 ] || return $rc
    say "$SEG $M: Pro batch (at most $secs s); $(sed -n 2,3p $cost | tr "\n" " ")"
    st pro "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $W --judge pro
    rc=$?
    say "$SEG $M: balance after the batch $(bal)"
    case $rc in
      0) st single "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W --judge pro || { say "$SEG $M: single (pro) FAILED"; return 1; }
         left=$("${J[@]}" python3 -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
ok = [r for r in rows if r['status'] == 'scored' or r.get('basis') in ('empty_output_no_judge_vote', 'pro_declined')]
print(len(rows) - len(ok))" "$W/judge2/single_pro.jsonl")
         [[ $left =~ ^[0-9]+$ ]] || { say "$SEG $M: cannot read the verdicts"; return 1; }
         [ "$left" = 0 ] && return 0
         tries=$((tries + 1))
         [ $tries -ge 2 ] && { say "$SEG $M: $left cells without a Pro verdict after $tries complete passes"; return 1; }
         say "$SEG $M: $left cells without a verdict; one more pass" ;;
      124) say "$SEG $M: pass cut before a peak hour; resumes off-peak" ;;
      *) say "$SEG $M: the Pro pass FAILED (rc $rc)"; return 1 ;;
    esac
  done
}
lane() {  # lane SEG MODEL
  setm $1 $2 || return 1
  [ -e $L/Q100_TABLE_${SEG}_${M}_DONE ] && return 0
  if [ -e $FULLDONE ]; then say "$SEG $M: the full-run judging is done; no 100-question pass"; return 0; fi
  if [ ! -e $L/Q100_JUDGE_${SEG}_${M}_DONE ]; then
    st judgeinput "${RD[@]}" $QH judge-input-$SEG $M || { say "$SEG $M: judge input FAILED"; return 1; }
    if [ ! -e $HOST/${W#$ROOT/}/judge2/cells.jsonl ]; then
      st cells "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation $JIN || { say "$SEG $M: cells FAILED"; return 1; }
    fi
    judge; local rc=$?
    [ $rc = 0 ] || { [ $rc = 9 ] || failed "judging of $SEG $M (rc $rc)"; return 1; }
    touch $L/Q100_JUDGE_${SEG}_${M}_DONE; say "$SEG $M: judged"
  fi
  "${RD[@]}" $TQ $SEG q100 q100 $M > Iteration_01/diag/out_q100_${SEG}_${M}_20261127.txt 2>> $MLOG || {
    failed "table of $SEG $M"; return 1; }
  touch $L/Q100_TABLE_${SEG}_${M}_DONE; say "$SEG $M: table Iteration_01/diag/out_q100_${SEG}_${M}_20261127.txt"
}
final() {  # the 100-question readings of every model: full-run verdicts where the full run is judged, else the 100-question pass
  local seg m fullm qm part
  : > $OUT.tmp
  for seg in a b; do
    fullm=""; qm=""
    for m in $ORDER; do
      if [ $seg = a ]; then [ -e $L/GATE2_JUDGE_${m}_DONE ] && [ -e $L/GATE2_SCORE_${m}_full_DONE ] && { fullm="$fullm $m"; continue; }
      else [ -e $L/GATE2B_JUDGE_${m}_DONE ] && { fullm="$fullm $m"; continue; }; fi
      [ -e $L/Q100_JUDGE_${seg}_${m}_DONE ] && qm="$qm $m"
    done
    for part in "full:$fullm" "q100:$qm"; do
      [ -n "${part#*:}" ] || continue
      "${RD[@]}" $TQ $seg ${part%%:*} q100 ${part#*:} >> $OUT.tmp 2>> $L/q100_1127_final.err || return 1
    done
  done
  mv $OUT.tmp $OUT
}

case ${1:-} in
  full)
    for f in $QH $TQ Iteration_01/diag/crow_pro_requests.py $L/balance.sh Iteration_01/diag/table_gate2_1127.py Iteration_01/diag/table_gate2b_1127.py; do
      [ -s $f ] || { echo "missing $f" >&2; exit 2; }
    done
    command -v flock > /dev/null || { echo "flock unavailable" >&2; exit 2; }
    mkdir $L/.q100_1127_running 2>/dev/null || { echo "already running ($L/.q100_1127_running)"; exit 3; }
    trap 'rmdir $L/.q100_1127_running 2>/dev/null' EXIT
    [ -e $L/Q100_DONE ] && { say "already done (Q100_DONE)"; exit 0; }
    rm -f $L/Q100_FAILED
    say "start (pid $$): helper $(sha256sum $QH | cut -c1-16), table $(sha256sum $TQ | cut -c1-16), launcher $(sha256sum $0 | cut -c1-16)"
    PIDS=""
    for m in $LANES_A; do ( lane a $m ) & PIDS="$PIDS $!"; done
    for m in $LANES_B; do ( lane b $m ) & PIDS="$PIDS $!"; done
    BAD=0
    for p in $PIDS; do wait $p || BAD=$((BAD + 1)); done
    final || { failed "final table (see $L/q100_1127_final.err)"; exit 1; }
    say "final table $OUT $(sha256sum $OUT | cut -c1-16)"
    if [ $BAD -gt 0 ]; then
      [ -e $L/Q100_COST_STOP ] && say "stopped by the cost guard ($BAD lanes)" || failed "$BAD lanes"
      exit 1
    fi
    touch $L/Q100_DONE; say "done: $OUT"
    exit 0 ;;
  *) echo "usage: run_q100_gate2_1127.sh full" >&2; exit 2 ;;
esac

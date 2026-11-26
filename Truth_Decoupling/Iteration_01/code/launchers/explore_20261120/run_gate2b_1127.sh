#!/bin/bash
# Iteration_01 exploration (executor xu, Claude Opus 5.5, 2026-11-27): queue P section 5, segment B
# (discussion/work_queue/2026-11-26_queue_p.md, progress entry 2026-11-27 05:12). TruthfulQA test split (main list per
# model): N1 at the larger E strength s2, with and without the frozen switch. Not reviewed by GPT.
# Per model (generation lanes: card 0 qwen3.5-9b then llama3.1-8b; card 1 gemma3-12b then gemma3-27b; cards 2,3 qwen3.8-27b):
#   1 generation (card lock): entry check (existing prompt-rule test arms x:E<el>.w.p<s> and x:E<el>.w.p<s>+C<cl>.w.m0.25,
#     plus x:E<el>.w.p<s2> where it exists, regenerated on the first 16 questions (--first-batches 2, files t16.*) must
#     equal the archived files row by row: diag/tqa_gate2b_1127.py entry-compare); then the new arms through the entries
#     of the main table (pospall.sh): scripts/review_checks/e44_positions.py prompt generate --split test --arms
#     x:E<el>.w.p<s2> (an existing complete file is verified and reused by the engine) and e44_positions2.py prompt
#     generate --split test --arms x:E<el>.w.p<s2>+C<cl>.w.m<c> (c = 0.25 0.5 1 2; the existing test-split acceptance
#     record of the pair E<el> / C<cl> at "prompt" is used as it is, nothing is written to it); tqa_gate2b_1127.py
#     check-gen; e44_positions2.py prompt export --gen <the five files> --name gate2b_1127_tqa_test (CPU);
#   2 judge input / cells (CPU): tqa_gate2b_1127.py judge-input (main-list questions; cells that cannot be handed in by
#     the ungated N1 at s2 get an empty output = no judge call), judge_two_axis_run.py cells -> work dir
#     <run>/eval_extra/judge_test_gate2b_1127_tqa_test;
#   3 judging (one judge lane, order qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b llama3.1-8b): Pro reply cache seeded with
#     the model's caches of judge_test_posprompt<tag>_test and judge_test_posprompt2<tag>_test (same text -> same verdict;
#     sources untouched), conflicts must be none, off-peak only (weekdays 08:30-12:00 and 14:00-18:00 Beijing are peak; a
#     pass is cut 2 minutes before a peak), no batch that could take this segment over CAP yuan (sum of the upper
#     estimates of crow_pro_requests.py, gate2b_1127_cost_spent.txt) or the balance under FLOOR yuan; llm then single (Pro);
#   4 table refresh: diag/table_gate2b_1127.py <judged models> > diag/out_gate2b_1127_tqa_20261127.txt.
# Cards through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock retried every 60 s); ours plus others at most 66560 MiB
# per card at the start, waits otherwise. No self-judge scoring in this segment.
# Markers in Iteration_01/logs/newsets_20261119: GATE2B_GEN_<model>_DONE, GATE2B_CELLS_<model>_DONE, GATE2B_JUDGE_<model>_DONE,
# GATE2B_TABLE_<model>_DONE, GATE2B_DONE / GATE2B_FAILED / GATE2B_COST_STOP; smoke GATE2B_SMOKE_DONE.
# Status lines "gate2b:" in crow_all.log; step logs gate2b_1127_<model>.log, gate2b_1127_judge_<model>.log.
# usage (host):  run_gate2b_1127.sh smoke    qwen3.5-9b: generation of the five arms (all test questions), judge input on
#                                          3 main-list questions, judging, table cell by cell (card 0)
#                run_gate2b_1127.sh full     the whole chain, five models
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling; ROOT=/workspace/Truth_Decoupling
cd $HOST || exit 2
export TZ=Asia/Shanghai
L=Iteration_01/logs/newsets_20261119
ALL=$L/crow_all.log
TF=$ROOT/envs/tf519/bin/python
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
HELP=Iteration_01/diag/tqa_gate2b_1127.py
TABLE=Iteration_01/diag/table_gate2b_1127.py
OUTT=Iteration_01/diag/out_gate2b_1127_tqa_20261127.txt
OUTS=Iteration_01/diag/out_gate2b_1127_smoke_20261127.txt
NAME=gate2b_1127_tqa_test
MAXMIB=66560; CAP=6; FLOOR=10
SPENT=$L/gate2b_1127_cost_spent.txt
ORDER="qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b llama3.1-8b"
MODE=${1:-}
say() { echo "$(date '+%F %T') gate2b: $*" >> $ALL; }
failed() { say "FAILED: $*"; touch $L/GATE2B_FAILED; }

setm() {  # setm MODEL: run_capilot_1126.sh setm (code copies, run dirs, interpreters, engine arguments) with --split test
  M=$1
  case $M in
    llama3.1-8b) CODE=$ROOT/code_e45_20261112; PYM=python3; run=results_e44_factorial_das/llama3.1-8b_side_E2
                 crun=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
                 EL=16; S=3.5; CL=19; S2=5; NEED=20000; CARD=0 ;;
    qwen3.5-9b)  CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.5-9b_E2
                 crun=$E44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b
                 EL=15; S=2; CL=16; S2=3; NEED=22000; CARD=0 ;;
    qwen3.8-27b) CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
                 crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
                 EL=28; S=1.75; CL=33; S2=3; NEED=34000; CARD=2,3 ;;
    gemma3-12b)  CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/12b/e44/gemma3-12b
                 crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/$G3/12b/source/gemma3-12b
                 EL=24; S=2.5; CL=25; S2=4; NEED=30000; CARD=1 ;;
    gemma3-27b)  CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
                 crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/$G3/27b/source/gemma3-27b
                 EL=28; S=1; CL=31; S2=2; NEED=62000; CARD=1 ;;
    *) echo "unknown model $M" >&2; return 2 ;;
  esac
  TAG=l$EL
  E2A=x:E$EL.w.p$S2
  ECARMS=""; local c
  for c in 0.25 0.5 1 2; do ECARMS="$ECARMS $E2A+C$CL.w.m$c"; done
  ECARMS=${ECARMS# }
  MLOG=$L/gate2b_1127_$M.log; JLOG=$L/gate2b_1127_judge_$M.log
  [ -n "${SMOKE:-}" ] && { MLOG=$L/gate2b_1127_smoke_$M.log; JLOG=$L/gate2b_1127_smoke_judge_$M.log; }
  WK=$run/eval_extra/judge_test_gate2b_1127_tqa_test
  WS=$run/eval_extra/judge_test_gate2b_1127_smoke_test
  SEEDS="$ROOT/$run/eval_extra/judge_test_posprompt${TAG}_test/judge2/cache $ROOT/$run/eval_extra/judge_test_posprompt2${TAG}_test/judge2/cache"
  COMMON=(--source-run $SRC --run-dir $ROOT/$run --e-on $E_ON --c-run-dir $ROOT/$crun --c-e-on $C_E_ON --split test
          --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
      -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  J=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -e CUDA_VISIBLE_DEVICES= -w $CODE truth_dec)
  return 0
}
dx() {  # dx DEVICES: the generation container command of run_capilot_1126.sh
  DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
}
fname() { local f=${1//:/--}; f=${f//+/-and-}; f=${f//./-}; echo "${f}-at-prompt_test.json"; }
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)
st() {  # st NAME CMD...: one step, output to the model log, its exit code in the common log
  local name=$1 rc; shift
  echo "=== $(date '+%F %T') $M $name: $*" >> $MLOG
  "$@" >> $MLOG 2>&1; rc=$?
  say "$M $name exit=$rc"
  return $rc
}
memwait() {  # memwait CARD MIB
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}')
    [[ $used =~ ^[0-9]+$ ]] || return 1
    [ $((used + $2)) -le $MAXMIB ] && { say "$M: card $1 holds $used MiB, asking $2"; return 0; }
    say "$M: card $1 holds $used MiB, $2 more would exceed $MAXMIB; waiting"; sleep 60
  done
}
LOCKED=""
lock() {  # lock CARDS FD0 MIB: flock every card (retry every 60 s), then the memory rule on each
  local g fd=$2
  LOCKED=""
  for g in ${1//,/ }; do
    eval "exec $fd>\"$HOST/Iteration_01/newsets/selfjudge/.gpu$g.lock\"" || { say "$M: cannot open the lock of card $g"; return 1; }
    until flock -n $fd; do sleep 60; done
    say "$M: holds the lock of card $g"
    LOCKED="$LOCKED $fd"; fd=$((fd + 1))
  done
  for g in ${1//,/ }; do memwait $g $3 || { say "$M: cannot read the memory of card $g"; return 1; }; done
}
unlock() { local fd; for fd in $LOCKED; do flock -u $fd; eval "exec $fd>&-"; done; LOCKED=""; say "$M: cards released"; }

entry_check() {  # existing prompt-rule test arms regenerated on the first 16 questions = the archived files (cards held)
  local a1=x:E$EL.w.p$S a2=x:E$EL.w.p$S+C$CL.w.m0.25 singles a
  singles=$a1
  [ -s $HOST/$run/eval_extra/gen/$(fname $E2A) ] && singles="$a1 $E2A"
  st entrygen1 "${DX[@]}" $PYM scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --arms $singles --first-batches 2 || {
    say "$M: entry check generation (single writes) FAILED"; return 1; }
  st entrygen2 "${DX[@]}" $PYM scripts/review_checks/e44_positions2.py prompt generate "${COMMON[@]}" --arms $a2 --first-batches 2 || {
    say "$M: entry check generation (two writes) FAILED"; return 1; }
  for a in $singles $a2; do
    st entrycompare "${RD[@]}" $HELP entry-compare $M $a || { say "$M: entry check: $a differs from the archived file"; return 1; }
  done
  say "$M: entry check passed ($singles $a2 reproduced on 16 questions, card(s) $CARD)"
}

gen_model() {  # gen_model MODEL FD0: entry check, generation, check, export
  setm $1 || return 1
  [ -e $L/GATE2B_FAILED ] && return 1
  [ -e $L/GATE2B_GEN_${M}_DONE ] && return 0
  local f gens="" a
  for a in $E2A $ECARMS; do gens="$gens $(fname $a)"; done
  for f in $(fname x:E$EL.w.p$S) $(fname x:E$EL.w.p$S+C$CL.w.m0.25); do
    [ -s $HOST/$run/eval_extra/gen/$f ] || { say "$M: no archived file $f"; return 1; }
  done
  lock $CARD $2 $NEED || return 1
  say "$M: generation on card(s) $CARD; arms $E2A $ECARMS"
  dx $CARD
  entry_check || { unlock; return 1; }
  st generate1 "${DX[@]}" $PYM scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --arms $E2A || {
    unlock; say "$M: generation of $E2A FAILED"; return 1; }
  st generate2 "${DX[@]}" $PYM scripts/review_checks/e44_positions2.py prompt generate "${COMMON[@]}" --arms $ECARMS || {
    unlock; say "$M: generation of the E + C arms FAILED"; return 1; }
  unlock
  st checkgen "${RD[@]}" $HELP check-gen $M || { say "$M: check-gen FAILED"; return 1; }
  dx ""
  st export "${DX[@]}" $PYM scripts/review_checks/e44_positions2.py prompt export "${COMMON[@]}" --gen $gens --name $NAME || {
    say "$M: export FAILED"; return 1; }
  touch $L/GATE2B_GEN_${M}_DONE; say "$M: generation done"
}

cells_model() {  # cells_model MODEL full|smoke
  setm $1 || return 1
  local W=$WK
  [ $2 = smoke ] && W=$WS
  [ $2 = full ] && [ -e $L/GATE2B_CELLS_${M}_DONE ] && return 0
  st judgeinput "${RD[@]}" $HELP judge-input $M $2 || { say "$M: judge input FAILED"; return 1; }
  if [ ! -e $HOST/$W/judge2/cells.jsonl ]; then
    st cells "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input_gate2b_1127.json || {
      say "$M: cells FAILED"; return 1; }
  fi
  [ $2 = full ] && { touch $L/GATE2B_CELLS_${M}_DONE; say "$M: cells ready"; }
  return 0
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
judge() {  # judge MODEL WORKDIR(relative to the project root): DeepSeek Pro, off-peak, within the cap and above the floor
  setm $1 || return 1
  local W=$2 MLOG=$JLOG cost=$L/gate2b_1127_cost_$1.txt secs b sp up rc tries=0 left s
  [ -n "${SMOKE:-}" ] && cost=$L/gate2b_1127_cost_smoke_$1.txt
  [ -f $HOST/$W/judge2/cells.jsonl ] || { say "$M: no cells in $W"; return 1; }
  for s in $SEEDS; do
    [ -f $HOST/${s#$ROOT/}/judge2_cache_deepseek-v4-pro.jsonl ] || { say "$M: no Pro reply cache in $s"; return 1; }
  done
  st merge "${J[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $SEEDS || { say "$M: cache merge FAILED"; return 1; }
  st conflicts "${J[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json || { say "$M: cache conflicts"; return 1; }
  while true; do
    secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
    [ -e $L/GATE2B_COST_STOP ] && return 9
    "${J[@]}" python3 -B $ROOT/Iteration_01/diag/crow_pro_requests.py $ROOT/$W > $cost 2>> $JLOG || { say "$M: request count FAILED"; return 1; }
    up=$(sed -n 's/^expected cost .*, upper \([0-9.]*\) CNY.*/\1/p' $cost)
    [[ $up =~ ^[0-9]+\.[0-9]+$ ]] || { say "$M: cannot read the cost estimate in $cost"; return 1; }
    b=$(bal); [[ $b =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "$M: cannot read the balance"; return 1; }
    sp=0; [ -s $SPENT ] && sp=$(cat $SPENT)
    [[ $sp =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "cannot read $SPENT"; return 1; }
    if python3 -c 'import sys; sp, b, up, cap, floor = map(float, sys.argv[1:]); sys.exit(0 if (sp + up > cap or b - up < floor) else 1)' $sp $b $up $CAP $FLOOR; then
      say "COST STOP before $M: earlier batches at most $sp yuan, this batch at most $up, cap $CAP; balance $b, floor $FLOOR"
      touch $L/GATE2B_COST_STOP
      return 9
    fi
    python3 -c 'import sys; print("%.4f" % (float(sys.argv[1]) + float(sys.argv[2])))' $sp $up > $SPENT
    say "$M: Pro batch (at most $secs s); balance $b; earlier batches at most $sp yuan; $(sed -n 2,3p $cost | tr "\n" " ")"
    st pro "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro
    rc=$?
    say "$M: balance after the batch $(bal)"
    case $rc in
      0) st single "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro || { say "$M: single (pro) FAILED"; return 1; }
         left=$("${J[@]}" python3 -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
ok = [r for r in rows if r['status'] == 'scored' or r.get('basis') in ('empty_output_no_judge_vote', 'pro_declined')]
print(len(rows) - len(ok))" "$ROOT/$W/judge2/single_pro.jsonl")
         [[ $left =~ ^[0-9]+$ ]] || { say "$M: cannot read the verdicts"; return 1; }
         [ "$left" = 0 ] && return 0
         tries=$((tries + 1))
         [ $tries -ge 2 ] && { say "$M: $left cells without a Pro verdict after $tries complete passes"; return 1; }
         say "$M: $left cells without a verdict; one more pass" ;;
      124) say "$M: pass cut before a peak hour; resumes off-peak" ;;
      *) say "$M: the Pro pass FAILED (rc $rc)"; return 1 ;;
    esac
  done
}
table() {  # table: the judged models
  local m done=""
  for m in $ORDER; do [ -e $L/GATE2B_JUDGE_${m}_DONE ] && done="$done $m"; done
  "${RD[@]}" $TABLE $done > $OUTT.tmp 2> $L/gate2b_1127_table.err && mv $OUTT.tmp $OUTT
}
judge_lane() {
  local m rc
  for m in $ORDER; do
    until [ -e $L/GATE2B_CELLS_${m}_DONE ]; do [ -e $L/GATE2B_FAILED ] && return 1; sleep 30; done
    if [ ! -e $L/GATE2B_JUDGE_${m}_DONE ]; then
      setm $m
      judge $m $WK; rc=$?
      [ $rc = 0 ] || { [ $rc = 9 ] || failed "judging of $m (rc $rc)"; return 1; }
      touch $L/GATE2B_JUDGE_${m}_DONE; say "$m: judged"
    fi
    M=$m
    if table; then
      touch $L/GATE2B_TABLE_${m}_DONE; say "$m: table $OUTT $(sha256sum $OUTT | cut -c1-16)"
    else
      failed "table after $m (see $L/gate2b_1127_table.err)"; return 1
    fi
  done
}
lane() {  # lane FD0 MODEL...: generation then cells, one model after the other
  local fd=$1 m; shift
  for m in "$@"; do
    gen_model $m $fd || { failed "generation lane at $m"; return 1; }
    cells_model $m full || { failed "cells of $m"; return 1; }
  done
}

checkfiles() {
  local f m
  for f in $HELP $TABLE Iteration_01/diag/crow_pro_requests.py $L/balance.sh Iteration_01/gpt_tools_20261126/gate_audit/falseqa_matched.py; do
    [ -s $f ] || { echo "missing $f" >&2; return 1; }
  done
  command -v flock > /dev/null || { echo "flock unavailable" >&2; return 1; }
  for m in $ORDER; do
    setm $m
    for f in e44_positions.py e44_positions2.py e44_split_test.py judge_two_axis_run.py judge2_cache_merge.py; do
      cmp -s Iteration_01/code/scripts/review_checks/$f ${CODE#$ROOT/}/scripts/review_checks/$f || { echo "$M: ${CODE#$ROOT/} has another $f" >&2; return 1; }
    done
  done
}

case $MODE in
  smoke)
    checkfiles || exit 2
    [ -e $L/GATE2B_SMOKE_DONE ] && { echo "smoke already done"; exit 0; }
    SMOKE=1
    say "smoke start (pid $$)"
    gen_model qwen3.5-9b 8 || { failed "smoke generation"; exit 1; }
    st smokeids "${RD[@]}" $HELP smoke-ids $M || { failed "smoke ids"; exit 1; }
    cells_model qwen3.5-9b smoke || { failed "smoke cells"; exit 1; }
    judge $M $WS || { failed "smoke judging"; exit 1; }
    "${RD[@]}" $TABLE smoke $M > $OUTS 2>> $MLOG || { failed "smoke table"; exit 1; }
    touch $L/GATE2B_SMOKE_DONE; say "smoke done: $OUTS"
    exit 0 ;;
  full)
    checkfiles || exit 2
    mkdir $L/.gate2b_1127_running 2>/dev/null || { echo "already running ($L/.gate2b_1127_running)"; exit 3; }
    trap 'rmdir $L/.gate2b_1127_running 2>/dev/null' EXIT
    [ -e $L/GATE2B_DONE ] && { say "already done (GATE2B_DONE)"; exit 0; }
    [ -e $L/GATE2B_SMOKE_DONE ] || { echo "run the smoke first" >&2; exit 2; }
    rm -f $L/GATE2B_FAILED
    say "full start (pid $$): helper $(sha256sum $HELP | cut -c1-16), table $(sha256sum $TABLE | cut -c1-16), launcher $(sha256sum $0 | cut -c1-16)"
    ( lane 20 qwen3.5-9b llama3.1-8b ) & P1=$!
    ( lane 30 gemma3-12b gemma3-27b ) & P2=$!
    ( lane 40 qwen3.8-27b ) & P3=$!
    judge_lane & P4=$!
    BAD=""
    wait $P1 || BAD="$BAD lane-card0"
    wait $P2 || BAD="$BAD lane-card1"
    wait $P3 || BAD="$BAD lane-cards23"
    wait $P4 || BAD="$BAD judge-lane"
    if [ -n "$BAD" ]; then
      [ -e $L/GATE2B_COST_STOP ] && say "stopped by the cost guard:$BAD" || failed "lanes:$BAD"
      exit 1
    fi
    touch $L/GATE2B_DONE; say "segment B done: $OUTT $(sha256sum $OUTT | cut -c1-16)"
    exit 0 ;;
  *) echo "usage: run_gate2b_1127.sh smoke | full" >&2; exit 2 ;;
esac

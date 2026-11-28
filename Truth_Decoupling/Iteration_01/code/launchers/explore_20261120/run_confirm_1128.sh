#!/bin/bash
# Iteration_01 exploration (executor hai, Claude Opus 5.5, 2026-11-28): queue Q (discussion/work_queue/2026-11-28_queue_q.md),
# the frozen simple rule on the FalseQA confirmation bank (newsets/falseqa/falseqa_confirm_bank_20261128.jsonl: the 386
# test.csv pairs outside the held-out 300). Not reviewed by GPT.
# Per model (settings: diag/falseqa_confirm_1128.py setting, = the plan's table):
#   1 generation (card lock): the entry check (x, x:E<el>.w.p<s> and x:E<el>.w.p<s>+C<cl>.w.m0.25 regenerated on the first
#     16 questions of the 700 bank through the same entry must equal the archived files: ns_h22.py compare), then on all
#     confirm questions through scripts/newsets/ns_h22.py falseqa none|prompt gen (NS_POS_TWO_WRITE=1, NS_POS_PLAIN_NAMES=1;
#     the program and settings of falseqa_model.sh) into newsets/gen_prompt/falseqa_confirm_1128/<model>/test:
#       x | x:E<el>.w.p<s> | x:C<cl>.w.m<cC> | x:E<el>.w.p<s>+C<cl>.w.m<k> (k = 0.25 0.5 1 2)
#     write records (scripts/newsets/n1s_check.py parts, unchanged) against the archived held-out file of the same arm;
#     falseqa_confirm_1128.py check-gen. Marker CONF_GEN_<model>_DONE.
#   2 judging (one parallel judge lane per model): the unchanged ns_h22 export of the eight arms
#     (newsets/judge/falseqa/<model>/export_confirm_1128.json); falseqa_confirm_1128.py judge-input (stage q100: the 100
#     questions of the 50 drawn pairs; cells not judged get an empty output); judge_two_axis_run.py cells; DeepSeek Pro
#     (llm then single, the container commands of run_q100_gate2_1127.sh unchanged); weekday peak hours skipped (budget());
#     no batch that could take queue Q over CAP yuan in all (sum of the upper estimates of crow_pro_requests.py, reserved
#     under a lock) or the balance under FLOOR yuan. Marker CONF_JUDGE100_<model>_DONE.
#   3 table: diag/table_confirm_1128.py q100 (regression on the held-out 600 first) > diag/out_confirm_1128_q100_20261128.txt;
#     CONF_TABLE100_DONE; its line "EXPAND_DECISION ... -> EXPAND yes|no" gives CONF_EXPAND_YES / CONF_EXPAND_NO.
#   4 yes: stage all (every confirm question; Pro cache seeded with the stage-1 cache, no cell may use a conflicting key:
#     diag/q100_conflict_use_1127.py), CONF_JUDGEALL_<model>_DONE; table all > diag/out_confirm_1128_all_20261128.txt.
#     The chain ends with CONF_DONE (after EXPAND_NO too) / CONF_FAILED / CONF_COST_STOP.
# GPU lanes (cards through Iteration_01/newsets/selfjudge/.gpu<N>.lock, flock retried every 60 s; ours plus others at most
# 66560 MiB per card at the start, waits otherwise; as run_gate2_1127.sh): lane 1 card 1: qwen3.5-9b, gemma3-12b,
# llama3.1-8b; lane 2: qwen3.8-27b on cards 2,3, then gemma3-27b on card 2.
# Status lines "conf:" in crow_all.log; step logs confirm_1128_<model>.log, confirm_1128_judge_<stage>_<model>.log.
# usage (host):  run_confirm_1128.sh smoke    qwen3.5-9b on the 3 smoke questions, the whole chain (card 1); CONF_SMOKE_DONE
#                run_confirm_1128.sh full     the whole chain, five models (needs CONF_SMOKE_DONE)
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling; ROOT=/workspace/Truth_Decoupling
cd $HOST || exit 2
export TZ=Asia/Shanghai
L=Iteration_01/logs/newsets_20261119
ALL=$L/crow_all.log
TF=$ROOT/envs/tf519/bin/python
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
OBANK=$NS/falseqa/falseqa_full_bank_20261123.jsonl
BANK=$NS/falseqa/falseqa_confirm_bank_20261128.jsonl
CODE=$ROOT/Iteration_01/code
NH=$CODE/scripts/newsets/ns_h22.py
NC=$CODE/scripts/newsets/n1s_check.py
MCFG=$CODE/configs/newsets/frozen_models_20261119.json
HELP=Iteration_01/diag/falseqa_confirm_1128.py
TABLE=Iteration_01/diag/table_confirm_1128.py
OUT100=Iteration_01/diag/out_confirm_1128_q100_20261128.txt
OUTALL=Iteration_01/diag/out_confirm_1128_all_20261128.txt
OUTS=Iteration_01/diag/out_confirm_1128_smoke_20261128.txt
JIN=judge_input_confirm_1128.json
MAXMIB=66560; CAP=10; FLOOR=10
SPENT=$L/confirm_1128_cost_spent.txt
ORDER="llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b"
MODE=${1:-}
say() { echo "$(date '+%F %T') conf: $*" >> $ALL; }
failed() { say "FAILED: $*"; touch $L/CONF_FAILED; }
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
JK=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)

setm() {  # setm MODEL: frozen setting (helper), interpreter, memory, cards (run_gate2_1127.sh's values)
  M=$1
  case $M in
    llama3.1-8b) PYM=python3; GNEED=20000; GCARD=1 ;;
    qwen3.5-9b)  PYM=$TF;     GNEED=22000; GCARD=1 ;;
    qwen3.8-27b) PYM=$TF;     GNEED=34000; GCARD=2,3 ;;
    gemma3-12b)  PYM=python3; GNEED=30000; GCARD=1 ;;
    gemma3-27b)  PYM=python3; GNEED=62000; GCARD=2 ;;
    *) echo "unknown model $M" >&2; return 2 ;;
  esac
  local v
  v=$("${RD[@]}" $HELP setting $M) || return 2
  read -r EL S CL C CC <<< "$v"
  [ -n "${CC:-}" ] || return 2
  ARMS_W="x:E$EL.w.p$S x:C$CL.w.m$CC"
  local k
  for k in 0.25 0.5 1 2; do ARMS_W="$ARMS_W x:E$EL.w.p$S+C$CL.w.m$k"; done
  GD=$NS/gen_prompt/falseqa_confirm_1128/$M/test
  EXP=$NS/judge/falseqa/$M/export_confirm_1128.json
  MLOG=$L/confirm_1128_$M.log
  [ -n "${SMOKE:-}" ] && { GD=$NS/gen_prompt/falseqa_confirm_1128_smoke/$M/test; MLOG=$L/confirm_1128_smoke_$M.log; }
  return 0
}
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
dq() {  # dq DEVICES: the generation container command of falseqa_model.sh
  DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
      -e CUDA_VISIBLE_DEVICES=$1 -w $ROOT truth_dec)
}
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
    eval "exec $fd>\"$HNS/selfjudge/.gpu$g.lock\"" || { say "$M: cannot open the lock of card $g"; return 1; }
    until flock -n $fd; do sleep 60; done
    say "$M: holds the lock of card $g"
    LOCKED="$LOCKED $fd"; fd=$((fd + 1))
  done
  for g in ${1//,/ }; do memwait $g $3 || { say "$M: cannot read the memory of card $g"; return 1; }; done
}
unlock() { local fd; for fd in $LOCKED; do flock -u $fd; eval "exec $fd>&-"; done; LOCKED=""; say "$M: cards released"; }
first() { ls $HOST/${1#$ROOT/}/s????????????.$2.json 2>/dev/null | head -1 | sed "s#^$HOST#$ROOT#"; }

gen() {  # gen none|prompt OUT ITEMS ARGS...: the generation command of falseqa_model.sh
  local pos=$1 out=$2 items=$3; shift 3
  st gen$pos "${DQ[@]}" $PYM -I $NH falseqa $pos gen --models $MCFG --model $M --task falseqa --items $items --split test --out $out "$@"
}

entry_check() {  # x and two write arms regenerated on the first 16 questions of the 700 bank = the archived files
  local EC=$NS/gen_prompt/h22chk/falseqa_confirm_1128${SMOKE:+_smoke}/$M/test A1=x:E$EL.w.p$S A2=x:E$EL.w.p$S+C$CL.w.m0.25 a f
  gen none $EC $OBANK --arms x --first-batches 2 || { say "$M: entry check generation (x) FAILED"; return 1; }
  gen prompt $EC $OBANK --arms $A1 $A2 --first-batches 2 || { say "$M: entry check generation FAILED"; return 1; }
  for a in x $A1 $A2; do
    f=$(first $EC $(tok $a)); [ -n "$f" ] || { say "$M: entry check file of $a missing"; return 1; }
    st entrycompare "${DC[@]}" $NH compare $f $NS/gen_prompt/falseqa/$M/test/$(tok $a).json || {
      say "$M: entry check: $a differs from the archived FalseQA file"; return 1; }
  done
  say "$M: entry check passed (3 archived FalseQA arms reproduced on 16 questions, card(s) $GCARD)"
}

parts_all() {  # parts_all FILEFN: write records of every write arm against the archived held-out file of the same arm
  local fn=$1 a f OG=$NS/gen_prompt/falseqa/$M/test
  for a in $ARMS_W; do
    f=$($fn $a); [ -n "$f" ] || { say "$M: generation file of $a missing"; return 1; }
    case $a in
      *+C*) st parts "${DC[@]}" $NC parts $f $OG/$(tok $a).json $S $S E && st parts "${DC[@]}" $NC parts $f $OG/$(tok $a).json 1 1 C ;;
      *:C*) st parts "${DC[@]}" $NC parts $f $OG/$(tok $a).json 1 1 C ;;
      *)    st parts "${DC[@]}" $NC parts $f $OG/$(tok $a).json $S $S E ;;
    esac || { say "$M: write record of $a FAILED"; return 1; }
  done
}
fullfile() { echo $GD/$(tok $1).json; }
smokefile() { first $GD $(tok $1); }

gpu_model() {  # gpu_model MODEL FD0
  setm $1 || return 1
  [ -e $L/CONF_FAILED ] && return 1
  [ -e $L/CONF_GEN_${M}_DONE ] && return 0
  lock $GCARD $2 $GNEED || return 1
  say "$M: generation on card(s) $GCARD; arms x $ARMS_W"
  dq $GCARD
  entry_check || { unlock; return 1; }
  gen none $GD $BANK --arms x || { unlock; say "$M: generation (x) FAILED"; return 1; }
  gen prompt $GD $BANK --arms $ARMS_W || { unlock; say "$M: generation FAILED"; return 1; }
  unlock
  parts_all fullfile || return 1
  st checkgen "${RD[@]}" $HELP check-gen $M full || { say "$M: check-gen FAILED"; return 1; }
  touch $L/CONF_GEN_${M}_DONE; say "$M: generation done"
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
  b=$(bal); [[ $b =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "$M $STAGE: cannot read the balance"; return 2; }
  exec 7>$HOST/$L/.confirm_1128_spent.lock
  flock 7
  sp=0; [ -s $SPENT ] && sp=$(cat $SPENT)
  if ! [[ $sp =~ ^[0-9]+(\.[0-9]+)?$ ]]; then flock -u 7; exec 7>&-; say "cannot read $SPENT"; return 2; fi
  if python3 -c 'import sys; sp, b, up, cap, floor = map(float, sys.argv[1:]); sys.exit(0 if (sp + up > cap or b - up < floor) else 1)' $sp $b $up $CAP $FLOOR; then
    flock -u 7; exec 7>&-
    say "COST STOP before $M $STAGE: earlier batches at most $sp yuan, this batch at most $up, cap $CAP; balance $b, floor $FLOOR"
    touch $L/CONF_COST_STOP
    return 9
  fi
  python3 -c 'import sys; print("%.4f" % (float(sys.argv[1]) + float(sys.argv[2])))' $sp $up > $SPENT
  flock -u 7; exec 7>&-
  say "$M $STAGE: batch reserved (at most $up yuan; earlier at most $sp; balance $b)"
  return 0
}
judge_stage() {  # judge_stage MODEL STAGE [SEED_CACHE_DIR ...]: judge input, cells, DeepSeek Pro (judge() of run_q100_gate2_1127.sh)
  setm $1 || return 1
  STAGE=$2; shift 2
  local W=$NS/judge/falseqa/$M/test_confirm_1128_$STAGE cost=$L/confirm_1128_cost_${STAGE}_$M.txt secs up rc tries=0 left s
  local HW=$HOST/${W#$ROOT/}
  MLOG=$L/confirm_1128_judge_${STAGE}_$M.log
  if [ $STAGE = smoke ]; then
    st judgeinput "${RD[@]}" $HELP judge-input $M smoke || { say "$M $STAGE: judge input FAILED"; return 1; }
  else
    if [ ! -e $HOST/${EXP#$ROOT/} ]; then
      st export "${J[@]}" python3 -I $NH falseqa none export --task falseqa --model $M --items $BANK --split test \
          --gen-dir $GD --only-arms x $ARMS_W --out $EXP || { say "$M: export FAILED"; return 1; }
    fi
    st judgeinput "${RD[@]}" $HELP judge-input $M $STAGE $EXP || { say "$M $STAGE: judge input FAILED"; return 1; }
  fi
  if [ ! -e $HW/judge2/cells.jsonl ]; then
    st cells "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation $JIN || { say "$M $STAGE: cells FAILED"; return 1; }
  fi
  if [ $# -gt 0 ]; then
    for s in "$@"; do
      [ -f $HOST/${s#$ROOT/}/judge2_cache_deepseek-v4-pro.jsonl ] || { say "$M $STAGE: no Pro reply cache in $s"; return 1; }
    done
    st merge "${J[@]}" python3 scripts/review_checks/judge2_cache_merge.py $W/judge2/cache "$@" || { say "$M $STAGE: cache merge FAILED"; return 1; }
    st conflicts "${J[@]}" python3 -B $ROOT/Iteration_01/diag/q100_conflict_use_1127.py $W || { say "$M $STAGE: a cell uses a request key with conflicting cached replies"; return 1; }
  else  # no seed: crow_pro_requests.py needs the work dir's own (empty) cache dir
    st cachedir "${J[@]}" mkdir -p $W/judge2/cache || { say "$M $STAGE: cannot make the cache dir"; return 1; }
  fi
  while true; do
    secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
    [ -e $L/CONF_COST_STOP ] && return 9
    "${J[@]}" python3 -B $ROOT/Iteration_01/diag/crow_pro_requests.py $W > $cost 2>> $MLOG || { say "$M $STAGE: request count FAILED"; return 1; }
    up=$(sed -n 's/^expected cost .*, upper \([0-9.]*\) CNY.*/\1/p' $cost)
    [[ $up =~ ^[0-9]+\.[0-9]+$ ]] || { say "$M $STAGE: cannot read the cost estimate in $cost"; return 1; }
    reserve $up; rc=$?
    [ $rc = 0 ] || return $rc
    say "$M $STAGE: Pro batch (at most $secs s); $(sed -n 2,3p $cost | tr "\n" " ")"
    st pro "${JK[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $W --judge pro
    rc=$?
    say "$M $STAGE: balance after the batch $(bal)"
    case $rc in
      0) st single "${JK[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W --judge pro || { say "$M $STAGE: single (pro) FAILED"; return 1; }
         left=$("${J[@]}" python3 -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
ok = [r for r in rows if r['status'] == 'scored' or r.get('basis') in ('empty_output_no_judge_vote', 'pro_declined')]
print(len(rows) - len(ok))" "$W/judge2/single_pro.jsonl")
         [[ $left =~ ^[0-9]+$ ]] || { say "$M $STAGE: cannot read the verdicts"; return 1; }
         [ "$left" = 0 ] && return 0
         tries=$((tries + 1))
         [ $tries -ge 2 ] && { say "$M $STAGE: $left cells without a Pro verdict after $tries complete passes"; return 1; }
         say "$M $STAGE: $left cells without a verdict; one more pass" ;;
      124) say "$M $STAGE: pass cut before a peak hour; resumes off-peak" ;;
      *) say "$M $STAGE: the Pro pass FAILED (rc $rc)"; return 1 ;;
    esac
  done
}
judge_lane() {  # judge_lane MODEL STAGE: wait for the generation, then judge the stage
  local m=$1 stage=$2 rc mk
  [ $stage = q100 ] && mk=$L/CONF_JUDGE100_${m}_DONE || mk=$L/CONF_JUDGEALL_${m}_DONE
  [ -e $mk ] && return 0
  until [ -e $L/CONF_GEN_${m}_DONE ]; do [ -e $L/CONF_FAILED ] && return 1; sleep 30; done
  if [ $stage = q100 ]; then
    judge_stage $m q100; rc=$?
  else
    judge_stage $m all $NS/judge/falseqa/$m/test_confirm_1128_q100/judge2/cache; rc=$?
  fi
  [ $rc = 0 ] || { [ $rc = 9 ] || failed "judging of $m $stage (rc $rc)"; return 1; }
  touch $mk; say "$m: $stage judged"
}
table() {  # table q100|all OUT
  M=table
  "${RD[@]}" $TABLE $1 > $2.tmp 2> $L/confirm_1128_table_$1.err && mv $2.tmp $2
}
checkfiles() {
  local f
  for f in $HELP $TABLE Iteration_01/diag/routes_1128.py Iteration_01/diag/out_routes_1128_20261128.txt Iteration_01/diag/crow_pro_requests.py \
           Iteration_01/diag/q100_conflict_use_1127.py $L/balance.sh Iteration_01/code/scripts/newsets/ns_h22.py \
           Iteration_01/code/scripts/newsets/n1s_check.py Iteration_01/newsets/falseqa/falseqa_confirm_bank_20261128.jsonl \
           Iteration_01/newsets/falseqa/falseqa_confirm_1128_ids_20261128.json; do
    [ -s $f ] || { echo "missing $f" >&2; return 1; }
  done
  command -v flock > /dev/null || { echo "flock unavailable" >&2; return 1; }
}
lanes() {  # lanes STAGE: one judge lane per model in parallel; 0 when all finished
  local m p bad=0 pids=""
  for m in $ORDER; do ( judge_lane $m $1 ) & pids="$pids $!"; done
  for p in $pids; do wait $p || bad=$((bad + 1)); done
  return $bad
}

case $MODE in
  smoke)
    checkfiles || exit 2
    [ -e $L/CONF_SMOKE_DONE ] && { echo "smoke already done"; exit 0; }
    SMOKE=1; setm qwen3.5-9b || exit 2
    say "smoke start (pid $$): helper $(sha256sum $HELP | cut -c1-16), table $(sha256sum $TABLE | cut -c1-16), launcher $(sha256sum $0 | cut -c1-16)"
    st smokeids "${RD[@]}" $HELP smoke-ids || { failed "smoke ids"; exit 1; }
    IDS=$NS/falseqa/falseqa_confirm_1128_smoke_ids.json
    lock $GCARD 8 $GNEED || { failed "smoke lock"; exit 1; }
    dq $GCARD
    entry_check || { unlock; failed "smoke entry check"; exit 1; }
    gen none $GD $BANK --arms x --only-ids $IDS || { unlock; failed "smoke generation (x)"; exit 1; }
    gen prompt $GD $BANK --arms $ARMS_W --only-ids $IDS || { unlock; failed "smoke generation"; exit 1; }
    unlock
    parts_all smokefile || { failed "smoke write records"; exit 1; }
    st checkgen "${RD[@]}" $HELP check-gen $M smoke || { failed "smoke check-gen"; exit 1; }
    judge_stage qwen3.5-9b smoke || { failed "smoke judging"; exit 1; }
    "${RD[@]}" $TABLE smoke qwen3.5-9b > $OUTS 2>> $L/confirm_1128_smoke_qwen3.5-9b.log || { failed "smoke table"; exit 1; }
    touch $L/CONF_SMOKE_DONE; say "smoke done: $OUTS"
    exit 0 ;;
  full)
    checkfiles || exit 2
    mkdir $L/.confirm_1128_running 2>/dev/null || { echo "already running ($L/.confirm_1128_running)"; exit 3; }
    trap 'rmdir $L/.confirm_1128_running 2>/dev/null' EXIT
    [ -e $L/CONF_DONE ] && { say "already done (CONF_DONE)"; exit 0; }
    [ -e $L/CONF_SMOKE_DONE ] || { echo "run the smoke first" >&2; exit 2; }
    rm -f $L/CONF_FAILED
    say "full start (pid $$): helper $(sha256sum $HELP | cut -c1-16), table $(sha256sum $TABLE | cut -c1-16), launcher $(sha256sum $0 | cut -c1-16)"
    ( for m in qwen3.5-9b gemma3-12b llama3.1-8b; do gpu_model $m 20 || { failed "gpu lane 1 at $m"; exit 1; }; done ) &
    P1=$!
    ( for m in qwen3.8-27b gemma3-27b; do gpu_model $m 40 || { failed "gpu lane 2 at $m"; exit 1; }; done ) &
    P2=$!
    lanes q100 & P3=$!
    BAD=""
    wait $P1 || BAD="$BAD lane1"
    wait $P2 || BAD="$BAD lane2"
    wait $P3 || BAD="$BAD judge-lanes"
    if [ -n "$BAD" ]; then
      [ -e $L/CONF_COST_STOP ] && say "stopped by the cost guard:$BAD" || failed "lanes:$BAD"
      exit 1
    fi
    if [ ! -e $L/CONF_TABLE100_DONE ]; then
      table q100 $OUT100 || { failed "table q100 (see $L/confirm_1128_table_q100.err)"; exit 1; }
      touch $L/CONF_TABLE100_DONE; say "table q100 $OUT100 $(sha256sum $OUT100 | cut -c1-16)"
    fi
    DEC=$(grep '^EXPAND_DECISION ' $OUT100 | sed -n 's/.*-> EXPAND \(yes\|no\)$/\1/p')
    case $DEC in
      yes) touch $L/CONF_EXPAND_YES; say "expansion: yes ($(grep '^EXPAND_DECISION ' $OUT100))" ;;
      no)  touch $L/CONF_EXPAND_NO; say "expansion: no ($(grep '^EXPAND_DECISION ' $OUT100)); chain ends"
           touch $L/CONF_DONE; exit 0 ;;
      *)   failed "cannot read the expansion decision in $OUT100"; exit 1 ;;
    esac
    lanes all || { [ -e $L/CONF_COST_STOP ] && say "stopped by the cost guard (stage all)" || failed "stage-all lanes"; exit 1; }
    table all $OUTALL || { failed "table all (see $L/confirm_1128_table_all.err)"; exit 1; }
    say "table all $OUTALL $(sha256sum $OUTALL | cut -c1-16)"
    touch $L/CONF_DONE; say "done"
    exit 0 ;;
  *) echo "usage: run_confirm_1128.sh smoke | full" >&2; exit 2 ;;
esac

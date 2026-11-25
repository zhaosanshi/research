#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor si, 2026-11-26 02 h; plan discussion/work_queue/2026-11-26_queue_o.md,
# sections 2 to 4, written before this script ran). Exploration, tune split only (192 TruthfulQA questions).
# For the five models at the main table's settings (E layer / strength, C layer; write rule "prompt"), two NEW arms:
#   x:E<l>.w.p<s>+C<c>.w.p0.25  and  x:E<l>.w.p<s>+C<c>.w.p0.5   (C toward CAUTION: in the engine the C direction
#   points from conf to caut, e44_factorial_das.family_spec plus_minus = [(caut, conf), ...], and "p" is +strength,
#   e44_eval_extra.parse_arm; N1's ladder uses the answering side .w.m<k>).
# Functions are those of pospall.sh (setm: code copies, run dirs, interpreters, container environment, engine
# arguments) and depth_queue3_1124.sh (cost cap, balance floor, off-peak guard, Pro passes), with these changes:
#   - entry scripts/review_checks/e44_positions_capilot_1126.py (= e44_positions2.py of the code copy, unchanged and
#     imported, with the pair's acceptance record written as selfcheck_<pair>-capilot-at-prompt.json, because the
#     existing record of each pair at "prompt" is a TEST-split record and must not be overwritten);
#   - per model: tune selfcheck of the pair (whole split), generate the two arms, sign check on the generation files
#     (rule detector or empty answer = blank; each caution arm may answer at most SIGNTOL more questions than single E),
#     export capilot_1126_tune, work dir <E run>/eval_extra/judge_tune_capilot_1126_tune (must not exist), cells;
#   - cards 1 and 2 only, through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, retried every 60 s); ours plus
#     others at most 66560 MiB per card; Llama-3.1-8B first on card 1 (its sign check gates the others), then card 1:
#     Gemma-3-12B, card 2: Qwen3.5-9B then Gemma-3-27B, then Qwen3.8-27B on cards 1,2 (both locks);
#   - DeepSeek Pro (the code copy's non-thinking judge config) off-peak only: no pass in weekday 08:30-12:00 and
#     14:00-18:00 Beijing time, a pass is cut by timeout before a peak and resumed; each work dir's reply cache is seeded
#     with the cache of the model's single-E tune work dir judge_tune_posprompt<tag>_tune; spending cap CAP yuan for
#     this pilot (sum of the upper estimates of crow_pro_requests.py, capilot_1126_cost_spent.txt), no batch that could
#     take the balance under FLOOR yuan;
#   - table: python3 -B Iteration_01/diag/table_capilot_1126.py > Iteration_01/diag/out_capilot_<yyyymmdd>.txt.
# Markers in Iteration_01/logs/newsets_20261119: CAPILOT_GEN_<model>_DONE (per model), CAPILOT_GEN_DONE / _FAILED,
# CAPILOT_JUDGE_<model>_DONE (per model), CAPILOT_JUDGE_DONE / _FAILED, CAPILOT_TABLE_DONE / _FAILED, CAPILOT_COST_STOP.
# Logs: capilot_1126_all.log (one line per step), capilot_1126_<model>.log, capilot_1126_judge_<model>.log.
# usage (host):  run_capilot_1126.sh            the whole chain
#                run_capilot_1126.sh dry KEY    every card command of one model with no card (CUDA_VISIBLE_DEVICES=,
#                                               timeout 90): expected ending "No CUDA GPUs are available"
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ALL=$L/capilot_1126_all.log
TF=$ROOT/envs/tf519/bin/python
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
ENTRY=$ROOT/Iteration_01/code/scripts/review_checks/e44_positions_capilot_1126.py
TABLE=Iteration_01/diag/table_capilot_1126.py
NAME=capilot_1126_tune
MAXMIB=66560; CAP=3; FLOOR=10; SIGNTOL=2
SPENT=$L/capilot_1126_cost_spent.txt
KEYS="llama q9 q27 g12 g27"
say() { echo "$(date '+%F %T') capilot: $*" >> $ALL; }

setm() {  # pospall.sh setm (paths, interpreters, environment) with the main-table E layer / strength of the task
  KEY=$1
  case $1 in
    llama) NG=llama3.1-8b; CODE=$ROOT/code_e45_20261112; PYM=python3; run=results_e44_factorial_das/llama3.1-8b_side_E2
           crun=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
           EL=16; ES=3.5; CL=19; NEED=20000; CARD=1 ;;
    q9)    NG=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.5-9b_E2
           crun=$E44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b
           EL=15; ES=2; CL=16; NEED=22000; CARD=2 ;;
    q27)   NG=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
           crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
           EL=28; ES=1.75; CL=33; NEED=34000; CARD=1,2 ;;
    g12)   NG=gemma3-12b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/12b/e44/gemma3-12b
           crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/$G3/12b/source/gemma3-12b
           EL=24; ES=2.5; CL=25; NEED=30000; CARD=1 ;;
    g27)   NG=gemma3-27b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
           crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/$G3/27b/source/gemma3-27b
           EL=28; ES=1; CL=31; NEED=62000; CARD=2 ;;
    *) say "unknown key $1"; exit 2 ;;
  esac
  TAG=l$EL
  MLOG=$L/capilot_1126_$NG.log; JLOG=$L/capilot_1126_judge_$NG.log
  WK=$run/eval_extra/judge_tune_$NAME
  SEED=$run/eval_extra/judge_tune_posprompt${TAG}_tune/judge2/cache
  ARMS="x:E$EL.w.p$ES+C$CL.w.p0.25 x:E$EL.w.p$ES+C$CL.w.p0.5"
  SCARM=x:E$EL.w.p$ES+C$CL.w.p0.5
  local dev=$CARD pre=()
  [ -n "${DRY:-}" ] && { dev=""; pre=(timeout 90); }
  DX=("${pre[@]}" docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$dev -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
  JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
      -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  COMMON=(--source-run $SRC --run-dir $ROOT/$run --e-on $E_ON --c-run-dir $ROOT/$crun --c-e-on $C_E_ON --split tune
          --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
}
st() {  # st NAME CMD...: one step, output to the model log, one line with the exit code in the common log
  local name=$1 rc; shift
  echo "=== $(date '+%F %T') $NG $name: $*" >> $MLOG
  "$@" >> $MLOG 2>&1; rc=$?
  echo "$(date '+%F %T') capilot: $NG $name exit=$rc" >> $ALL
  return $rc
}
fname() { local f=${1//:/--}; f=${f//+/-and-}; f=${f//./-}; echo "${f}-at-prompt_tune.json"; }
memwait() {  # memwait CARD MIB
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}')
    [[ $used =~ ^[0-9]+$ ]] || return 1
    [ $((used + $2)) -le $MAXMIB ] && { say "$NG: card $1 holds $used MiB, asking $2"; return 0; }
    say "$NG: card $1 holds $used MiB, $2 more would exceed $MAXMIB; waiting"; sleep 60
  done
}
signcheck() {  # answered questions (rule detector off and not empty) of single E and the two caution arms
  python3 - "$HOST/$run/eval_extra/gen" "$(fname x:E$EL.w.p$ES)" $SIGNTOL $(for a in $ARMS; do fname $a; done) <<'EOF'
import json, os, sys
gen, efile, tol, arms = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
def load(n):
    d = json.load(open(os.path.join(gen, n), encoding="utf-8"))
    if d.get("completed") is not True or d.get("partial") is not False or d.get("split") != "tune":
        sys.exit("%s is not a complete tune file" % n)
    return d
e = load(efile)
ids = [r["pair_id"] for r in e["rows"]]
ans = lambda d: sum(1 for r in d["rows"] if not r["refusal_rule"] and r["text"].strip())
ne = ans(e)
print("single E %s: %d questions, answered %d; parts %s" % (efile, len(ids), ne, [(p["family"], p["layer"], p["beta"]) for p in e["parts"]]))
ok = True
for n in arms:
    d = load(n)
    if [r["pair_id"] for r in d["rows"]] != ids:
        sys.exit("%s covers other questions" % n)
    parts = [(p["family"], p["layer"], p["beta"]) for p in d["parts"]]
    if len(parts) != 2 or parts[0] != (e["parts"][0]["family"], e["parts"][0]["layer"], e["parts"][0]["beta"]) or parts[1][0] != "C" or parts[1][2] <= 0:
        sys.exit("%s: parts %s are not single E plus C with a positive strength" % (n, parts))
    na = ans(d)
    good = na <= ne + tol
    ok &= good
    print("%s: parts %s, answered %d (single E %d, difference %+d) [%s]" % (n, parts, na, ne, na - ne, "ok" if good else "MORE THAN SINGLE E: SIGN"))
sys.exit(0 if ok else 1)
EOF
}

gen_model() {  # gen_model KEY: tune selfcheck, two arms, sign check, export, cells (cards from setm)
  setm $1
  [ -e $L/CAPILOT_GEN_${NG}_DONE ] && { say "$NG: generation already done"; return 0; }
  [ -e $HOST/$WK ] && { say "$NG: work dir $WK exists before generation; stop"; return 1; }
  for a in $ARMS; do [ -e $HOST/$run/eval_extra/gen/$(fname $a) ] && { say "$NG: $(fname $a) exists before generation; stop"; return 1; }; done
  [ -s $HOST/$run/eval_extra/gen/$(fname x:E$EL.w.p$ES) ] || { say "$NG: no single-E file $(fname x:E$EL.w.p$ES)"; return 1; }
  local g fd=8
  for g in ${CARD//,/ }; do
    eval "exec $fd>\"$HOST/Iteration_01/newsets/selfjudge/.gpu$g.lock\"" || { say "$NG: cannot open the lock of card $g"; return 1; }
    until flock -n $fd; do sleep 60; done
    say "$NG: holds the lock of card $g"
    fd=$((fd + 1))
  done
  for g in ${CARD//,/ }; do memwait $g $NEED || { say "$NG: cannot read the memory of card $g"; return 1; }; done
  say "$NG: card part starts on card(s) $CARD; arms $ARMS"
  st selfcheck "${DX[@]}" $PYM $ENTRY prompt selfcheck "${COMMON[@]}" --arms $SCARM || { say "$NG: selfcheck FAILED"; return 1; }
  st generate "${DX[@]}" $PYM $ENTRY prompt generate "${COMMON[@]}" --arms $ARMS || { say "$NG: generation FAILED"; return 1; }
  st signcheck signcheck || { say "$NG: sign check FAILED (see $MLOG)"; tail -n 3 $MLOG | sed 's/^/    /' >> $ALL; return 1; }
  tail -n 3 $MLOG | sed 's/^/    /' >> $ALL
  local gens="" a
  for a in $ARMS; do gens="$gens $(fname $a)"; done
  st export "${DX[@]}" $PYM $ENTRY prompt export "${COMMON[@]}" --gen $gens --name $NAME || { say "$NG: export FAILED"; return 1; }
  mkdir -p $HOST/$WK/judge2/cache && cp $HOST/$run/eval_extra/judge_tune/$NAME.json $HOST/$WK/judge_input.json || { say "$NG: work dir FAILED"; return 1; }
  st cells "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$WK --generation judge_input.json || { say "$NG: cells FAILED"; return 1; }
  for g in $(seq 8 $((fd - 1))); do flock -u $g; eval "exec $g>&-"; done
  say "$NG: card(s) $CARD released"
  touch $L/CAPILOT_GEN_${NG}_DONE
}

budget() {  # seconds this lane may judge now (0 = peak hour: weekdays 08:30-12:00 and 14:00-18:00)
  local dow=$(date +%u) hm=$((10#$(date +%H%M))) now=$(date +%s) t
  if [ $dow -ge 6 ]; then echo 7200; return; fi
  if [ $hm -lt 830 ]; then t=$(( $(date -d "08:30" +%s) - now ))
  elif [ $hm -lt 1200 ]; then t=0
  elif [ $hm -lt 1400 ]; then t=$(( $(date -d "14:00" +%s) - now ))
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
nverdict() {
  "${JX[@]}" python3 -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
ok = [r for r in rows if r['status'] == 'scored' or r.get('basis') in ('empty_output_no_judge_vote', 'pro_declined')]
print(len(rows) - len(ok))" "$ROOT/$WK/judge2/single_pro.jsonl"
}
single_pro() { "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$WK --judge pro > $HOST/$WK/single_pro_stdout.md; }
judge() {  # judge KEY: DeepSeek Pro on the model's capilot work dir, off-peak, within the cap
  setm $1
  local MLOG=$JLOG cost=$L/capilot_1126_cost_$NG.txt secs b sp up rc tries=0 left
  [ -f $HOST/$WK/judge2/cells.jsonl ] || { say "$NG: no cells in $WK"; return 1; }
  [ -f $HOST/$SEED/judge2_cache_deepseek-v4-pro.jsonl ] || { say "$NG: no Pro reply cache in $SEED"; return 1; }
  st merge "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$WK/judge2/cache $ROOT/$SEED || { say "$NG: cache merge FAILED"; return 1; }
  st conflicts "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$WK/judge2/cache/conflicts.json || { say "$NG: cache conflicts"; return 1; }
  while true; do
    secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
    [ -e $L/CAPILOT_COST_STOP ] && return 9
    "${JX[@]}" python3 -B $ROOT/Iteration_01/diag/crow_pro_requests.py $ROOT/$WK > $cost 2>> $JLOG || { say "$NG: request count FAILED"; return 1; }
    up=$(sed -n 's/^expected cost .*, upper \([0-9.]*\) CNY.*/\1/p' $cost)
    [[ $up =~ ^[0-9]+\.[0-9]+$ ]] || { say "$NG: cannot read the cost estimate in $cost"; return 1; }
    b=$(bal); [[ $b =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "$NG: cannot read the balance"; return 1; }
    sp=0; [ -s $SPENT ] && sp=$(cat $SPENT)
    [[ $sp =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "cannot read $SPENT"; return 1; }
    if python3 -c 'import sys; sp, b, up, cap, floor = map(float, sys.argv[1:]); sys.exit(0 if (sp + up > cap or b - up < floor) else 1)' $sp $b $up $CAP $FLOOR; then
      say "COST STOP before $NG: earlier batches at most $sp yuan, this batch at most $up, cap $CAP; balance $b, floor $FLOOR"
      touch $L/CAPILOT_COST_STOP
      return 9
    fi
    python3 -c 'import sys; print("%.4f" % (float(sys.argv[1]) + float(sys.argv[2])))' $sp $up > $SPENT
    say "$NG: Pro batch (at most $secs s); balance $b; earlier batches at most $sp yuan; $(sed -n 3p $cost)"
    st pro "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$WK --judge pro
    rc=$?
    say "$NG: balance after the batch $(bal)"
    case $rc in
      0) st single single_pro || { say "$NG: single (pro) FAILED"; return 1; }
         left=$(nverdict)
         [[ $left =~ ^[0-9]+$ ]] || { say "$NG: cannot read the verdicts"; return 1; }
         [ "$left" = 0 ] && { touch $L/CAPILOT_JUDGE_${NG}_DONE; return 0; }
         tries=$((tries + 1))
         [ $tries -ge 2 ] && { say "$NG: $left cells without a Pro verdict after $tries complete passes"; return 1; }
         say "$NG: $left cells without a verdict; one more pass" ;;
      124) say "$NG: pass cut before a peak hour; resumes off-peak" ;;
      *) say "$NG: the Pro pass FAILED (rc $rc)"; return 1 ;;
    esac
  done
}
judge_lane() {  # judges every model whose generation is done, in the order they finish; ends when all five are judged
  local k left gfail
  while true; do
    left=0
    for k in $KEYS; do
      setm $k
      [ -e $L/CAPILOT_JUDGE_${NG}_DONE ] && continue
      if [ -e $L/CAPILOT_GEN_${NG}_DONE ]; then
        judge $k || { say "$NG: judging stopped (rc $?)"; touch $L/CAPILOT_JUDGE_FAILED; return 1; }
        say "$NG: judged"
      else
        left=$((left + 1))
      fi
    done
    [ $left = 0 ] && { touch $L/CAPILOT_JUDGE_DONE; say "all five judged"; return 0; }
    gfail=0; [ -e $L/CAPILOT_GEN_FAILED ] && gfail=1
    [ $gfail = 1 ] && [ -e $L/.capilot_gen_ended ] && { say "generation failed; $left models not judged"; touch $L/CAPILOT_JUDGE_FAILED; return 1; }
    sleep 60
  done
}

if [ "${1:-}" = dry ]; then
  DRY=1; setm ${2:?KEY}
  MLOG=$L/capilot_1126_dry_$NG.log
  for ph in selfcheck generate; do
    a=$SCARM; [ $ph = generate ] && a=$ARMS
    echo "=== $(date '+%F %T') dry $NG $ph" >> $MLOG
    "${DX[@]}" $PYM $ENTRY prompt $ph "${COMMON[@]}" --arms $a >> $MLOG 2>&1
    echo "dry $NG $ph exit=$? last line: $(grep -v '^\s*$' $MLOG | tail -n 1)"
  done
  exit 0
fi

mkdir $L/.capilot_running 2>/dev/null || { echo "already running ($L/.capilot_running)"; exit 3; }
trap 'rmdir $L/.capilot_running 2>/dev/null' EXIT
for f in $TABLE Iteration_01/diag/crow_pro_requests.py $L/balance.sh ${ENTRY#$ROOT/}; do
  [ -s $f ] || { say "missing $f"; touch $L/CAPILOT_GEN_FAILED; exit 2; }
done
for k in $KEYS; do
  setm $k
  for f in e44_positions.py e44_positions2.py; do
    cmp -s Iteration_01/code/scripts/review_checks/$f ${CODE#$ROOT/}/scripts/review_checks/$f || { say "$NG: ${CODE#$ROOT/} has another $f; stop"; touch $L/CAPILOT_GEN_FAILED; exit 2; }
  done
done
rm -f $L/.capilot_gen_ended
say "start (pid $$)"
( judge_lane ) &
JL=$!
gen_model llama || { touch $L/CAPILOT_GEN_FAILED $L/.capilot_gen_ended; say "llama failed: nothing else is generated"; wait $JL; exit 1; }
( gen_model g12 || touch $L/CAPILOT_GEN_FAILED ) &
P1=$!
( gen_model q9 && gen_model g27 || touch $L/CAPILOT_GEN_FAILED ) &
P2=$!
wait $P1 $P2
if [ -e $L/CAPILOT_GEN_FAILED ]; then
  say "a model failed on cards 1 / 2; Qwen3.8-27B is not started"
else
  gen_model q27 || touch $L/CAPILOT_GEN_FAILED
fi
[ -e $L/CAPILOT_GEN_FAILED ] || { touch $L/CAPILOT_GEN_DONE; say "generation done for the five models"; }
touch $L/.capilot_gen_ended
wait $JL
if [ -e $L/CAPILOT_JUDGE_DONE ]; then
  d=$(date +%Y%m%d)
  out=Iteration_01/diag/out_capilot_$d.txt
  if python3 -B $TABLE > $out 2>&1; then touch $L/CAPILOT_TABLE_DONE; say "table $out"; else touch $L/CAPILOT_TABLE_FAILED; say "table FAILED (see $out)"; fi
fi
say "chain ended"

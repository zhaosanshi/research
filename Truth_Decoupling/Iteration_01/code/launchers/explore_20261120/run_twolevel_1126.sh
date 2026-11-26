#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor wei, 2026-11-26 16 h; plan discussion/work_queue/2026-11-26_queue_o.md,
# section 6 "two-level E", written 15:54 before this script). Exploration, tune split only (192 TruthfulQA questions).
# For the models whose second-level single-E answers at the write rule "prompt" do not exist yet on tune, NEW single-
# write arms x:E<l>.w.p<s2> (E layer = main-table layer):
#   qwen3.5-9b  x:E15.w.p3          (2.5 exists: judge_tune_depthE_tune)
#   qwen3.8-27b x:E28.w.p2.5, p3
#   gemma3-12b  x:E24.w.p3          (4 exists: judge_tune_pospromptl24_tune)
#   gemma3-27b  x:E28.w.p1.5, p2
#   llama3.1-8b nothing             (5 and 7 exist: judge_tune_pospromptl16_tune)
# Functions are those of run_capilot_1126.sh (setm paths, interpreters, container environment, engine arguments, locks,
# memory check, cost cap, balance floor, off-peak guard, Pro passes), with these changes:
#   - entry scripts/review_checks/e44_positions.py prompt (single writes; no acceptance record needed, as in pospall.sh
#     and depth_queue_1124.sh) of the model's own code copy, which must be byte-identical to Iteration_01/code's;
#   - per model: generate the arms, check the generation files (complete, tune, positions prompt, one E part at the
#     layer and strength, the questions of the main single-E file), export twolevel_1126_l<l>_tune, work dir
#     <E run>/eval_extra/judge_tune_twolevel_1126_l<l>_tune (must not exist), cells;
#   - cards 1 and 2 through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, retried every 60 s); ours plus others at
#     most 66560 MiB per card; card 1: Gemma-3-12B then Gemma-3-27B, card 2: Qwen3.5-9B, then Qwen3.8-27B on cards 1,2;
#   - Pro reply cache of each work dir seeded with the model's main single-E tune work dir judge_tune_posprompt<tag>_tune;
#     cap CAP yuan for this task (twolevel_1126_cost_spent.txt), no batch that could take the balance under FLOOR;
#   - table: python3 -B Iteration_01/diag/table_twolevel_1126.py > Iteration_01/diag/out_twolevel_1126_tune_<yyyymmdd>.txt
# Markers in Iteration_01/logs/newsets_20261119: TWOLEVEL_GEN_<model>_DONE (per model), TWOLEVEL_GEN_DONE, TWOLEVEL_FAILED,
# TWOLEVEL_JUDGE_<model>_DONE, TWOLEVEL_JUDGE_DONE, TWOLEVEL_TABLE_DONE, TWOLEVEL_COST_STOP.
# Logs: one line per step in crow_all.log (prefix "twolevel:"), twolevel_1126_<model>.log, twolevel_1126_judge_<model>.log.
# usage (host):  run_twolevel_1126.sh            the whole chain
#                run_twolevel_1126.sh dry KEY    the generate command of one model with no card (CUDA_VISIBLE_DEVICES=,
#                                                timeout 90): expected ending "No CUDA GPUs are available"
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ALL=$L/crow_all.log
TF=$ROOT/envs/tf519/bin/python
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
ENTRY=scripts/review_checks/e44_positions.py
TABLE=Iteration_01/diag/table_twolevel_1126.py
MAXMIB=66560; CAP=2; FLOOR=10
SPENT=$L/twolevel_1126_cost_spent.txt
KEYS="q9 q27 g12 g27"
say() { echo "$(date '+%F %T') twolevel: $*" >> $ALL; }
fail() { say "FAILED: $*"; touch $L/TWOLEVEL_FAILED; }

setm() {  # run_capilot_1126.sh setm with the second-level strengths of the task
  KEY=$1
  case $1 in
    q9)    NG=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.5-9b_E2
           crun=$E44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b
           EL=15; ES=2; S2="3"; NEED=22000; CARD=2 ;;
    q27)   NG=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
           crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
           EL=28; ES=1.75; S2="2.5 3"; NEED=34000; CARD=1,2 ;;
    g12)   NG=gemma3-12b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/12b/e44/gemma3-12b
           crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/$G3/12b/source/gemma3-12b
           EL=24; ES=2.5; S2="3"; NEED=30000; CARD=1 ;;
    g27)   NG=gemma3-27b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
           crun=$run; E_ON=E2; C_E_ON=E2; SRC=$ROOT/$G3/27b/source/gemma3-27b
           EL=28; ES=1; S2="1.5 2"; NEED=62000; CARD=1 ;;
    *) say "unknown key $1"; exit 2 ;;
  esac
  TAG=l$EL
  NAME=twolevel_1126_${TAG}_tune
  MLOG=$L/twolevel_1126_$NG.log; JLOG=$L/twolevel_1126_judge_$NG.log
  WK=$run/eval_extra/judge_tune_$NAME
  SEED=$run/eval_extra/judge_tune_posprompt${TAG}_tune/judge2/cache
  ARMS=""; local s; for s in $S2; do ARMS="$ARMS x:E$EL.w.p$s"; done; ARMS=${ARMS# }
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
  echo "$(date '+%F %T') twolevel: $NG $name exit=$rc" >> $ALL
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
gencheck() {  # the new generation files: complete tune files at "prompt", one E part at (EL, s2), the main file's questions
  python3 - "$HOST/$run/eval_extra/gen" "$(fname x:E$EL.w.p$ES)" $EL "$S2" <<'EOF'
import json, os, sys
gen, efile, el, s2 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split()
def load(n):
    d = json.load(open(os.path.join(gen, n), encoding="utf-8"))
    if d.get("completed") is not True or d.get("partial") is not False or d.get("split") != "tune" or d.get("positions") != "prompt":
        sys.exit("%s is not a complete tune file at the prompt rule" % n)
    return d
e = load(efile)
ids = [r["pair_id"] for r in e["rows"]]
for s in s2:
    n = "x--E%d-w-p%s-at-prompt_tune.json" % (el, s.replace(".", "-"))
    d = load(n)
    if [r["pair_id"] for r in d["rows"]] != ids or len(ids) != 192:
        sys.exit("%s covers other questions" % n)
    parts = [(p["family"], p["layer"], p["kind"], p["beta"], p["e_on"]) for p in d["parts"]]
    if parts != [("E", el, "w", float(s), e["parts"][0]["e_on"])]:
        sys.exit("%s: parts %s" % (n, parts))
    na = sum(1 for r in d["rows"] if not r["refusal_rule"] and r["text"].strip())
    ne = sum(1 for r in e["rows"] if not r["refusal_rule"] and r["text"].strip())
    print("%s: parts %s, answered %d (main single E %d)" % (n, parts, na, ne))
EOF
}

gen_model() {  # gen_model KEY: generate, check, export, cells (cards from setm)
  setm $1
  [ -e $L/TWOLEVEL_GEN_${NG}_DONE ] && { say "$NG: generation already done"; return 0; }
  [ -e $HOST/$WK ] && { say "$NG: work dir $WK exists before generation; stop"; return 1; }
  [ -e $HOST/$run/eval_extra/judge_tune/$NAME.json ] && { say "$NG: export $NAME.json exists before generation; stop"; return 1; }
  for a in $ARMS; do [ -e $HOST/$run/eval_extra/gen/$(fname $a) ] && { say "$NG: $(fname $a) exists before generation; stop"; return 1; }; done
  [ -s $HOST/$run/eval_extra/gen/$(fname x:E$EL.w.p$ES) ] || { say "$NG: no main single-E file $(fname x:E$EL.w.p$ES)"; return 1; }
  local g fd=8
  for g in ${CARD//,/ }; do
    eval "exec $fd>\"$HOST/Iteration_01/newsets/selfjudge/.gpu$g.lock\"" || { say "$NG: cannot open the lock of card $g"; return 1; }
    until flock -n $fd; do sleep 60; done
    say "$NG: holds the lock of card $g"
    fd=$((fd + 1))
  done
  for g in ${CARD//,/ }; do memwait $g $NEED || { say "$NG: cannot read the memory of card $g"; return 1; }; done
  say "$NG: card part starts on card(s) $CARD; arms $ARMS"
  st generate "${DX[@]}" $PYM $ENTRY prompt generate "${COMMON[@]}" --arms $ARMS || { say "$NG: generation FAILED"; return 1; }
  st gencheck gencheck || { say "$NG: generation check FAILED (see $MLOG)"; tail -n 3 $MLOG | sed 's/^/    /' >> $ALL; return 1; }
  tail -n $(echo $S2 | wc -w) $MLOG | sed 's/^/    /' >> $ALL
  local gens="" a
  for a in $ARMS; do gens="$gens $(fname $a)"; done
  st export "${DX[@]}" $PYM $ENTRY prompt export "${COMMON[@]}" --gen $gens --name $NAME || { say "$NG: export FAILED"; return 1; }
  mkdir -p $HOST/$WK/judge2/cache && cp $HOST/$run/eval_extra/judge_tune/$NAME.json $HOST/$WK/judge_input.json || { say "$NG: work dir FAILED"; return 1; }
  st cells "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$WK --generation judge_input.json || { say "$NG: cells FAILED"; return 1; }
  for g in $(seq 8 $((fd - 1))); do flock -u $g; eval "exec $g>&-"; done
  say "$NG: card(s) $CARD released"
  touch $L/TWOLEVEL_GEN_${NG}_DONE
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
judge() {  # judge KEY: DeepSeek Pro on the model's twolevel work dir, off-peak, within the cap
  setm $1
  local MLOG=$JLOG cost=$L/twolevel_1126_cost_$NG.txt secs b sp up rc tries=0 left
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
    [ -e $L/TWOLEVEL_COST_STOP ] && return 9
    "${JX[@]}" python3 -B $ROOT/Iteration_01/diag/crow_pro_requests.py $ROOT/$WK > $cost 2>> $JLOG || { say "$NG: request count FAILED"; return 1; }
    up=$(sed -n 's/^expected cost .*, upper \([0-9.]*\) CNY.*/\1/p' $cost)
    [[ $up =~ ^[0-9]+\.[0-9]+$ ]] || { say "$NG: cannot read the cost estimate in $cost"; return 1; }
    b=$(bal); [[ $b =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "$NG: cannot read the balance"; return 1; }
    sp=0; [ -s $SPENT ] && sp=$(cat $SPENT)
    [[ $sp =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "cannot read $SPENT"; return 1; }
    if python3 -c 'import sys; sp, b, up, cap, floor = map(float, sys.argv[1:]); sys.exit(0 if (sp + up > cap or b - up < floor) else 1)' $sp $b $up $CAP $FLOOR; then
      say "COST STOP before $NG: earlier batches at most $sp yuan, this batch at most $up, cap $CAP; balance $b, floor $FLOOR"
      touch $L/TWOLEVEL_COST_STOP
      return 9
    fi
    python3 -c 'import sys; print("%.4f" % (float(sys.argv[1]) + float(sys.argv[2])))' $sp $up > $SPENT
    say "$NG: Pro batch (at most $secs s); balance $b; earlier batches at most $sp yuan; $(sed -n 2,3p $cost | tr "\n" " ")"
    st pro "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$WK --judge pro
    rc=$?
    say "$NG: balance after the batch $(bal)"
    case $rc in
      0) st single single_pro || { say "$NG: single (pro) FAILED"; return 1; }
         left=$(nverdict)
         [[ $left =~ ^[0-9]+$ ]] || { say "$NG: cannot read the verdicts"; return 1; }
         [ "$left" = 0 ] && { touch $L/TWOLEVEL_JUDGE_${NG}_DONE; return 0; }
         tries=$((tries + 1))
         [ $tries -ge 2 ] && { say "$NG: $left cells without a Pro verdict after $tries complete passes"; return 1; }
         say "$NG: $left cells without a verdict; one more pass" ;;
      124) say "$NG: pass cut before a peak hour; resumes off-peak" ;;
      *) say "$NG: the Pro pass FAILED (rc $rc)"; return 1 ;;
    esac
  done
}

if [ "${1:-}" = dry ]; then
  DRY=1; setm ${2:?KEY}
  MLOG=$L/twolevel_1126_dry_$NG.log
  echo "=== $(date '+%F %T') dry $NG generate" >> $MLOG
  "${DX[@]}" $PYM $ENTRY prompt generate "${COMMON[@]}" --arms $ARMS >> $MLOG 2>&1
  echo "dry $NG generate exit=$? last line: $(grep -v '^\s*$' $MLOG | tail -n 1)"
  exit 0
fi

mkdir $L/.twolevel_1126_running 2>/dev/null || { echo "already running ($L/.twolevel_1126_running)"; exit 3; }
trap 'rmdir $L/.twolevel_1126_running 2>/dev/null' EXIT
for f in $TABLE Iteration_01/diag/crow_pro_requests.py $L/balance.sh; do
  [ -s $f ] || { fail "missing $f"; exit 2; }
done
for k in $KEYS; do
  setm $k
  cmp -s Iteration_01/code/$ENTRY ${CODE#$ROOT/}/$ENTRY || { fail "$NG: ${CODE#$ROOT/} has another $ENTRY"; exit 2; }
done
say "start (pid $$); llama3.1-8b: nothing to generate (E16 at 5 and 7 judged in judge_tune_pospromptl16_tune)"
touch $L/TWOLEVEL_GEN_llama3.1-8b_DONE
# generation: card 1 Gemma-3-12B then Gemma-3-27B, card 2 Qwen3.5-9B; then Qwen3.8-27B on cards 1,2
( gen_model g12 && gen_model g27 || touch $L/.twolevel_1126_gen_failed ) &
P1=$!
( gen_model q9 || touch $L/.twolevel_1126_gen_failed ) &
P2=$!
wait $P1 $P2
[ -e $L/.twolevel_1126_gen_failed ] && { fail "generation on card 1 / 2; Qwen3.8-27B is not started"; exit 1; }
gen_model q27 || { fail "generation of qwen3.8-27b"; exit 1; }
touch $L/TWOLEVEL_GEN_DONE; say "generation done"
# judging (off-peak only, see budget), one model after the other
for k in $KEYS; do
  setm $k
  [ -e $L/TWOLEVEL_JUDGE_${NG}_DONE ] && continue
  judge $k || { fail "judging of $NG stopped (rc $?)"; exit 1; }
  say "$NG: judged"
done
touch $L/TWOLEVEL_JUDGE_DONE; say "all judged"
d=$(date +%Y%m%d)
out=Iteration_01/diag/out_twolevel_1126_tune_$d.txt
if python3 -B $TABLE > $out 2>&1; then
  touch $L/TWOLEVEL_TABLE_DONE; say "table $out $(sha256sum $out | cut -c1-16)"
else
  fail "table (see $out)"; exit 1
fi
say "chain ended"

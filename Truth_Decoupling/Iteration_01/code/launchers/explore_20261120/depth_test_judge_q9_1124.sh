#!/bin/bash
# Iteration_01 (main session, Claude Fable 5.2, 2026-11-24 20 h; plan discussion/work_queue/2026-11-24_queue_n.md,
# progress entry of 20:06, whose selection rule, reading and predictions were committed before this script ran):
# DeepSeek Pro (non-thinking, off-peak only) on the test-split work dir of the late single-E setting of Llama-3.1-8B
# (<E run>/eval_extra/judge_test_depthE_test, written by depth_confirm_1124.sh). This copy: Qwen3.5-9B (work dir
# written by depth_late_q9_1124.sh, seed judge_test_pospromptl15_test; plan entry of 20:50). The functions are those of
# depth_queue3_1124.sh, copied by program with three changes: the work dirs are judge_test_*, the failure marker is
# DEPTH_TESTJUDGE_<model>_FAILED, the guard directory is .depth_testjudge_running. The reply cache is seeded with the
# test-split cache of the main-table single E (judge_test_pospromptl16_test). Spending is added to the plan's
# depth_cost_spent.txt (cap 15 yuan for the plan, floor 10 yuan of balance). Marker DEPTH_TESTJUDGE_<model>_DONE.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/depth_test_judge_q9_1124.sh \
#                 > Iteration_01/logs/newsets_20261119/depth_test_judge_q9.out 2>&1 < /dev/null &
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ALL=$L/depth_all.log
DS="python3 -B Iteration_01/diag/depth_scan.py"
CAPD=Iteration_01/capability/depthscan
LOCK=Iteration_01/newsets/selfjudge/.gpu0.lock
CARD=0
MAXMIB=66560
CAP=15
FLOOR=10
SPLIT=test
KS="0.25 0.5 1 2"
LADDER="0.5 0.75 1 1.5 2 2.5 3 3.5 4 5 6 8"
say() { echo "$(date '+%F %T') depth: $*" >> $ALL; }
mkdir $L/.depth_testjudge_q9_running 2>/dev/null || { echo "already running ($L/.depth_testjudge_q9_running)"; exit 3; }
trap 'rmdir $L/.depth_testjudge_q9_running 2>/dev/null' EXIT
for f in Iteration_01/diag/depth_scan.py Iteration_01/diag/crow_pro_requests.py $L/balance.sh; do
  [ -s $f ] || { say "missing $f"; exit 2; }
done

setm() {
  KEY=$1; TRAINL=""; NEEDT=""
  case $1 in
    llama) NG=llama3.1-8b; CODE=$W/code_e45_20261112; PYM=python3; RUN=results_e44_factorial_das/llama3.1-8b_side_E2
           CRUN=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; SRC=$W/results_qa_das_v1u/llama3.1-8b/full
           EL=16; ES=3.5; TAG=l16; NEED=20000; NEWC="17 18 20 21"; LAYERS="10 12 14 15 16 17 18 20 22 24 26" ;;
    q9)    NG=qwen3.5-9b; CODE=$W/code_ff_explore_20261113b; PYM=$W/envs/tf519/bin/python; RUN=results_newmodels_ff_20261112/e44/qwen3.5-9b_E2
           CRUN=results_newmodels_ff_20261112/e44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; SRC=$W/results_newmodels_ff_20261112/source/qwen3.5-9b
           EL=15; ES=2; TAG=l15; NEED=22000; NEEDT=30000; NEWC="17 18 19"; LAYERS="12 13 14 15 16 17 18 20 22 24"
           TRAINL="20 22 24"; TCODE=$W/code_ff_explore_20261113; TREF=15 ;;
    *) say "unknown key $1"; exit 2 ;;
  esac
  MLOG=$L/depth_testjudge_$NG.log; JLOG=$L/depth_testjudge_$NG.log
  DX=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models
      -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES=$CARD -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
  JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
      -e DEEPSEEK_API_KEY_FILE=$W/logs/.deepseek_key -w $CODE truth_dec)
  COMMON=(--source-run $SRC --run-dir $W/$RUN --e-on $E_ON --c-run-dir $W/$CRUN --c-e-on $C_E_ON --split tune
          --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
}
st() {  # st NAME CMD...: one step, output to the model log, one line with the exit code in depth_all.log
  local name=$1 rc; shift
  echo "=== $(date '+%F %T') $NG $name: $*" >> $MLOG
  "$@" >> $MLOG 2>&1; rc=$?
  echo "$(date '+%F %T') $NG $name exit=$rc" >> $ALL
  return $rc
}
fail() { say "$NG test judging FAILED: $* (see $MLOG)"; echo "$(date '+%F %T') FAILED: $*" >> $MLOG; touch $L/DEPTH_TESTJUDGE_${NG}_FAILED; }
fname() { local f=${1//:/--}; f=${f//+/-and-}; f=${f//./-}; echo "${f}-at-prompt_tune.json"; }
memwait() {  # memwait MIB: until ours plus what card 1 holds stays within MAXMIB (checked once a minute)
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$CARD '$1 == g {print $2}')
    [[ $used =~ ^[0-9]+$ ]] || return 1
    [ $((used + $1)) -le $MAXMIB ] && { say "$NG: card $CARD holds $used MiB, asking $1"; return 0; }
    say "$NG: card $CARD holds $used MiB, $1 more would exceed $MAXMIB; waiting"; sleep 60
  done
}
mmlu() {  # mmlu OUTSTEM ARMS...: cap_mcq_sweep.py on the 1,000-item list (as layer_probe.sh)
  local out=$CAPD/$1.json; shift
  [ -s $out ] && { echo "$out exists, kept" >> $MLOG; return 0; }
  docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$CARD -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $W truth_dec \
    $PYM -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model $NG --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_1000_20261119.json \
    --out $out --arms "$@"
}
accept_train() {  # accept_train LAYER: the new train log against layer TREF of the same run
  python3 -c '
import hashlib, json, os, sys
run, l, ref = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a = json.load(open("%s/train_logs/E_L%d.json" % (run, l))); b = json.load(open("%s/train_logs/E_L%d.json" % (run, ref)))
ok = True
for k in ("identity", "identity_sha256", "protocol", "e_on", "frozen_source"):
    same = a.get(k) == b.get(k)
    ok &= same
    print("  L%d %s: %s" % (l, k, "same" if same else "DIFFERENT"))
ta, tb = dict(a["training_contract"]), dict(b["training_contract"])
la, lb = ta.pop("layer"), tb.pop("layer")
diff = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
ok &= not diff and la == l and a.get("layer") == l and a.get("pilot") is False and a.get("family") == "E"
print("  L%d training_contract: %d keys besides layer, differing %s; layer %s; pilot %s; family %s" % (l, len(ta), diff or "none", la, a.get("pilot"), a.get("family")))
p = "%s/dirs/%s" % (run, a.get("direction_file"))
sha = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
ok &= sha is not None and sha == a.get("direction_file_sha256") and a.get("direction_file") == "E_L%d.pt" % l
print("  L%d direction file %s: sha256 %s" % (l, a.get("direction_file"), "matches the log" if sha == a.get("direction_file_sha256") else "MISSING OR DIFFERENT"))
sys.exit(0 if ok else 1)' $RUN $1 $TREF
}

gpu_model() {  # gpu_model KEY: everything of one model that needs card 1
  setm $1
  rm -f $L/DEPTH_${NG}_FAILED
  [ -e $L/DEPTH2_${NG}_GEN_DONE ] && { say "$NG: card part already done"; return 0; }
  say "$NG: card part starts"
  local need=$NEED tl="" l a arms gens wk
  if [ -n "$TRAINL" ]; then
    for l in $TRAINL; do [ -s $RUN/dirs/E_L$l.pt ] && [ -s $RUN/train_logs/E_L$l.json ] || tl="$tl $l"; done
    tl=${tl# }
    [ -n "$tl" ] && need=$NEEDT
  fi
  memwait $need || { fail "cannot read the memory of card $CARD"; return 1; }
  # a: training (q9)
  if [ -n "$tl" ]; then
    local tpym
    tpym=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("python", "python3"))' ${TCODE#$W/}/configs/models/$NG.json) \
      || { fail "python of the model config in ${TCODE#$W/}"; return 1; }
    st train_E docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models \
      -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES=$CARD -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$TCODE -e DEEPSEEK_API_KEY_FILE=$W/logs/.deepseek_key \
      -w $TCODE truth_dec "$tpym" scripts/e44_factorial_das.py train --source-run $SRC --batch-size 8 --max-new-tokens 48 \
      --max-target-tokens 24 --e-on E2 --run-dir $W/$RUN --family E --layers $tl --val-n 16 \
      || { fail "training of E layers $tl"; return 1; }
  fi
  for l in $TRAINL; do st accept_E$l accept_train $l || { fail "acceptance of the train log of E layer $l"; return 1; }; done
  # b: entry checks
  mkdir -p $CAPD
  st mmlu_x mmlu ${NG}_x x || { fail "MMLU arm x"; return 1; }
  st check_x $DS checkx $KEY || { fail "entry check 1: MMLU x differs from the stored reference"; return 1; }
  st regen16 "${DX[@]}" $PYM scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --first-batches 2 \
    --arms x:E$EL.w.p$ES || { fail "entry check 2: regeneration of x:E$EL.w.p$ES"; return 1; }
  st check_gen $DS checkgen $KEY || { fail "entry check 2: the 16 regenerated answers differ from the archive"; return 1; }
  # c: MMLU sweep and the rule of plan 2.1
  if [ ! -e $L/DEPTH1_${NG}_MMLU_DONE ]; then
    for l in $LAYERS; do
      arms=""; for a in $LADDER; do arms="$arms x:E$l.w.p$a"; done
      st mmlu_L$l mmlu ${NG}_L$l x $arms || { fail "MMLU sweep of layer $l"; return 1; }
    done
    st choose $DS choose $KEY || { fail "choice of the strengths"; return 1; }
    for l in $($DS need-ext $KEY); do
      st mmlu_L${l}_ext mmlu ${NG}_L${l}_ext x x:E$l.w.p12 x:E$l.w.p16 || { fail "MMLU strengths 12 / 16 of layer $l"; return 1; }
    done
    st choose2 $DS choose $KEY || { fail "choice of the strengths after 12 / 16"; return 1; }
    [ -z "$($DS need-ext $KEY)" ] || { fail "layers still without a choice after 12 / 16"; return 1; }
    touch $L/DEPTH1_${NG}_MMLU_DONE
  fi
  # d: stage-1 generation, export, cells
  if [ ! -e $L/DEPTH1_${NG}_GEN_DONE ]; then
    arms=$($DS arms $KEY) || { fail "arm list of stage 1"; return 1; }
    [ -n "$arms" ] || { fail "no layer has a strength inside the budget"; return 1; }
    gens=""; for a in $arms; do gens="$gens $(fname $a)"; done
    wk=$RUN/eval_extra/judge_tune_depthE_tune
    say "$NG: stage 1 arms ($(echo $arms | wc -w)): $arms"
    st gen_E "${DX[@]}" $PYM scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --arms $arms \
      || { fail "stage-1 generation"; return 1; }
    st pos_E $DS checkpos $KEY E || { fail "stage-1 generation files (positions / rows)"; return 1; }
    if [ ! -f $wk/judge2/cells.jsonl ]; then
      st export_E "${DX[@]}" $PYM scripts/review_checks/e44_positions.py prompt export "${COMMON[@]}" --gen $gens --name depthE_tune \
        || { fail "stage-1 export"; return 1; }
      mkdir -p $wk/judge2/cache && cp $RUN/eval_extra/judge_tune/depthE_tune.json $wk/judge_input.json || { fail "stage-1 work dir"; return 1; }
      st cells_E "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W/$wk --generation judge_input.json \
        || { fail "stage-1 cells"; return 1; }
    fi
    touch $L/DEPTH1_${NG}_GEN_DONE
  fi
  # stage 2: C after E
  local cl k
  for cl in $NEWC; do
    arms=""; for k in $KS; do arms="$arms x:E$EL.w.p$ES+C$cl.w.m$k"; done
    st selfcheck_C$cl "${DX[@]}" $PYM scripts/review_checks/e44_positions2.py prompt selfcheck "${COMMON[@]}" \
      --arms x:E$EL.w.p$ES+C$cl.w.m1 || { fail "selfcheck of the pair E$EL / C$cl at prompt"; return 1; }
    st gen_C$cl "${DX[@]}" $PYM scripts/review_checks/e44_positions2.py prompt generate "${COMMON[@]}" --arms $arms \
      || { fail "stage-2 generation, C layer $cl"; return 1; }
  done
  st pos_C $DS checkpos $KEY C || { fail "stage-2 generation files (positions / rows)"; return 1; }
  wk=$RUN/eval_extra/judge_tune_depthC_tune
  if [ ! -f $wk/judge2/cells.jsonl ]; then
    gens=""; for a in $($DS c-arms $KEY); do gens="$gens $(fname $a)"; done
    st export_C "${DX[@]}" $PYM scripts/review_checks/e44_positions2.py prompt export "${COMMON[@]}" --gen $gens --name depthC_tune \
      || { fail "stage-2 export"; return 1; }
    mkdir -p $wk/judge2/cache && cp $RUN/eval_extra/judge_tune/depthC_tune.json $wk/judge_input.json || { fail "stage-2 work dir"; return 1; }
    st cells_C "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W/$wk --generation judge_input.json \
      || { fail "stage-2 cells"; return 1; }
  fi
  touch $L/DEPTH2_${NG}_GEN_DONE
  say "$NG: card part done"
}

budget() {  # seconds this lane may run now (0 = peak hour); layer_probe.sh
  local dow=$(date +%u) hm=$((10#$(date +%H%M))) now=$(date +%s) t
  if [ $dow -ge 6 ]; then echo 7200; return; fi
  if [ $hm -lt 900 ]; then t=$(( $(date -d "09:00" +%s) - now ))
  elif [ $hm -lt 1200 ]; then t=0
  elif [ $hm -lt 1400 ]; then t=$(( $(date -d "14:00" +%s) - now ))
  elif [ $hm -lt 1800 ]; then t=0
  else t=7200; fi
  [ $t -gt 7200 ] && t=7200
  [ $t -lt 180 ] && t=0
  echo $t
}
bal() {  # the total balance (balance.sh prints only the balance JSON)
  bash $L/balance.sh 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin)[0]["total_balance"])
except Exception:
    print("")'
}
nverdict() {  # nverdict WORK -> cells without a Pro verdict (empty answers and declined ones are verdicts)
  "${JX[@]}" python3 -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
ok = [r for r in rows if r['status'] == 'scored' or r.get('basis') in ('empty_output_no_judge_vote', 'pro_declined')]
print(len(rows) - len(ok))" "$W/$1/judge2/single_pro.jsonl"
}
single_pro() { "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W/$1 --judge pro > $1/single_pro_stdout.md; }
to_file() { local out=$1; shift; "$@" > $out 2>&1; }
judge() {  # judge KEY NAME SEEDNAME: DeepSeek Pro on <run>/eval_extra/judge_tune_<NAME>, off-peak, within the cap
  setm $1
  local wk=$RUN/eval_extra/judge_${SPLIT}_$2 seed=$RUN/eval_extra/judge_${SPLIT}_$3/judge2/cache cost=$L/depth_cost_${NG}_$2.txt
  local secs b sp up rc tries=0 left
  local MLOG=$JLOG
  [ -f $wk/judge2/cells.jsonl ] || { fail "no cells in $wk"; return 1; }
  [ -f $seed/judge2_cache_deepseek-v4-pro.jsonl ] || { fail "no Pro reply cache in $seed"; return 1; }
  st merge_$2 "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $W/$wk/judge2/cache $W/$seed \
    || { fail "cache merge into $wk"; return 1; }
  st conflicts_$2 "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $W/$wk/judge2/cache/conflicts.json \
    || { fail "cache conflicts in $wk"; return 1; }
  while true; do
    secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
    [ -e $L/DEPTH_COST_STOP ] && return 9
    "${JX[@]}" python3 -B $W/Iteration_01/diag/crow_pro_requests.py $W/$wk > $cost 2>> $JLOG || { fail "request count of $wk"; return 1; }
    up=$(sed -n 's/^expected cost .*, upper \([0-9.]*\) CNY.*/\1/p' $cost)
    [[ $up =~ ^[0-9]+\.[0-9]+$ ]] || { fail "cannot read the cost estimate in $cost"; return 1; }
    b=$(bal); [[ $b =~ ^[0-9]+(\.[0-9]+)?$ ]] || { fail "cannot read the balance"; return 1; }
    sp=0; [ -s $L/depth_cost_spent.txt ] && sp=$(cat $L/depth_cost_spent.txt)
    [[ $sp =~ ^[0-9]+(\.[0-9]+)?$ ]] || { fail "cannot read $L/depth_cost_spent.txt"; return 1; }
    if python3 -c 'import sys; sp, b, up, cap, floor = map(float, sys.argv[1:]); sys.exit(0 if (sp + up > cap or b - up < floor) else 1)' $sp $b $up $CAP $FLOOR; then
      say "COST STOP before $NG $2: earlier batches at most $sp yuan, this batch at most $up, cap $CAP; balance $b, floor $FLOOR"
      touch $L/DEPTH_COST_STOP
      return 9
    fi
    python3 -c 'import sys; print("%.4f" % (float(sys.argv[1]) + float(sys.argv[2])))' $sp $up > $L/depth_cost_spent.txt
    say "$NG $2: Pro batch (at most $secs s); balance $b; earlier batches at most $sp yuan; $(sed -n 3p $cost)"
    st pro_$2 "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $W/$wk --judge pro
    rc=$?
    say "$NG $2: balance after the batch $(bal)"
    case $rc in
      0) st single_$2 single_pro $wk || { fail "single (pro) of $wk"; return 1; }
         left=$(nverdict $wk)
         [[ $left =~ ^[0-9]+$ ]] || { fail "cannot read the verdicts of $wk"; return 1; }
         [ "$left" = 0 ] && return 0
         tries=$((tries + 1))
         [ $tries -ge 2 ] && { fail "$left cells of $wk without a Pro verdict after $tries complete passes"; return 1; }
         say "$NG $2: $left cells without a verdict; one more pass" ;;
      124) say "$NG $2: pass cut before a peak hour; resumes off-peak" ;;
      *) fail "the Pro pass on $wk (rc $rc)"; return 1 ;;
    esac
  done
}
judge_model() {  # judge_model KEY: both work dirs, then the two tables
  setm $1
  [ -e $L/DEPTH_${NG}_FAILED ] && { say "$NG: failed earlier, not judged"; return 1; }
  [ -e $L/DEPTH2_${NG}_GEN_DONE ] || { say "$NG: card part not done, not judged"; return 1; }
  if [ ! -e $L/DEPTH_${NG}_PRO_DONE ]; then
    judge $1 depthE_tune posprompt${TAG}_tune || return $?
    judge $1 depthC_tune posprompt2${TAG}_tune || return $?
    touch $L/DEPTH_${NG}_PRO_DONE
  fi
  setm $1
  local d=$(date +%Y%m%d)
  st table_E to_file Iteration_01/diag/out_depth_scan_${NG}_$d.txt $DS table $KEY || { fail "stage-1 table (see Iteration_01/diag/out_depth_scan_${NG}_$d.txt)"; return 1; }
  st table_C to_file Iteration_01/diag/out_depth_c_${NG}_$d.txt $DS c $KEY || { fail "stage-2 table (see Iteration_01/diag/out_depth_c_${NG}_$d.txt)"; return 1; }
  touch $L/DEPTH_${NG}_TABLE_DONE
  say "$NG: tables Iteration_01/diag/out_depth_scan_${NG}_$d.txt, out_depth_c_${NG}_$d.txt"
}

setm q9
say "$NG test judging: start (pid $$)"
until [ -e $L/DEPTH_LATE_${NG}_TESTGEN_DONE ] || [ -e $L/DEPTH_LATE_${NG}_FAILED ]; do sleep 20; done
[ -e $L/DEPTH_LATE_${NG}_TESTGEN_DONE ] || { fail "the test generation failed"; exit 1; }
rm -f $L/DEPTH_TESTJUDGE_${NG}_FAILED
judge q9 depthE_test pospromptl15_test; rc=$?
say "$NG test judging ended (rc $rc)"
[ $rc = 0 ] && touch $L/DEPTH_TESTJUDGE_${NG}_DONE
exit $rc

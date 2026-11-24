#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor ren, 2026-11-24 13 h; plan discussion/work_queue/2026-11-24_queue_n.md
# sections 2, 3 and 5): stages 1 and 2 of plan N, unattended, for Llama-3.1-8B then Qwen3.5-9B.
#   wait  for TFG3_gemma3-12b_DONE (the true/false run on card 1), then flock -n on Iteration_01/newsets/selfjudge/.gpu1.lock
#         every 10 s (held for the whole GPU part; a queue that takes it first is simply waited for); ours plus others at
#         most 66560 MiB on card 1 (llama asks 20000, q9 22000, q9 while training 30000).
#   per model, card 1 (markers make every part resumable; a step that fails writes DEPTH_<model>_FAILED and stops
#   that model only):
#     q9 only, a: E directions of layers 19-24 in the E2 run, trained with the arguments of that run's original E
#        training (code_ff_explore_20261113/scripts/review_checks/newmodel_e44.sh STAGE=run line 178, called by
#        ff_explore_20261113_chain.sh with CODE_PATH=code_ff_explore_20261113 VARIANT=E2 E_ON=E2 BRANCH=E):
#        e44_factorial_das.py train --source-run <source> --batch-size 8 --max-new-tokens 48 --max-target-tokens 24
#        --e-on E2 --run-dir <E2 run> --family E --layers ... --val-n 16, same container environment and interpreter;
#        acceptance: identity, identity_sha256, protocol, e_on, frozen_source and training_contract (apart from its
#        layer) equal to those of layer 15; family E, pilot false; direction file present with the logged sha256.
#     b: entry checks: MMLU 1,000 items, arm x, equals the stored reference (llama 640, q9 776 correct, files of
#        capability/sweep); the main-table arm (llama x:E16.w.p3.5, q9 x:E15.w.p2) regenerated at the prompt rule on
#        the first 16 tune questions (--first-batches 2 -> gen/t16.<file>) equals the archived generation, 16 of 16.
#     c: MMLU sweep, one cap_mcq_sweep.py call per layer (12 strengths; llama 8-27, q9 10-24) into
#        Iteration_01/capability/depthscan/<model>_L<l>.json, the rule of plan 2.1 by depth_scan.py choose, 12 and 16
#        for a layer whose 8 is still inside the 1.0 line (<model>_L<l>_ext.json), the rule again.
#     d: single E at s1 and s05 per layer (one arm when equal), tune, plain prompt, core rule "prompt"
#        (scripts/review_checks/e44_positions.py prompt; positions checked in every file), export depthE_tune, cells.
#     stage 2: E at the main-table layer and strength; per new C layer (llama 17 18 20 21, q9 17 18 19) the pair
#        selfcheck at "prompt" (e44_positions2.py prompt selfcheck x:E.w.p<s>+C<l>.w.m1, whole split) and the four
#        N1 re-ask arms (C at 0.25 / 0.5 / 1 / 2); export depthC_tune, cells.
#   then the lock is released and the four work dirs are judged by DeepSeek Pro (non-thinking judge config of the
#   code copy), off-peak only (budget() of layer_probe.sh: no pass in weekday 09-12 and 14-18 Beijing; a pass is cut
#   by timeout before a peak and resumed), one pass at a time. Each work dir's reply cache is seeded with ONE existing
#   cache of the same run (depthE <- judge_tune_posprompt<tag>_tune, depthC <- judge_tune_posprompt2<tag>_tune;
#   merging all prompt-rule caches of Llama gives 5 keys with different replies). Spending cap 15 yuan for the plan:
#   before every batch the balance is read (balance.sh) and compared with the balance before the first batch
#   (depth_cost_start.txt); spent > 15, or spent + the batch's upper cost estimate (crow_pro_requests.py) > 15, writes
#   DEPTH_COST_STOP and stops all judging. The balance difference also counts other queues' judging on the same key.
#   Tables: Iteration_01/diag/out_depth_scan_<model>_<yyyymmdd>.txt and out_depth_c_<model>_<yyyymmdd>.txt.
# Markers in Iteration_01/logs/newsets_20261119: DEPTH1_<model>_MMLU_DONE, DEPTH1_<model>_GEN_DONE,
# DEPTH2_<model>_GEN_DONE, DEPTH_<model>_PRO_DONE, DEPTH_<model>_TABLE_DONE, DEPTH_QUEUE_DONE; DEPTH_<model>_FAILED,
# DEPTH_COST_STOP. One line per finished step with its exit code in depth_all.log; command output in
# depth_<model>.log and depth_judge_<model>.log.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/depth_queue_1124.sh \
#                 > Iteration_01/logs/newsets_20261119/depth_queue.out 2>&1 < /dev/null &
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ALL=$L/depth_all.log
DS="python3 -B Iteration_01/diag/depth_scan.py"
CAPD=Iteration_01/capability/depthscan
LOCK=Iteration_01/newsets/selfjudge/.gpu1.lock
CARD=1
MAXMIB=66560
CAP=15
KS="0.25 0.5 1 2"
LADDER="0.5 0.75 1 1.5 2 2.5 3 3.5 4 5 6 8"
say() { echo "$(date '+%F %T') depth: $*" >> $ALL; }
mkdir $L/.depth_queue_running 2>/dev/null || { echo "queue already running ($L/.depth_queue_running)"; exit 3; }
trap 'rmdir $L/.depth_queue_running 2>/dev/null' EXIT
for f in Iteration_01/diag/depth_scan.py Iteration_01/diag/crow_pro_requests.py $L/balance.sh; do
  [ -s $f ] || { say "missing $f"; exit 2; }
done

setm() {
  KEY=$1; TRAINL=""; NEEDT=""
  case $1 in
    llama) NG=llama3.1-8b; CODE=$W/code_e45_20261112; PYM=python3; RUN=results_e44_factorial_das/llama3.1-8b_side_E2
           CRUN=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; SRC=$W/results_qa_das_v1u/llama3.1-8b/full
           EL=16; ES=3.5; TAG=l16; NEED=20000; NEWC="17 18 20 21"; LAYERS=$(seq 8 27) ;;
    q9)    NG=qwen3.5-9b; CODE=$W/code_ff_explore_20261113b; PYM=$W/envs/tf519/bin/python; RUN=results_newmodels_ff_20261112/e44/qwen3.5-9b_E2
           CRUN=results_newmodels_ff_20261112/e44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; SRC=$W/results_newmodels_ff_20261112/source/qwen3.5-9b
           EL=15; ES=2; TAG=l15; NEED=22000; NEEDT=30000; NEWC="17 18 19"; LAYERS=$(seq 10 24)
           TRAINL="19 20 21 22 23 24"; TCODE=$W/code_ff_explore_20261113; TREF=15 ;;
    *) say "unknown key $1"; exit 2 ;;
  esac
  MLOG=$L/depth_$NG.log; JLOG=$L/depth_judge_$NG.log
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
fail() { say "$NG FAILED: $* (see $MLOG)"; echo "$(date '+%F %T') FAILED: $*" >> $MLOG; touch $L/DEPTH_${NG}_FAILED; }
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
      st mmlu_L$l mmlu ${NG}_L$l $arms || { fail "MMLU sweep of layer $l"; return 1; }
    done
    st choose $DS choose $KEY || { fail "choice of the strengths"; return 1; }
    for l in $($DS need-ext $KEY); do
      st mmlu_L${l}_ext mmlu ${NG}_L${l}_ext x:E$l.w.p12 x:E$l.w.p16 || { fail "MMLU strengths 12 / 16 of layer $l"; return 1; }
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
  local wk=$RUN/eval_extra/judge_tune_$2 seed=$RUN/eval_extra/judge_tune_$3/judge2/cache cost=$L/depth_cost_${NG}_$2.txt
  local secs b b0 up rc tries=0 left
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
    [ -s $L/depth_cost_start.txt ] || echo "$b" > $L/depth_cost_start.txt
    b0=$(cat $L/depth_cost_start.txt)
    if python3 -c 'import sys; b0, b, up, cap = map(float, sys.argv[1:]); sys.exit(0 if (b0 - b > cap or b0 - b + up > cap) else 1)' $b0 $b $up $CAP; then
      say "COST STOP before $NG $2: balance at the first batch $b0, now $b, this batch at most $up, cap $CAP yuan (the balance also moves with other queues)"
      touch $L/DEPTH_COST_STOP
      return 9
    fi
    say "$NG $2: Pro batch (at most $secs s); balance $b (first batch $b0); $(sed -n 3p $cost)"
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

say "start (pid $$)"
if [ ! -e $L/DEPTH2_llama3.1-8b_GEN_DONE ] || [ ! -e $L/DEPTH2_qwen3.5-9b_GEN_DONE ]; then
  seen=0
  say "waiting for $L/TFG3_gemma3-12b_DONE"
  until [ -e $L/TFG3_gemma3-12b_DONE ]; do
    [ $seen = 0 ] && [ -e $L/TFG3_gemma3-12b_FAILED ] && { say "TFG3_gemma3-12b_FAILED exists; plan 5.1 waits for _DONE, still waiting"; seen=1; }
    sleep 60
  done
  say "seen TFG3_gemma3-12b_DONE; trying the lock of card $CARD every 10 s"
  exec 9>"$LOCK" || { say "cannot open $LOCK"; exit 1; }
  until flock -n 9; do sleep 10; done
  say "holds the lock of card $CARD"
  gpu_model llama
  gpu_model q9
  flock -u 9; exec 9>&-
  say "card $CARD released"
fi
judge_model llama; say "llama judging and tables ended (rc $?)"
judge_model q9; say "q9 judging and tables ended (rc $?)"
touch $L/DEPTH_QUEUE_DONE
say "queue done"

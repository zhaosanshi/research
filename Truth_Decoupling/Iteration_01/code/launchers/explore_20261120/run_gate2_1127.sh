#!/bin/bash
# Iteration_01 exploration (executor you, Claude Opus 5.5, 2026-11-27): queue P section 5, segment A
# (discussion/work_queue/2026-11-26_queue_p.md; readings defined by 2026-11-26_gpt_gate_review.md). FalseQA held-out 600
# behind the same frozen switch: N1 at the larger E strength s2 against N1 at the original strength s. Not reviewed by GPT.
# Per model (order of the plan: qwen3.5-9b, qwen3.8-27b, gemma3-12b, gemma3-27b, llama3.1-8b):
#   1 generation (card lock): the entry check (two existing FalseQA arms regenerated on the first 16 questions through
#     the same entry must equal the archived files: ns_h22.py compare), then the six new arms on the 700 bank questions
#     through scripts/newsets/ns_h22.py falseqa prompt gen (NS_POS_TWO_WRITE=1, NS_POS_PLAIN_NAMES=1; the program and
#     settings of falseqa_model.sh) into newsets/gen_prompt/falseqa_gate2_1127/<model>/test:
#       x:E<el>.w.p<s2> | x:E<el>.w.p<s2>+C<cl>.w.m<c> (c = 0.25 0.5 1 2) | conf_E2:E<el>.w.p<s2>
#     write records (scripts/newsets/n1s_check.py parts): every E part against the existing FalseQA file of the same base
#     at s (conf_E2: at s_sel) with the amount scaled by the strength ratio, every C part against the C part of the
#     existing x:E<el>.w.p<s>+C<cl>.w.m<c>; diag/falseqa_gate2_1127.py check-gen;
#   2 export / cells (CPU): the unchanged ns_h22 export of the six arms (newsets/judge/falseqa/<model>/export_gate2_1127.json),
#     diag/falseqa_gate2_1127.py judge-input (ON held-out questions only; cells that cannot be handed in get an empty
#     output; rows checked against the export), judge_two_axis_run.py cells -> work dir
#     newsets/judge/falseqa/<model>/test_gate2_1127_heldout;
#   3 self-judge margins (card lock): falseqa_gate2_1127.py export-margins, the E45 regression where falseqa_score.sh runs
#     it, diag/falseqa_score_gate2_1127.py score (batch 1, no write), falseqa_gate2_1127.py check-margins;
#   4 judging (judge lane, one model after the other in the plan order): Pro reply cache seeded with the model's
#     test_full700_heldout cache (same text -> same verdict), conflicts must be none, off-peak only (weekdays 08:30-12:00
#     and 14:00-18:00 Beijing are peak; a pass is cut 2 minutes before a peak), expected cost printed before each batch;
#     no batch that could take this segment over CAP yuan or the balance under FLOOR yuan; llm then single (Pro);
#   5 table refresh: diag/table_gate2_1127.py <finished models> > diag/out_gate2_1127_falseqa_20261127.txt.
# GPU lanes (cards through Iteration_01/newsets/selfjudge/.gpu<N>.lock, flock retried every 60 s; ours plus others at
# most 66560 MiB per card at the start, waits otherwise): lane 1 card 1: qwen3.5-9b, gemma3-12b, llama3.1-8b; lane 2:
# qwen3.8-27b generation on cards 2,3 (scoring on card 2, as falseqa_score.sh: one card), then gemma3-27b on card 2.
# Markers in Iteration_01/logs/newsets_20261119: GATE2_GEN_<model>_DONE, GATE2_CELLS_<model>_DONE, GATE2_SCORE_<model>_DONE,
# GATE2_JUDGE_<model>_DONE, GATE2_TABLE_<model>_DONE, GATE2_A_DONE / GATE2_FAILED / GATE2_COST_STOP; smoke GATE2_SMOKE_DONE.
# Status lines "gate2:" in crow_all.log; step logs gate2_1127_<model>.log, gate2_1127_judge_<model>.log.
# usage (host):  run_gate2_1127.sh dry MODEL     the generation and scoring commands without a card (expected stop
#                                                "No CUDA GPUs are available"; in-container timeout 150 s)
#                run_gate2_1127.sh entry MODEL   the entry check alone on the model's generation card(s)
#                run_gate2_1127.sh smoke         qwen3.5-9b on the 3 smoke questions, the whole chain (card 1)
#                run_gate2_1127.sh full          the whole chain, five models
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling; ROOT=/workspace/Truth_Decoupling
cd $HOST || exit 2
export TZ=Asia/Shanghai
L=Iteration_01/logs/newsets_20261119
ALL=$L/crow_all.log
TF=$ROOT/envs/tf519/bin/python
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
BANK=$NS/falseqa/falseqa_full_bank_20261123.jsonl
CODE=$ROOT/Iteration_01/code
NH=$CODE/scripts/newsets/ns_h22.py
NC=$CODE/scripts/newsets/n1s_check.py
MCFG=$CODE/configs/newsets/frozen_models_20261119.json
HELP=Iteration_01/diag/falseqa_gate2_1127.py
SCORE=Iteration_01/diag/falseqa_score_gate2_1127.py
TABLE=Iteration_01/diag/table_gate2_1127.py
OUTT=Iteration_01/diag/out_gate2_1127_falseqa_20261127.txt
OUTS=Iteration_01/diag/out_gate2_1127_smoke_20261127.txt
MAXMIB=66560; CAP=6; FLOOR=10
SPENT=$L/gate2_1127_cost_spent.txt
ORDER="qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b llama3.1-8b"
MODE=${1:-}
say() { echo "$(date '+%F %T') gate2: $*" >> $ALL; }
failed() { say "FAILED: $*"; touch $L/GATE2_FAILED; }

setm() {  # setm MODEL: frozen setting, interpreters, memory, cards, scoring code copy (falseqa_model.sh / falseqa_score.sh)
  M=$1
  case $M in
    llama3.1-8b) EL=16; S=3.5;  CL=19; S2=5; PYM=python3; GNEED=20000; GCARD=1;   SNEED=17000; SCARD=1
                 SCODE=code_e45_20261112;     SCFG=code_e45_20261112/configs/models/$M.json; REG=1 ;;
    qwen3.5-9b)  EL=15; S=2;    CL=16; S2=3; PYM=$TF;     GNEED=22000; GCARD=1;   SNEED=20000; SCARD=1
                 SCODE=code_newmodels_20261112; SCFG=code_newmodels_20261112/configs/models/$M.json; REG=1 ;;
    qwen3.8-27b) EL=28; S=1.75; CL=33; S2=3; PYM=$TF;     GNEED=34000; GCARD=2,3; SNEED=60000; SCARD=2
                 SCODE=code_newmodels_20261112; SCFG=code_newmodels_20261112/configs/models/$M.json; REG=1 ;;
    gemma3-12b)  EL=24; S=2.5;  CL=25; S2=4; PYM=python3; GNEED=30000; GCARD=1;   SNEED=28000; SCARD=1
                 SCODE=code_newmodels_20261112; SCFG=code_newmodels_20261112/configs/models/$M.json; REG=0 ;;
    gemma3-27b)  EL=28; S=1;    CL=31; S2=2; PYM=python3; GNEED=62000; GCARD=2;   SNEED=60000; SCARD=2
                 SCODE=code_newmodels_20261112; SCFG=Iteration_01/newsets/selfjudge/_configs/$M.json; REG=0 ;;
    *) echo "unknown model $M" >&2; return 2 ;;
  esac
  SS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["strength_dot"])' $HNS/h22/n1s_choice_$M.json) || return 2
  E2A=x:E$EL.w.p$S2
  ARMS="$E2A"; local c
  for c in 0.25 0.5 1 2; do ARMS="$ARMS $E2A+C$CL.w.m$c"; done
  ARMS="$ARMS conf_E2:E$EL.w.p$S2"
  MLOG=$L/gate2_1127_$M.log; JLOG=$L/gate2_1127_judge_$M.log
  [ -n "${SMOKE:-}" ] && { MLOG=$L/gate2_1127_smoke_$M.log; JLOG=$L/gate2_1127_smoke_judge_$M.log; }
  return 0
}
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
D() { echo "$1" | tr . -; }
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
JK=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
dq() {  # dq DEVICES: the generation container command of falseqa_model.sh
  DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
      -e CUDA_VISIBLE_DEVICES=$1 -w $ROOT truth_dec)
}
dx() {  # dx DEVICES: the scoring container command of falseqa_score.sh
  DX=(docker exec -u 1000:1001 -e "HOME=$ROOT/hf_cache" -e "HF_HOME=$ROOT/hf_cache" -e "TD_DATA_DIR=$ROOT/code/data/processed"
      -e "TD_MODEL_ROOT=$ROOT/models" -e "TD_RESULTS_DIR=$ROOT/Iteration_01/newsets/selfjudge/_e45_unused" -e CUDA_VISIBLE_DEVICES=$1
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=16
      -e "PYTHONPATH=$ROOT/$SCODE" -w "$ROOT" truth_dec)
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

parts_new() {  # parts_new GENFILE ARM: write records against the existing FalseQA files (n1s_check.py parts, unchanged)
  local f=$1 a=$2 c OG=$NS/gen_prompt/falseqa/$M/test
  case $a in
    conf_E2:*) st parts "${DC[@]}" $NC parts $f $OG/conf_E2--E$EL-w-p$(D $SS).json $S2 $SS E ;;
    *+C*) c=${a##*.w.m}
          st parts "${DC[@]}" $NC parts $f $OG/x--E$EL-w-p$(D $S).json $S2 $S E &&
          st parts "${DC[@]}" $NC parts $f $OG/x--E$EL-w-p$(D $S)-and-C$CL-w-m$(D $c).json 1 1 C ;;
    *) st parts "${DC[@]}" $NC parts $f $OG/x--E$EL-w-p$(D $S).json $S2 $S E ;;
  esac
}

entry_check() {  # two existing FalseQA arms regenerated on the first 16 questions = the archived files (cards held by the caller)
  local EC=$NS/gen_prompt/h22chk/falseqa_gate2_1127/$M/test A1=x:E$EL.w.p$S A2=x:E$EL.w.p$S+C$CL.w.m0.25 a f
  st entrygen "${DQ[@]}" $PYM -I $NH falseqa prompt gen --models $MCFG --model $M --task falseqa --items $BANK --split test \
      --out $EC --arms $A1 $A2 --first-batches 2 || { say "$M: entry check generation FAILED"; return 1; }
  for a in $A1 $A2; do
    f=$(first $EC $(tok $a)); [ -n "$f" ] || { say "$M: entry check file of $a missing"; return 1; }
    st entrycompare "${DC[@]}" $NH compare $f $NS/gen_prompt/falseqa/$M/test/$(tok $a).json || {
      say "$M: entry check: $a differs from the archived FalseQA file"; return 1; }
  done
  say "$M: entry check passed (2 archived FalseQA arms reproduced on 16 questions, card(s) $GCARD)"
}

gpu_model() {  # gpu_model MODEL FD0: generation, cells, self-judge margins
  setm $1 || return 1
  local GD=$NS/gen_prompt/falseqa_gate2_1127/$M/test a f
  local W=$NS/judge/falseqa/$M/test_gate2_1127_heldout EXP=$NS/judge/falseqa/$M/export_gate2_1127.json
  [ -e $L/GATE2_FAILED ] && return 1
  if [ ! -e $L/GATE2_GEN_${M}_DONE ]; then
    lock $GCARD $2 $GNEED || return 1
    say "$M: generation on card(s) $GCARD; arms $ARMS"
    dq $GCARD
    entry_check || { unlock; return 1; }
    st generate "${DQ[@]}" $PYM -I $NH falseqa prompt gen --models $MCFG --model $M --task falseqa --items $BANK --split test \
        --out $GD --arms $ARMS || { unlock; say "$M: generation FAILED"; return 1; }
    unlock
    for a in $ARMS; do parts_new $GD/$(tok $a).json $a || { say "$M: write record of $a FAILED"; return 1; }; done
    st checkgen "${RD[@]}" $HELP check-gen $M full || { say "$M: check-gen FAILED"; return 1; }
    touch $L/GATE2_GEN_${M}_DONE; say "$M: generation done"
  fi
  if [ ! -e $L/GATE2_CELLS_${M}_DONE ]; then
    if [ ! -e $HOST/${EXP#$ROOT/} ]; then
      st export "${J[@]}" python3 -I $NH falseqa none export --task falseqa --model $M --items $BANK --split test \
          --gen-dir $GD --only-arms $ARMS --out $EXP || { say "$M: export FAILED"; return 1; }
    fi
    st judgeinput "${RD[@]}" $HELP judge-input $M full $EXP || { say "$M: judge input FAILED"; return 1; }
    if [ ! -e $HOST/${W#$ROOT/}/judge2/cells.jsonl ]; then
      st cells "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation judge_input_gate2_1127.json || {
        say "$M: cells FAILED"; return 1; }
    fi
    touch $L/GATE2_CELLS_${M}_DONE; say "$M: cells ready"
  fi
  score_model full $(( $2 + 4 )) || return 1
}

score_model() {  # score_model full|smoke FD: batch-1 self-judge margins of the new answers (falseqa_score.sh's path)
  local PH=$1 P SRCS PY RL
  [ $PH = full ] && P=fqa_gate2_1127 || P=fqa_gate2_1127s
  [ -e $L/GATE2_SCORE_${M}_${PH}_DONE ] && return 0
  if [ ! -s $HNS/selfjudge/$M/${P}_items_items.json ]; then
    st exportmargins "${RD[@]}" $HELP export-margins $M $PH || { say "$M: export-margins FAILED"; return 1; }
  fi
  if [ -s $HNS/selfjudge/$M/${P}_items_items.json ]; then
    [ -f "$HOST/$SCFG" ] || { say "$M: no model config $SCFG"; return 1; }
    lock $SCARD $2 $SNEED || return 1
    dx $SCARD
    PY=$(docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$ROOT/$SCFG") || {
      unlock; say "$M: interpreter"; return 1; }
    if [ "$REG" = 1 ]; then
      RL=$L/gate2_1127_${M}_${PH}_regression.log
      "${DX[@]}" "$PY" -B Iteration_01/diag/selfjudge_probe.py score "$M" --regress-only > $RL 2>&1; local rc=$?
      cat $RL >> $MLOG
      { [ $rc = 0 ] && grep -q "REGRESSION OK" $RL; } || { unlock; say "$M: E45 regression FAILED (see $RL)"; return 1; }
      say "$M: E45 regression passed"
    fi
    st score "${DX[@]}" "$PY" -B $SCORE score $M $PH || { unlock; say "$M: score FAILED"; return 1; }
    unlock
    st checkmargins "${RD[@]}" $HELP check-margins $M $PH || { say "$M: check-margins FAILED"; return 1; }
  fi
  touch $L/GATE2_SCORE_${M}_${PH}_DONE; say "$M: self-judge margins ($PH) done"
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
judge() {  # judge MODEL WORKDIR: DeepSeek Pro on the work dir, off-peak, within the cap and above the floor
  setm $1 || return 1
  local W=$2 HW=$HOST/${2#$ROOT/} MLOG=$JLOG cost=$L/gate2_1127_cost_$1.txt secs b sp up rc tries=0 left
  local SEED=$NS/judge/falseqa/$M/test_full700_heldout/judge2/cache
  [ -f $HW/judge2/cells.jsonl ] || { say "$M: no cells in $W"; return 1; }
  [ -f $HOST/${SEED#$ROOT/}/judge2_cache_deepseek-v4-pro.jsonl ] || { say "$M: no Pro reply cache in $SEED"; return 1; }
  st merge "${J[@]}" python3 scripts/review_checks/judge2_cache_merge.py $W/judge2/cache $SEED || { say "$M: cache merge FAILED"; return 1; }
  st conflicts "${J[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $W/judge2/cache/conflicts.json || { say "$M: cache conflicts"; return 1; }
  while true; do
    secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
    [ -e $L/GATE2_COST_STOP ] && return 9
    "${J[@]}" python3 -B $ROOT/Iteration_01/diag/crow_pro_requests.py $W > $cost 2>> $JLOG || { say "$M: request count FAILED"; return 1; }
    up=$(sed -n 's/^expected cost .*, upper \([0-9.]*\) CNY.*/\1/p' $cost)
    [[ $up =~ ^[0-9]+\.[0-9]+$ ]] || { say "$M: cannot read the cost estimate in $cost"; return 1; }
    b=$(bal); [[ $b =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "$M: cannot read the balance"; return 1; }
    sp=0; [ -s $SPENT ] && sp=$(cat $SPENT)
    [[ $sp =~ ^[0-9]+(\.[0-9]+)?$ ]] || { say "cannot read $SPENT"; return 1; }
    if python3 -c 'import sys; sp, b, up, cap, floor = map(float, sys.argv[1:]); sys.exit(0 if (sp + up > cap or b - up < floor) else 1)' $sp $b $up $CAP $FLOOR; then
      say "COST STOP before $M: earlier batches at most $sp yuan, this batch at most $up, cap $CAP; balance $b, floor $FLOOR"
      touch $L/GATE2_COST_STOP
      return 9
    fi
    python3 -c 'import sys; print("%.4f" % (float(sys.argv[1]) + float(sys.argv[2])))' $sp $up > $SPENT
    say "$M: Pro batch (at most $secs s); balance $b; earlier batches at most $sp yuan; $(sed -n 2,3p $cost | tr "\n" " ")"
    st pro "${JK[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $W --judge pro
    rc=$?
    say "$M: balance after the batch $(bal)"
    case $rc in
      0) st single "${JK[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W --judge pro || { say "$M: single (pro) FAILED"; return 1; }
         left=$("${J[@]}" python3 -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
ok = [r for r in rows if r['status'] == 'scored' or r.get('basis') in ('empty_output_no_judge_vote', 'pro_declined')]
print(len(rows) - len(ok))" "$W/judge2/single_pro.jsonl")
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
table() {  # table: the finished models (judged and scored)
  local m done=""
  for m in $ORDER; do
    [ -e $L/GATE2_JUDGE_${m}_DONE ] && [ -e $L/GATE2_SCORE_${m}_full_DONE ] && done="$done $m"
  done
  "${RD[@]}" $TABLE $done > $OUTT.tmp 2> $L/gate2_1127_table.err && mv $OUTT.tmp $OUTT
}
judge_lane() {
  local m rc
  for m in $ORDER; do
    until [ -e $L/GATE2_CELLS_${m}_DONE ]; do [ -e $L/GATE2_FAILED ] && return 1; sleep 30; done
    if [ ! -e $L/GATE2_JUDGE_${m}_DONE ]; then
      judge $m $NS/judge/falseqa/$m/test_gate2_1127_heldout; rc=$?
      [ $rc = 0 ] || { [ $rc = 9 ] || failed "judging of $m (rc $rc)"; return 1; }
      touch $L/GATE2_JUDGE_${m}_DONE; say "$m: judged"
    fi
    until [ -e $L/GATE2_SCORE_${m}_full_DONE ]; do [ -e $L/GATE2_FAILED ] && return 1; sleep 30; done
    M=$m
    if table; then
      touch $L/GATE2_TABLE_${m}_DONE; say "$m: table $OUTT $(sha256sum $OUTT | cut -c1-16)"
    else
      failed "table after $m (see $L/gate2_1127_table.err)"; return 1
    fi
  done
}

checkfiles() {
  local f
  for f in $HELP $SCORE $TABLE Iteration_01/diag/crow_pro_requests.py $L/balance.sh Iteration_01/gpt_tools_20261126/gate_audit/falseqa_matched.py \
           Iteration_01/code/scripts/newsets/ns_h22.py Iteration_01/code/scripts/newsets/n1s_check.py; do
    [ -s $f ] || { echo "missing $f" >&2; return 1; }
  done
  command -v flock > /dev/null || { echo "flock unavailable" >&2; return 1; }
}

case $MODE in
  dry)
    setm ${2:?model} || exit 2
    DL=$L/gate2_1127_dry_$M.log
    DD=$NS/gen_prompt/falseqa_gate2_1127_dry/$M/test
    dq ""
    echo "=== $(date '+%F %T') dry $M generate" >> $DL
    "${DQ[@]}" timeout -k 10 150 $PYM -I $NH falseqa prompt gen --models $MCFG --model $M --task falseqa --items $BANK --split test \
        --out $DD --arms $ARMS --first-batches 1 >> $DL 2>&1
    echo "dry $M generate exit=$? no-cuda-line=$(grep -c 'No CUDA GPUs are available' $DL) last: $(grep -v '^\s*$' $DL | tail -n 1 | cut -c1-200)"
    if [ -e $HOST/Iteration_01/newsets/gen_prompt/falseqa_gate2_1127_dry ]; then
      echo "dry run left $(find $HOST/Iteration_01/newsets/gen_prompt/falseqa_gate2_1127_dry -type f | wc -l) files; removing the dry dir"
      rm -rf $HOST/Iteration_01/newsets/gen_prompt/falseqa_gate2_1127_dry
    fi
    dx ""
    PY=$(docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$ROOT/$SCFG")
    if [ "$REG" = 1 ]; then
      echo "=== $(date '+%F %T') dry $M regression" >> $DL
      "${DX[@]}" timeout -k 10 150 "$PY" -B Iteration_01/diag/selfjudge_probe.py score "$M" --regress-only >> $DL 2>&1
      echo "dry $M regression exit=$? last: $(grep -v '^\s*$' $DL | tail -n 1 | cut -c1-200)"
    fi
    echo "=== $(date '+%F %T') dry $M score" >> $DL
    "${DX[@]}" timeout -k 10 150 "$PY" -B $SCORE score $M smoke >> $DL 2>&1
    echo "dry $M score exit=$? last: $(grep -v '^\s*$' $DL | tail -n 1 | cut -c1-200)"
    echo "no-cuda lines in $DL: $(grep -c 'No CUDA GPUs are available' $DL)"
    exit 0 ;;
  entry)
    checkfiles || exit 2
    setm ${2:?model} || exit 2
    lock $GCARD 8 $GNEED || exit 1
    dq $GCARD
    entry_check; rc=$?
    unlock
    echo "entry $M on card(s) $GCARD exit=$rc"
    exit $rc ;;
  smoke)
    checkfiles || exit 2
    [ -e $L/GATE2_SMOKE_DONE ] && { echo "smoke already done"; exit 0; }
    SMOKE=1; setm qwen3.5-9b || exit 2
    say "smoke start (pid $$)"
    GD=$NS/gen_prompt/falseqa_gate2_1127_smoke/$M/test
    W=$NS/judge/falseqa/$M/test_gate2_1127_smoke
    st smokeids "${RD[@]}" $HELP smoke-ids $M || { failed "smoke ids"; exit 1; }
    IDS=$NS/gate2_1127/smoke_idlist_gate2_1127.json
    lock $GCARD 8 $GNEED || { failed "smoke lock"; exit 1; }
    dq $GCARD
    st generate "${DQ[@]}" $PYM -I $NH falseqa prompt gen --models $MCFG --model $M --task falseqa --items $BANK --split test \
        --out $GD --arms $ARMS --only-ids $IDS || { unlock; failed "smoke generation"; exit 1; }
    unlock
    for a in $ARMS; do f=$(first $GD $(tok $a)); [ -n "$f" ] || { failed "smoke file of $a"; exit 1; }
      parts_new $f $a || { failed "smoke write record of $a"; exit 1; }; done
    st checkgen "${RD[@]}" $HELP check-gen $M smoke || { failed "smoke check-gen"; exit 1; }
    st judgeinput "${RD[@]}" $HELP judge-input $M smoke || { failed "smoke judge input"; exit 1; }
    if [ ! -e $HOST/${W#$ROOT/}/judge2/cells.jsonl ]; then
      st cells "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation judge_input_gate2_1127.json || {
        failed "smoke cells"; exit 1; }
    fi
    judge $M $W || { failed "smoke judging"; exit 1; }
    score_model smoke 12 || { failed "smoke scoring"; exit 1; }
    "${RD[@]}" $TABLE smoke $M > $OUTS 2>> $MLOG || { failed "smoke table"; exit 1; }
    touch $L/GATE2_SMOKE_DONE; say "smoke done: $OUTS"
    exit 0 ;;
  full)
    checkfiles || exit 2
    mkdir $L/.gate2_1127_running 2>/dev/null || { echo "already running ($L/.gate2_1127_running)"; exit 3; }
    trap 'rmdir $L/.gate2_1127_running 2>/dev/null' EXIT
    [ -e $L/GATE2_A_DONE ] && { say "already done (GATE2_A_DONE)"; exit 0; }
    [ -e $L/GATE2_SMOKE_DONE ] || { echo "run the smoke first" >&2; exit 2; }
    rm -f $L/GATE2_FAILED
    say "full start (pid $$): helper $(sha256sum $HELP | cut -c1-16), score $(sha256sum $SCORE | cut -c1-16), table $(sha256sum $TABLE | cut -c1-16)"
    ( for m in qwen3.5-9b gemma3-12b llama3.1-8b; do gpu_model $m 20 || { failed "gpu lane 1 at $m"; exit 1; }; done ) &
    P1=$!
    ( for m in qwen3.8-27b gemma3-27b; do gpu_model $m 40 || { failed "gpu lane 2 at $m"; exit 1; }; done ) &
    P2=$!
    judge_lane & P3=$!
    BAD=""
    wait $P1 || BAD="$BAD lane1"
    wait $P2 || BAD="$BAD lane2"
    wait $P3 || BAD="$BAD judge-lane"
    if [ -n "$BAD" ]; then
      [ -e $L/GATE2_COST_STOP ] && say "stopped by the cost guard:$BAD" || failed "lanes:$BAD"
      exit 1
    fi
    touch $L/GATE2_A_DONE; say "segment A done: $OUTT $(sha256sum $OUTT | cut -c1-16)"
    exit 0 ;;
  *) echo "usage: run_gate2_1127.sh dry MODEL | entry MODEL | smoke | full" >&2; exit 2 ;;
esac

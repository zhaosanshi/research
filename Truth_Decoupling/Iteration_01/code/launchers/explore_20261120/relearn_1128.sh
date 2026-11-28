#!/bin/bash
# Iteration_01 exploration (executor R-yi, Claude Opus 5.5, 2026-11-28): work queue R, R7 / R8
# (discussion/work_queue/2026-11-28_queue_r.md sections 2, 3.2, 5): learn E and C anew on PopQA and on FalseQA for
# qwen3.5-9b and gemma3-27b, every layer, and stop after training. Not reviewed by GPT.
# Program: scripts/e44_factorial_das.py (unchanged), called through scripts/newsets/relearn_1128_e44.py (adds the upper
# truth level FR; with E1 / E2 the unchanged program). Settings = the TruthfulQA runs of the main table (defaults: jsd,
# content, random init, 10 epochs, lr 1e-3, clip 1, seed 42, split seed 42; --val-n 16, batch 8, 48 / 24 tokens).
# Questions: source dirs Iteration_01/results/relearn_1128/<set>/<model>/source (scripts/newsets/relearn_1128_source.py;
# bank.json = the model's TruthfulQA source bank.json byte for byte). Prompts: PopQA e44_factorial_relearn_1128_fr.json
# (the 11 prompts + conf_FR / caut_FR), FalseQA e44_factorial_v1.json. Upper truth level: PopQA qwen3.5-9b E2,
# gemma3-27b FR (the 11-19 rule applied per model; see the progress note); FalseQA E2. Both families use it.
# Run dirs: Iteration_01/results/relearn_1128/<set>/<model>/run_<E_ON>. Only freeze (train + tune) and train run;
# no generation, readout or judging (smoke mode: 3-question pilot including generation on 3 tune questions).
# Cards: through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock -n, any of cards 0-3, retried every 60 s); at the
# start ours plus others at most 66560 MiB on the card (gemma3-27b training asks 66560: an empty card, as
# crow2_gen_from_1125.sh). gemma3-27b runs with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (its TruthfulQA
# training runner newmodel_e44_expseg.sh). The program is never split across cards: gemma3-27b layers are trained from
# the deepest down; the first layer that runs out of memory ends that job (marker RELEARN_<job>_OOM with the lowest
# layer done), every other failure stops the chain (RELEARN_FAILED).
# usage (host):  setsid -f bash Iteration_01/logs/newsets_20261119/relearn_1128.sh smoke > ..._smoke.out 2>&1 < /dev/null
#                setsid -f bash Iteration_01/logs/newsets_20261119/relearn_1128.sh full  > ..._full.out 2>&1 < /dev/null
#                (full needs RELEARN_SMOKE_DONE; WORKERS=<n> parallel workers, default 4)
# Status lines in Iteration_01/logs/newsets_20261119/relearn_1128_all.log; one log per job relearn_1128_<job>.log.
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling; ROOT=/workspace/Truth_Decoupling
cd $HOST || exit 2
export TZ=Asia/Shanghai
L=Iteration_01/logs/newsets_20261119
ALL=$L/relearn_1128_all.log
HNS=$HOST/Iteration_01/newsets
CODE=$ROOT/Iteration_01/code
ENTRY=scripts/newsets/relearn_1128_e44.py
RR=Iteration_01/results/relearn_1128
TF=$ROOT/envs/tf519/bin/python
P11=$CODE/configs/prompts/e44_factorial_v1.json
P13=$CODE/configs/prompts/e44_factorial_relearn_1128_fr.json
MAXMIB=66560
QLOCK=$L/.relearn_1128_queue.lock
MODE=${1:-}
say() { echo "$(date '+%F %T') relearn: $*" >> $ALL; }

setm() {  # setm MODEL SET: interpreter, memory asked (freeze / train), layers, prompts, upper truth level, dirs
  M=$1; S=$2
  case $M in
    qwen3.5-9b) PYM=$TF;     NEEDF=22000; NEEDT=${Q9_NEEDT:-44000}; NL=32; XENV=() ;;
    gemma3-27b) PYM=python3; NEEDF=62000; NEEDT=66560;              NL=62; XENV=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) ;;
    *) echo "unknown model $M" >&2; return 2 ;;
  esac
  case $M:$S in
    qwen3.5-9b:popqa) PR=$P13; EON=E2 ;;
    gemma3-27b:popqa) PR=$P13; EON=FR ;;
    *:falseqa)        PR=$P11; EON=E2 ;;
    *) echo "unknown set $S" >&2; return 2 ;;
  esac
  SRC=$ROOT/$RR/$S/$M/source
  RUN=$ROOT/$RR/$S/$M/run_$EON
  HRUN=$HOST/$RR/$S/$M/run_$EON
  [ -f $HOST/$RR/$S/$M/source/bank.json ] && [ -f $HOST/$RR/$S/$M/source/questions.json ] || { echo "no source $SRC" >&2; return 2; }
  return 0
}
dx() {  # dx CARD ARGS...: the model interpreter on the entry, in the container
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES="$1" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE "${XENV[@]}" -w $CODE truth_dec \
    "$PYM" $ENTRY "${@:2}"
}
COMMON=(--batch-size 8 --max-new-tokens 48 --max-target-tokens 24)

CARD=""; CFD=""
getcard() {  # getcard MIB: the first card whose lock is free and that has room; holds its lock (fd CFD) until putcard
  local g fd used
  while true; do
    [ -e $L/RELEARN_FAILED ] && return 1
    for g in 0 1 2 3; do
      exec {fd}>"$HNS/selfjudge/.gpu$g.lock" || return 1
      if flock -n $fd; then
        used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1 == g {print $2}')
        if [[ $used =~ ^[0-9]+$ ]] && [ $((used + $1)) -le $MAXMIB ]; then
          CARD=$g; CFD=$fd; say "$JOB: card $g (holds $used MiB, asking $1)"; return 0
        fi
      fi
      exec {fd}>&-
    done
    sleep 60
  done
}
putcard() { [ -n "$CFD" ] && exec {CFD}>&-; say "$JOB: released card $CARD"; CARD=""; CFD=""; }
POLL=""
poll_start() {  # highest memory in use on the card every 5 s, to $1
  ( peak=0; while true; do
      u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$CARD '$1 == g {print $2}')
      [[ $u =~ ^[0-9]+$ ]] && [ $u -gt $peak ] && { peak=$u; echo $peak > "$1"; }
      sleep 5; done ) &
  POLL=$!
}
poll_stop() { [ -n "$POLL" ] && kill $POLL 2>/dev/null; wait $POLL 2>/dev/null; POLL=""; }
layers_of() {  # layers_of SPEC NL: all | hi | lo, deepest first
  # 2026-11-29 (main session, queue R section 4): only the middle 50% of the layers are candidates. Layer l (counted
  # from 0) is kept when 0.3 * NL < l + 1 <= 0.8 * NL, i.e. 10 (l + 1) > 3 NL and 10 (l + 1) <= 8 NL.
  local l
  case $1 in
    all) seq $(($2 - 1)) -1 0 ;;
    hi)  seq $(($2 - 1)) -1 31 ;;
    lo)  seq 30 -1 0 ;;
  esac | while read -r l; do
    [ $((10 * (l + 1))) -gt $((3 * $2)) ] && [ $((10 * (l + 1))) -le $((8 * $2)) ] && echo $l
  done | tr '\n' ' '
}
done_layers() {  # done_layers FAMILY LAYERS: the layers with a direction and a training log
  local l out=""
  for l in $2; do [ -f $HRUN/dirs/${1}_L$l.pt ] && [ -f $HRUN/train_logs/${1}_L$l.json ] && out="$out $l"; done
  echo $out
}

run_job() {  # run_job ID MODEL SET KIND FAMILY SPEC
  JOB=$1; local kind=$4 fam=$5 spec=$6 rc=0 need=0 lay="" got="" missing="" low="" l log=$L/relearn_1128_$1.log
  [ "$fam" = - ] && fam=""
  setm $2 $3 || { touch $L/RELEARN_${JOB}_FAILED $L/RELEARN_FAILED; say "$JOB FAILED (settings)"; return 1; }
  if [ $kind = freeze ]; then need=$NEEDF; else need=$NEEDT; lay=$(layers_of $spec $NL); fi
  getcard $need || return 1
  mkdir -p $HRUN
  poll_start $L/relearn_1128_$JOB.mem
  echo "=== $(date '+%F %T') $JOB on card $CARD: $kind $M $S e_on=$EON${fam:+ family $fam layers $lay}" >> $log
  if [ $kind = freeze ]; then
    dx $CARD freeze --source-run $SRC --run-dir $RUN --prompts-e44 $PR "${COMMON[@]}" --splits train tune >> $log 2>&1; rc=$?
    [ $rc = 0 ] && { [ -f $HRUN/frozen_train.json ] && [ -f $HRUN/frozen_tune.json ] || rc=97; }
  else
    dx $CARD train --source-run $SRC --run-dir $RUN --prompts-e44 $PR "${COMMON[@]}" --e-on $EON --family $fam \
      --layers $lay --val-n 16 >> $log 2>&1; rc=$?
  fi
  poll_stop; putcard
  say "$JOB exit=$rc peak $(cat $L/relearn_1128_$JOB.mem 2>/dev/null) MiB"
  if [ $kind = train ]; then
    got=$(done_layers $fam "$lay")
    for l in $lay; do [[ " $got " == *" $l "* ]] || missing="$missing$l "; done
    if [ $rc = 0 ] && [ -z "$missing" ]; then touch $L/RELEARN_${JOB}_DONE; say "$JOB DONE ($(echo $got | wc -w) layers)"; return 0; fi
    if [ $rc != 0 ] && grep -qE "OutOfMemoryError|CUDA out of memory" $log; then
      low=$(echo $got | tr ' ' '\n' | sort -n | head -1)
      echo "out of memory; layers done: $got; lowest done ${low:-none}; not done: $missing" > $L/RELEARN_${JOB}_OOM
      say "$JOB OOM: lowest layer done ${low:-none}; not done: $missing"; return 0
    fi
  elif [ $rc = 0 ]; then touch $L/RELEARN_${JOB}_DONE; say "$JOB DONE"; return 0; fi
  touch $L/RELEARN_${JOB}_FAILED $L/RELEARN_FAILED; say "$JOB FAILED rc=$rc (see $log)"; return 1
}

JOBS=(  # ID MODEL SET KIND FAMILY SPEC DEPENDENCY, in order of preference
  "F_q9_popqa qwen3.5-9b popqa freeze - - -"
  "F_g27_popqa gemma3-27b popqa freeze - - -"
  "T_g27_popqa_C_hi gemma3-27b popqa train C hi F_g27_popqa"
  "T_g27_popqa_E_hi gemma3-27b popqa train E hi F_g27_popqa"
  "T_g27_popqa_C_lo gemma3-27b popqa train C lo F_g27_popqa"
  "T_g27_popqa_E_lo gemma3-27b popqa train E lo F_g27_popqa"
  "T_q9_popqa_C qwen3.5-9b popqa train C all F_q9_popqa"
  "T_q9_popqa_E qwen3.5-9b popqa train E all F_q9_popqa"
  "F_q9_falseqa qwen3.5-9b falseqa freeze - - -"
  "F_g27_falseqa gemma3-27b falseqa freeze - - -"
  "T_g27_falseqa_C_hi gemma3-27b falseqa train C hi F_g27_falseqa"
  "T_g27_falseqa_E_hi gemma3-27b falseqa train E hi F_g27_falseqa"
  "T_g27_falseqa_C_lo gemma3-27b falseqa train C lo F_g27_falseqa"
  "T_g27_falseqa_E_lo gemma3-27b falseqa train E lo F_g27_falseqa"
  "T_q9_falseqa_C qwen3.5-9b falseqa train C all F_q9_falseqa"
  "T_q9_falseqa_E qwen3.5-9b falseqa train E all F_q9_falseqa"
)
claim() {  # prints the first unclaimed job whose dependency is done; "ALL" when every job is claimed; "" otherwise
  local j id dep left=0
  for j in "${JOBS[@]}"; do
    set -- $j; id=$1; dep=$7
    [ -e $L/RELEARN_${id}_TAKEN ] && continue
    left=1
    if [ "$dep" = - ] || [ -e $L/RELEARN_${dep}_DONE ]; then touch $L/RELEARN_${id}_TAKEN; echo "$j"; return; fi
  done
  [ $left = 0 ] && echo ALL
}
worker() {
  local w=$1 j
  while true; do
    [ -e $L/RELEARN_FAILED ] && { say "worker $w: stop flag"; return; }
    j=$( { flock 9; claim; } 9>$QLOCK )
    [ "$j" = ALL ] && { say "worker $w: no job left"; return; }
    [ -z "$j" ] && { sleep 60; continue; }
    say "worker $w takes ${j%% *}"
    run_job $j || { say "worker $w: job failed, stopping"; return; }
  done
}

summ() {  # summ MODEL SET [--include-pilot] [RUN]: the program's judge-free summary (CPU)
  setm $1 $2 || return 2
  [ -n "${4:-}" ] && RUN=$4
  dx "" summarize --source-run $SRC --run-dir $RUN ${3:-} > /dev/null
}

case "$MODE" in
smoke)
  [ -e $L/RELEARN_SMOKE_DONE ] && { echo "smoke already done" >&2; exit 0; }
  JOB=smoke; setm qwen3.5-9b popqa || exit 2
  R1=$ROOT/$RR/popqa/qwen3.5-9b/smoke_E2; H1=$HOST/$RR/popqa/qwen3.5-9b/smoke_E2
  R2=$ROOT/$RR/popqa/qwen3.5-9b/smoke_FR; H2=$HOST/$RR/popqa/qwen3.5-9b/smoke_FR
  log=$L/relearn_1128_smoke.log; mkdir -p $H1 $H2
  PIL=(--n-check 3)
  GE="conf.wq.p1 conf_E2.wq.m1 caut.wq.p1 caut_E2.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
  GC="conf.wq.p1 caut.wq.m1 conf_E2.wq.p1 caut_E2.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.md.m1 x.rand.p1 x.rand.m1"
  GF="conf.wq.p1 conf_FR.wq.m1 caut.wq.p1 caut_FR.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
  say "smoke start (qwen3.5-9b, PopQA, 3 questions, layer 16)"
  getcard $NEEDT || exit 1
  poll_start $L/relearn_1128_smoke.mem
  st() { echo "=== $(date '+%F %T') smoke: $*" >> $log; dx $CARD "$@" >> $log 2>&1; }
  ok=1
  st freeze --source-run $SRC --run-dir $R1 --prompts-e44 $PR "${COMMON[@]}" --splits train tune "${PIL[@]}" || ok=0
  [ $ok = 1 ] && { st train --source-run $SRC --run-dir $R1 --prompts-e44 $PR "${COMMON[@]}" --e-on E2 --family E --layers 16 --val-n 1 "${PIL[@]}" || ok=0; }
  [ $ok = 1 ] && { st train --source-run $SRC --run-dir $R1 --prompts-e44 $PR "${COMMON[@]}" --e-on E2 --family C --layers 16 --val-n 1 "${PIL[@]}" || ok=0; }
  [ $ok = 1 ] && { st generate --source-run $SRC --run-dir $R1 --prompts-e44 $PR "${COMMON[@]}" --e-on E2 --family E --layers 16 --split tune --arms $GE "${PIL[@]}" || ok=0; }
  [ $ok = 1 ] && { st generate --source-run $SRC --run-dir $R1 --prompts-e44 $PR "${COMMON[@]}" --e-on E2 --family C --layers 16 --split tune --arms $GC "${PIL[@]}" || ok=0; }
  if [ $ok = 1 ]; then  # FR: the same pilot freeze (same identity: source, prompts file, model, caps), copied byte for byte
    for f in t.frozen_train.json t.frozen_tune.json; do cp -p $H1/$f $H2/$f && cmp -s $H1/$f $H2/$f || ok=0; done
  fi
  [ $ok = 1 ] && { st train --source-run $SRC --run-dir $R2 --prompts-e44 $PR "${COMMON[@]}" --e-on FR --family E --layers 16 --val-n 1 "${PIL[@]}" || ok=0; }
  [ $ok = 1 ] && { st train --source-run $SRC --run-dir $R2 --prompts-e44 $PR "${COMMON[@]}" --e-on FR --family C --layers 16 --val-n 1 "${PIL[@]}" || ok=0; }
  [ $ok = 1 ] && { st generate --source-run $SRC --run-dir $R2 --prompts-e44 $PR "${COMMON[@]}" --e-on FR --family E --layers 16 --split tune --arms $GF "${PIL[@]}" || ok=0; }
  poll_stop; putcard
  if [ $ok = 1 ]; then
    for R in $R1 $R2; do echo "=== $(date '+%F %T') smoke: summarize $R" >> $log; dx "" summarize --source-run $SRC --run-dir $R --include-pilot >> $log 2>&1 || ok=0; done
  fi
  say "smoke ok=$ok peak $(cat $L/relearn_1128_smoke.mem 2>/dev/null) MiB"
  [ $ok = 1 ] && touch $L/RELEARN_SMOKE_DONE || { touch $L/RELEARN_SMOKE_FAILED; exit 1; } ;;
full)
  [ -e $L/RELEARN_SMOKE_DONE ] || { echo "run smoke first" >&2; exit 2; }
  [ -e $L/RELEARN_FAILED ] && { echo "RELEARN_FAILED exists: look at it first" >&2; exit 2; }
  say "full start: ${#JOBS[@]} jobs, ${WORKERS:-4} workers; sha256 $(sha256sum $0 | cut -c1-16)"
  for w in $(seq 1 ${WORKERS:-4}); do worker $w & sleep 5; done
  wait
  [ -e $L/RELEARN_FAILED ] && { say "full stopped: RELEARN_FAILED"; exit 1; }
  for ms in qwen3.5-9b:popqa gemma3-27b:popqa qwen3.5-9b:falseqa gemma3-27b:falseqa; do
    summ ${ms%%:*} ${ms##*:} && say "summary ${ms}: $HRUN/summary.md" || say "summary ${ms} failed"
  done
  touch $L/RELEARN_FULL_DONE; say "full done" ;;
*) echo "usage: relearn_1128.sh smoke|full" >&2; exit 2 ;;
esac

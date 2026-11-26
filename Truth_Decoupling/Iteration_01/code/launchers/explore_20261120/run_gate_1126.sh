#!/bin/bash
# 2026-11-26, exploration (executor shen, Claude Opus 5.5): the per-question switch ("gate"), stage 1
# (discussion/work_queue/2026-11-26_queue_p.md). GPU forward passes only; no generation, no judge, no API.
# Programs (Iteration_01/diag on the server, code/diag in the repository):
#   gate_capture_1126.py  draw (CPU, once) and capture MODEL (GPU, model interpreter, one card; qwen3.8-27b two cards)
#   gate_fit_1126.py      fit MODEL (CPU, container python3, sklearn) and report
#   gate_sim_1126.py      readings 1-5 (CPU, container python3, existing answers and verdicts)
# usage (host): run_gate_1126.sh dry MODEL     argument / data check without a card (expects "No CUDA GPUs")
#               run_gate_1126.sh full          draw, five captures (lanes as run_ecmmlu_1126.sh), fits, report, simulation
# Cards only through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, retried every 60 s), held for the model's capture;
# ours plus others at most 66560 MiB per card at the start (waits otherwise). Lanes: card 0 llama3.1-8b, qwen3.5-9b,
# gemma3-12b; card 1 gemma3-27b; cards 2,3 qwen3.8-27b. Each model's fit (CPU) starts in the background after its capture.
# Markers in Iteration_01/logs/newsets_20261119: GATE_CAPTURE_<model>_DONE, GATE_FIT_<model>_DONE, GATE_FIT_DONE,
# GATE_SIM_DONE, GATE_FAILED; status lines with prefix "gate:" in crow_all.log there.
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling; ROOT=/workspace/Truth_Decoupling
cd $HOST || exit 2
L=Iteration_01/logs/newsets_20261119
TF=$ROOT/envs/tf519/bin/python
MAXMIB=66560
CAP=Iteration_01/diag/gate_capture_1126.py
FIT=Iteration_01/diag/gate_fit_1126.py
SIM=Iteration_01/diag/gate_sim_1126.py
OUT=Iteration_01/newsets/gate_1126
FITOUT=Iteration_01/diag/out_gate_1126_fit_20261126.txt
SIMOUT=Iteration_01/diag/out_gate_1126_sim_20261126.txt
MODELS="llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b"
WORKERS=32
MODE=${1:-}
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') gate: $*" >> $L/crow_all.log; }
fail() { say "FAILED: $*"; touch $L/GATE_FAILED; exit 1; }

setm() {  # setm MODEL: interpreter and memory asked per card (as run_ecmmlu_1126.sh)
  case $1 in
    llama3.1-8b) PYM=python3; NEED=20000 ;;
    qwen3.5-9b)  PYM=$TF;     NEED=22000 ;;
    qwen3.8-27b) PYM=$TF;     NEED=34000 ;;
    gemma3-12b)  PYM=python3; NEED=30000 ;;
    gemma3-27b)  PYM=python3; NEED=62000 ;;
    *) echo "unknown model $1" >&2; return 2 ;;
  esac
}
dx() {  # dx DEVICES CMD...: the container command of run_ecmmlu_1126.sh
  local dev=$1; shift
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$dev -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -w $ROOT truth_dec "$@"
}
memwait() {  # memwait CARD MIB: wait until ours plus others stays within MAXMIB
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}')
    [[ $used =~ ^[0-9]+$ ]] || return 1
    [ $((used + $2)) -le $MAXMIB ] && { say "card $1 holds $used MiB, asking $2"; return 0; }
    say "card $1 holds $used MiB, $2 more would exceed $MAXMIB; waiting"; sleep 60
  done
}
capture_one() {  # capture_one MODEL CARDS FD0: lock, memory check, one capture
  local NG=$1 CARD=$2 fd=$3 g f0 log=$L/log_gate_1126_capture_$1.log
  setm $NG || return 2
  if [ -e $L/GATE_CAPTURE_${NG}_DONE ]; then
    [ -s $OUT/$NG/capture_gate_1126.json ] || { say "$NG: GATE_CAPTURE_${NG}_DONE but no capture file"; return 1; }
    say "$NG: capture already done"; return 0
  fi
  [ -e $OUT/$NG ] && { say "$NG: $OUT/$NG exists without GATE_CAPTURE_${NG}_DONE; not overwritten"; return 1; }
  f0=$fd
  for g in ${CARD//,/ }; do
    eval "exec $fd>\"$HOST/Iteration_01/newsets/selfjudge/.gpu$g.lock\"" || { say "$NG: cannot open the lock of card $g"; return 1; }
    until flock -n $fd; do sleep 60; done
    say "$NG: holds the lock of card $g"
    fd=$((fd + 1))
  done
  for g in ${CARD//,/ }; do memwait $g $NEED || { say "$NG: cannot read the memory of card $g"; return 1; }; done
  say "$NG: capture start on card(s) $CARD"
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') $NG capture cards $CARD" >> $log
  dx $CARD $PYM -I $CAP capture $NG >> $log 2>&1
  local rc=$?
  for g in $(seq $f0 $((fd - 1))); do flock -u $g; eval "exec $g>&-"; done
  say "$NG: capture exit=$rc; card(s) $CARD released"
  [ $rc -eq 0 ] && [ -s $OUT/$NG/capture_gate_1126.json ] || { say "$NG: capture failed (see $log)"; return 1; }
  touch $L/GATE_CAPTURE_${NG}_DONE
}
fit_one() {  # fit_one MODEL: CPU fit in the container (no card)
  local NG=$1 log=$L/log_gate_1126_fit_$1.log
  if [ -e $L/GATE_FIT_${NG}_DONE ]; then say "$NG: fit already done"; return 0; fi
  say "$NG: fit start"
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') $NG fit" >> $log
  dx "" python3 -I $FIT fit $NG $WORKERS >> $log 2>&1
  local rc=$?
  say "$NG: fit exit=$rc"
  [ $rc -eq 0 ] && [ -s $OUT/$NG/fit_A_gate_1126.json ] && [ -s $OUT/$NG/fit_B_gate_1126.json ] || { say "$NG: fit failed (see $log)"; return 1; }
  touch $L/GATE_FIT_${NG}_DONE
}
lane() {  # lane CARDS FD0 MODEL...: captures in order (the next capture does not wait for a fit); each model's fit
  # starts in the background once its capture is done; the lane fails if any capture or fit fails
  local card=$1 fd=$2 m p bad=0 pids=""; shift 2
  for m in "$@"; do
    capture_one $m $card $fd || { bad=1; break; }
    fit_one $m & pids="$pids $!"
  done
  for p in $pids; do wait $p || bad=1; done
  return $bad
}

for f in $CAP $FIT $SIM; do [ -s $f ] || { echo "missing $f" >&2; exit 2; }; done
case $MODE in
  dry)
    NG=${2:?model}; setm $NG || exit 2
    [ -e $OUT/draws_gate_1126.json ] || { echo "no draw file: run the draw first" >&2; exit 2; }
    dx "" $PYM -I $CAP capture $NG
    echo "dry exit=$?"
    [ -e $OUT/$NG ] && { echo "dry run left $OUT/$NG: unexpected" >&2; exit 1; }
    exit 0 ;;
  draw)
    dx "" python3 -I $CAP draw; exit $? ;;
  full)
    command -v flock > /dev/null || fail "flock unavailable"
    [ -e $L/GATE_SIM_DONE ] && { say "already done (GATE_SIM_DONE)"; exit 0; }
    rm -f $L/GATE_FAILED
    say "start: capture $(sha256sum $CAP | cut -c1-16), fit $(sha256sum $FIT | cut -c1-16), sim $(sha256sum $SIM | cut -c1-16)"
    dx "" python3 -I $CAP draw >> $L/log_gate_1126_draw.log 2>&1 || fail "draw (see $L/log_gate_1126_draw.log)"
    lane 0 8 llama3.1-8b qwen3.5-9b gemma3-12b & PA=$!
    lane 1 12 gemma3-27b & PB=$!
    lane 2,3 16 qwen3.8-27b & PC=$!
    BAD=""
    wait $PA || BAD="$BAD card0-lane"
    wait $PB || BAD="$BAD card1-lane"
    wait $PC || BAD="$BAD cards23-lane"
    [ -z "$BAD" ] || fail "lanes failed:$BAD"
    for m in $MODELS; do [ -e $L/GATE_FIT_${m}_DONE ] || fail "no fit of $m"; done
    [ -e $FITOUT ] && fail "$FITOUT exists; not overwritten"
    dx "" python3 -I $FIT report > $FITOUT 2> $L/log_gate_1126_fit_report.err || fail "fit report (see $L/log_gate_1126_fit_report.err)"
    touch $L/GATE_FIT_DONE
    say "fit report $FITOUT $(sha256sum $FITOUT | cut -c1-16)"
    [ -e $SIMOUT ] && fail "$SIMOUT exists; not overwritten"
    dx "" python3 -B $SIM > $SIMOUT 2> $L/log_gate_1126_sim.err || fail "simulation (see $L/log_gate_1126_sim.err)"
    say "done: $SIMOUT $(sha256sum $SIMOUT | cut -c1-16)"
    touch $L/GATE_SIM_DONE ;;
  sim)
    [ -e $SIMOUT ] && { echo "$SIMOUT exists" >&2; exit 1; }
    dx "" python3 -B $SIM ;;
  *) echo "usage: run_gate_1126.sh dry MODEL | draw | full | sim" >&2; exit 2 ;;
esac

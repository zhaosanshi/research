#!/bin/bash
# 2026-11-26, exploration (executor wu, Claude Opus 5.5): MMLU by option probability when E and C are written
# TOGETHER (discussion/work_queue/2026-11-26_queue_o.md, section 5). No judge, no API, GPU only.
# Program, unchanged: Iteration_01/code/scripts/capability/cap_mcq_sweep.py (default write position "content").
# Items / ids / model config as sweep_all.sh, layer_probe.sh and depth_queue_1124.sh: the 1,000-item MMLU list
# Iteration_01/capability/sets/mmlu_1000_20261119.json; --ref Iteration_01/capability/mcq/<model>.json (v2 file).
# cap_mcq_sweep.py does not read a selfcheck record (it uses ctx.writes and its own hooks, each checked to fire once
# per forward), so no selfcheck is made here.
# Per model 10 arms (explicit main-table layers; the config's E#/C# would give other E layers for three models):
#   x; x:C<cl>.w.m2; for s in (main, larger): x:E<el>.w.p<s>; x:E<el>.w.p<s>+C<cl>.w.m{0.5,1,2}
# usage (host): run_ecmmlu_1126.sh dry MODEL     argument / load check without a card (expects "No CUDA GPUs")
#               run_ecmmlu_1126.sh sanity        llama3.1-8b, first 2 batches, all 10 arms, card 0
#               run_ecmmlu_1126.sh full          five models, then report and table
# Cards only through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, retried every 60 s), held for the model's run;
# ours plus others at most 66560 MiB per card at the start (waits otherwise). Lanes: card 0 llama3.1-8b, qwen3.5-9b,
# gemma3-12b; card 1 gemma3-27b; cards 2,3 qwen3.8-27b.
# Markers in Iteration_01/logs/newsets_20261119: ECMMLU_<model>_DONE, ECMMLU_TABLE_DONE, ECMMLU_FAILED,
# ECMMLU_SANITY_DONE; status lines with prefix "ecmmlu:" in crow_all.log there.
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling; ROOT=/workspace/Truth_Decoupling
cd $HOST || exit 2
L=Iteration_01/logs/newsets_20261119
OUT=Iteration_01/capability/ecmmlu_1126
SAN=Iteration_01/capability/ecmmlu_1126_sanity
TF=$ROOT/envs/tf519/bin/python
MAXMIB=66560
CFG=Iteration_01/code/configs/newsets/frozen_models_20261119.json
ITEMS=data_external/mmlu_20261119/all/test-00000-of-00001.parquet
IDS=Iteration_01/capability/sets/mmlu_1000_20261119.json
PROG=Iteration_01/code/scripts/capability/cap_mcq_sweep.py
TABLE=Iteration_01/diag/table_ecmmlu_1126.py
REPORT=Iteration_01/diag/out_ecmmlu_1126_report_20261126.txt
TABOUT=Iteration_01/diag/out_ecmmlu_1126_table_20261126.txt
MODELS="llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b"
MODE=${1:-}
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') ecmmlu: $*" >> $L/crow_all.log; }
fail() { say "FAILED: $*"; touch $L/ECMMLU_FAILED; exit 1; }

setm() {  # setm MODEL: interpreter, main-table E / C layers, main and larger E strength, memory asked per card
  case $1 in
    llama3.1-8b) PYM=python3; EL=16; CL=19; S1=3.5;  S2=5; NEED=20000 ;;
    qwen3.5-9b)  PYM=$TF;     EL=15; CL=16; S1=2;    S2=3; NEED=22000 ;;
    qwen3.8-27b) PYM=$TF;     EL=28; CL=33; S1=1.75; S2=3; NEED=34000 ;;
    gemma3-12b)  PYM=python3; EL=24; CL=25; S1=2.5;  S2=4; NEED=30000 ;;
    gemma3-27b)  PYM=python3; EL=28; CL=31; S1=1;    S2=2; NEED=62000 ;;
    *) echo "unknown model $1" >&2; return 2 ;;
  esac
}
arms() {  # the 10 arms of the current model
  local s k o="x x:C$CL.w.m2"
  for s in $S1 $S2; do
    o="$o x:E$EL.w.p$s"
    for k in 0.5 1 2; do o="$o x:E$EL.w.p$s+C$CL.w.m$k"; done
  done
  echo $o
}
dx() {  # dx DEVICES CMD...: the container command of sweep_all.sh / l16_all.sh mm()
  local dev=$1; shift
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$dev -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 \
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
run_one() {  # run_one MODEL CARDS FD0 KIND(full|sanity): lock, memory check, one cap_mcq_sweep.py run
  local NG=$1 CARD=$2 fd=$3 kind=$4 g out mk log first=() f0
  setm $NG || return 2
  if [ $kind = sanity ]; then
    out=$SAN/${NG}_first2_ecmmlu_1126.json; mk=$L/ECMMLU_SANITY; log=$L/log_ecmmlu_1126_sanity_$NG.log; first=(--first-batches 2)
  else
    out=$OUT/$NG.json; mk=$L/ECMMLU_$NG; log=$L/log_ecmmlu_1126_$NG.log
  fi
  if [ -e ${mk}_DONE ]; then
    [ -s $out ] || { say "$NG: ${mk}_DONE but no $out"; return 1; }
    say "$NG: already done ($out)"; return 0
  fi
  [ -e $out ] && { say "$NG: $out exists without ${mk}_DONE; not overwritten"; return 1; }
  f0=$fd
  for g in ${CARD//,/ }; do
    eval "exec $fd>\"$HOST/Iteration_01/newsets/selfjudge/.gpu$g.lock\"" || { say "$NG: cannot open the lock of card $g"; return 1; }
    until flock -n $fd; do sleep 60; done
    say "$NG: holds the lock of card $g"
    fd=$((fd + 1))
  done
  for g in ${CARD//,/ }; do memwait $g $NEED || { say "$NG: cannot read the memory of card $g"; return 1; }; done
  say "$NG ($kind): start on card(s) $CARD; arms $(arms)"
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') $NG $kind cards $CARD" >> $log
  dx $CARD $PYM -I $PROG run --models $CFG --model $NG --items $ITEMS --ids $IDS --out $out \
     --ref Iteration_01/capability/mcq/$NG.json "${first[@]}" --arms $(arms) >> $log 2>&1
  local rc=$?
  for g in $(seq $f0 $((fd - 1))); do flock -u $g; eval "exec $g>&-"; done
  say "$NG ($kind): exit=$rc; card(s) $CARD released"
  [ $rc -eq 0 ] && [ -s $out ] || { say "$NG: run failed (see $log)"; return 1; }
  grep -q "^SKIP" $log && { say "$NG: an arm was skipped (see $log)"; return 1; }
  touch ${mk}_DONE
}
lane() {  # lane CARDS FD0 MODEL...: models in order on the same card(s); stops at the first failure
  local card=$1 fd=$2 m; shift 2
  for m in "$@"; do run_one $m $card $fd full || return 1; done
}

for f in $CFG $ITEMS $IDS $PROG; do [ -s $f ] || { echo "missing $f" >&2; exit 2; }; done
case $MODE in
  dry)
    NG=${2:?model}; setm $NG || exit 2
    mkdir -p $SAN
    dx "" $PYM -I $PROG run --models $CFG --model $NG --items $ITEMS --ids $IDS --out $SAN/dry_${NG}_ecmmlu_1126.json \
       --ref Iteration_01/capability/mcq/$NG.json --first-batches 1 --arms $(arms)
    echo "dry exit=$?"
    [ -e $SAN/dry_${NG}_ecmmlu_1126.json ] && { echo "dry run wrote an output: unexpected" >&2; exit 1; }
    exit 0 ;;
  sanity)
    command -v flock > /dev/null || { echo "flock unavailable" >&2; exit 2; }
    rm -f $L/ECMMLU_SANITY_FAILED
    mkdir -p $SAN || exit 2
    run_one llama3.1-8b 0 8 sanity || { touch $L/ECMMLU_SANITY_FAILED; exit 1; }
    exit 0 ;;
  full)
    command -v flock > /dev/null || fail "flock unavailable"
    [ -s $TABLE ] || fail "no table program $TABLE"
    [ -e $L/ECMMLU_TABLE_DONE ] && { say "already done (ECMMLU_TABLE_DONE)"; exit 0; }
    rm -f $L/ECMMLU_FAILED
    mkdir -p $OUT || fail "cannot make $OUT"
    say "start: program $(sha256sum $PROG | cut -c1-16), table $(sha256sum $TABLE | cut -c1-16), ids $(sha256sum $IDS | cut -c1-16)"
    lane 0 8 llama3.1-8b qwen3.5-9b gemma3-12b & PA=$!
    lane 1 12 gemma3-27b & PB=$!
    lane 2,3 16 qwen3.8-27b & PC=$!
    BAD=""
    wait $PA || BAD="$BAD card0-lane"
    wait $PB || BAD="$BAD card1-lane"
    wait $PC || BAD="$BAD cards23-lane"
    [ -z "$BAD" ] || fail "lanes failed:$BAD"
    J=""; for m in $MODELS; do [ -s $OUT/$m.json ] || fail "no $OUT/$m.json"; J="$J $OUT/$m.json"; done
    [ -e $REPORT ] && fail "$REPORT exists; not overwritten"
    [ -e $TABOUT ] && fail "$TABOUT exists; not overwritten"
    python3 -I $PROG report $J > $REPORT 2> $L/log_ecmmlu_1126_report.err || fail "report"
    python3 -I $TABLE --report $REPORT $J > $TABOUT 2> $L/log_ecmmlu_1126_table.err || fail "table (see $L/log_ecmmlu_1126_table.err)"
    say "done: $REPORT $(sha256sum $REPORT | cut -c1-16), $TABOUT $(sha256sum $TABOUT | cut -c1-16)"
    touch $L/ECMMLU_TABLE_DONE ;;
  *) echo "usage: run_ecmmlu_1126.sh dry MODEL | sanity | full" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# Exploration chain d of 2026-11-13 (Claude; from_claude/44 sections 2.2 and 5.4). Llama's C at strengths 0.5 and 0.25 has
# no random or mean-difference control yet (its 11-11 strength sweep, proB on layers 19 and 31, has only the trained
# direction). This chain gives Llama the 9B's C trial exactly: the eight control arms (md and rand, each sign, 0.5 and 0.25)
# are generated with the round-1 C directions unchanged (E1 rows, as trained on 11-11) on layers 19 and 31, then the 9B's
# C trial set (C_S: w, md, rand at +-0.5 and +-0.25; base x) is judged per layer, AllenAI then DeepSeek Pro (cache seeded
# from that layer's proB work dir, which judged the four w arms and the base), and tables at 0.5 and 0.25 are made.
# Same code copy / source directory rules as chain b (Llama's 11-11 run: code_e45_20261112 + results_qa_das_v1u full;
# tables in the 13b copy next to this file). GPU 3, once the 09:38 chain's q8_run_E_b and chain b's AllenAI passes are over
# (red team 11:56: one of our GPU jobs at a time there); chain c's layers 9-17 wait for this chain's AllenAI passes.
# Logs and flags in logs/explore_d/.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}                     # overrides: dry runs only
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/explore_d
POLL=${CHAIN_POLL:-30}
G=${CHAIN_GPU:-3}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain d holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if env "${CLEAN[@]}" "$@" bash "$RUNNER" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.DONE"; note "done  $name"
  else
    echo "failed, see $name.log" > "$LOG/$name.FAIL"; note "FAIL  $name"; return 1
  fi
}
give_up() {
  local n
  for n in "$@"; do
    [ -e "$LOG/$n.DONE" ] || [ -e "$LOG/$n.FAIL" ] || { echo "not tried: an earlier job of its group failed" > "$LOG/$n.FAIL"; note "skip  $n"; }
  done
}
wait_done() {
  local f
  for f in "$@"; do
    until [ -e "$LOG/$f.DONE" ]; do [ -e "$LOG/$f.FAIL" ] && return 1; sleep "$POLL"; done
  done
}
wait_over() {  # wait_over DIR NAME...: until each NAME in DIR has DONE or FAIL
  local d=$1 f; shift
  for f in "$@"; do
    until [ -e "$d/$f.DONE" ] || [ -e "$d/$f.FAIL" ]; do sleep "$POLL"; done
  done
}

MAINLL=$W/results_e44_factorial_das/llama3.1-8b
V=(MODEL=llama3.1-8b RUN_DIR=$MAINLL SOURCE_DIR=$W/results_qa_das_v1u/llama3.1-8b/full E_ON=E1 BRANCH=C)
OLD=$W/code_e45_20261112
NEW=$W/code_ff_explore_20261113b
CTRL="x.md.p0.5 x.md.m0.5 x.rand.p0.5 x.rand.m0.5 x.md.p0.25 x.md.m0.25 x.rand.p0.25 x.rand.m0.25"
C_S="x.w.p0.5 x.w.m0.5 x.md.p0.5 x.md.m0.5 x.rand.p0.5 x.rand.m0.5 x.w.p0.25 x.w.m0.25 x.md.p0.25 x.md.m0.25 x.rand.p0.25 x.rand.m0.25"
LAYERS_C="19 31"

LLB=""; for L in 8 9 10 11 12 13 14 15 16 17 18; do LLB="$LLB ll_e2_allenai_L$L"; done
gpu_lane() {
  local L
  wait_over "$LOGS/explore" q8_run_E_b
  wait_over "$LOGS/explore_b" $LLB
  job llc_extra "${V[@]}" CODE_PATH=$OLD STAGE=extra GPU=$G LAYERS="$LAYERS_C" ARMS="$CTRL" || { give_up llc_allenai_L19 llc_allenai_L31; return; }
  for L in $LAYERS_C; do
    job llc_allenai_L$L "${V[@]}" CODE_PATH=$OLD STAGE=pro GPU=$G LAYER=$L TAG=S JUDGES=allenai ARMS="$C_S" BASES=x || true
  done
}
pro_lane() {  # pro_lane LAYER
  local L=$1 S
  { wait_done llc_allenai_L$L &&
    job llc_pro_L$L "${V[@]}" CODE_PATH=$OLD STAGE=pro LAYER=$L TAG=S JUDGES=pro ARMS="$C_S" BASES=x CACHE_FROM=$MAINLL/judge_tune_proB_C_L$L/judge2/cache; } \
    || { give_up llc_pro_L$L llc_tables_L${L}_s0.5 llc_tables_L${L}_s0.25; return; }
  for S in 0.5 0.25; do job llc_tables_L${L}_s$S "${V[@]}" CODE_PATH=$NEW STAGE=tables LAYERS=$L TAG=S STRENGTH=$S || true; done
}

note "chain start (pid $$)"
gpu_lane &
pro_lane 19 & pro_lane 31 &
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

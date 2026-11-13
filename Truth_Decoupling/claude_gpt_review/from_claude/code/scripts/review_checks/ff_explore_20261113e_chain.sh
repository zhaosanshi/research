#!/usr/bin/env bash
# Exploration chain e of 2026-11-13 (Claude; from_claude/44 section 5.4): the 9B's C trial on Qwen3-8B. Round 1's C
# directions of Qwen3-8B unchanged (E1 rows, strength 1 as trained), only the strength changes: the twelve C_S arms (w, md,
# rand at +-0.5 and +-0.25, base x) on the layers given in C_LAYERS -- by the rule written in 44 before the screen: the
# layers plan 43 section 5 sends to Pro for C (all shortlisted layers) -- then AllenAI and DeepSeek Pro per layer and tables
# at 0.5 and 0.25, exactly as chain 1 did for the 9B (c_s_*).
#   C_LAYERS="..." [CHAIN_GPUS="g ..."] ff_explore_20261113e_chain.sh
# The generation and the AllenAI passes of layer i go to GPU i mod (number of CHAIN_GPUS) (default "1 3": the cards chain c
# uses least at that time; ours share a card at most two at a time there). A layer that is also a C strength-sweep layer of
# the round-1 Pro chain (newmodel_e44_pro_chain.sh: proB_C_L<layer>, which judges the four w arms at +-0.25 / +-0.5) waits
# for that job to be over and seeds its Pro cache from it; otherwise the cache is the step-0 one (runner default).
# Runner: the 13b copy next to this file. Logs and flags in logs/explore_e/.
set -uo pipefail
: "${C_LAYERS:?give the C Pro layers of Qwen3-8B}"
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}                     # overrides: dry runs only
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/explore_e
POLL=${CHAIN_POLL:-30}
read -r -a GPUS <<< "${CHAIN_GPUS:-1 3}"
C_DOSE_PRO=" ${C_DOSE_LAYERS:-} "                               # the pro chain's C dose layers (their proB caches)
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain e holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR
       -u C_LAYERS -u C_DOSE_LAYERS)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if env "${CLEAN[@]}" CODE_PATH=$W/code_ff_explore_20261113b "$@" bash "$RUNNER" > "$LOG/$name.log" 2>&1; then
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
wait_over() {  # wait_over FILE_PREFIX...: until FILE_PREFIX.DONE or FILE_PREFIX.FAIL exists
  local p
  for p in "$@"; do
    until [ -e "$p.DONE" ] || [ -e "$p.FAIL" ]; do sleep "$POLL"; done
  done
}

Q8=qwen3-8b
C_S="x.w.p0.5 x.w.m0.5 x.md.p0.5 x.md.m0.5 x.rand.p0.5 x.rand.m0.5 x.w.p0.25 x.w.m0.25 x.md.p0.25 x.md.m0.25 x.rand.p0.25 x.rand.m0.25"
note "chain start (pid $$): C_LAYERS=[$C_LAYERS] C_DOSE_LAYERS=[${C_DOSE_LAYERS:-}] GPUS=[${GPUS[*]}]"

gpu_part() {  # gpu_part L GPU: extra arms, then AllenAI on the same card
  local L=$1 g=$2
  # a C dose layer: the round-1 Pro chain generates the four w arms there (extra_C_dose); let it finish them first, so the
  # same file is never written by two jobs (the core then validates and skips them)
  [[ "$C_DOSE_PRO" == *" $L "* ]] && wait_over "$LOGS/${Q8}_extra_C_dose"
  { job q8c_extra_L$L MODEL=$Q8 STAGE=extra BRANCH=C GPU=$g LAYERS=$L ARMS="$C_S" &&
    job q8c_allenai_L$L MODEL=$Q8 STAGE=pro BRANCH=C LAYER=$L TAG=S JUDGES=allenai GPU=$g ARMS="$C_S" BASES=x; } \
    || give_up q8c_allenai_L$L
}
pro_part() {  # pro_part L: DeepSeek Pro (no card) once the layer's AllenAI pass is done, then its tables
  local L=$1 S cf=""
  wait_done q8c_allenai_L$L || { give_up q8c_pro_L$L q8c_tables_L${L}_s0.5 q8c_tables_L${L}_s0.25; return 0; }
  if [[ "$C_DOSE_PRO" == *" $L "* ]]; then
    wait_over "$LOGS/${Q8}_proB_C_L$L"
    [ -e "$LOGS/${Q8}_proB_C_L$L.DONE" ] && cf=$W/results_newmodels_ff_20261112/e44/$Q8/judge_tune_proB_C_L$L/judge2/cache
  fi
  job q8c_pro_L$L MODEL=$Q8 STAGE=pro BRANCH=C LAYER=$L TAG=S JUDGES=pro ARMS="$C_S" BASES=x ${cf:+CACHE_FROM=$cf} \
    || { give_up q8c_tables_L${L}_s0.5 q8c_tables_L${L}_s0.25; return 0; }
  for S in 0.5 0.25; do job q8c_tables_L${L}_s$S MODEL=$Q8 STAGE=tables BRANCH=C LAYERS=$L TAG=S STRENGTH=$S || true; done
}
lane() {  # lane INDEX: the GPU parts of the layers whose position mod the number of cards is INDEX, one after another
  local i=$1 k=0 L
  for L in $C_LAYERS; do
    [ $((k % ${#GPUS[@]})) = "$i" ] && gpu_part "$L" "${GPUS[$i]}"
    k=$((k + 1))
  done
}
for i in "${!GPUS[@]}"; do lane "$i" & done
for L in $C_LAYERS; do pro_part "$L" & done
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

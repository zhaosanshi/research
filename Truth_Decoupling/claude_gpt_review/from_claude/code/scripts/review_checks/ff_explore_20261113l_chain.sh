#!/usr/bin/env bash
# Chain l of 2026-11-13 (Claude, 20:5x; 赵磊 20:4x "跑"): free-form E44 on Qwen3.8-27B, first part -- the steps that do not
# depend on any rule still to be written down: the source directory, the three-question execution checks (E and C, layers
# L/8 and L/2 of the pilot; also shows whether the 27B and its training fit on one card), step 0 (the 11 prompts frozen on
# train + tune, the no-write readouts, AllenAI) and DeepSeek Pro on the step-0 tune cells. The truth sentence of the run is
# E2 (the sentence written down for every model on 11-13); step 0 freezes all 11 prompts whatever E_ON is.
# Runner newmodel_e44.sh next to this file, code copy code_ff_explore_20261113b (the 9B's copy, same core). Card 0.
# Mechanics as chains b-k: clean environment per job, DONE flags skip, FAIL marks the rest "not tried", one chain at a time.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore_l}
G=${CHAIN_GPU:-0}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain l holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
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
V=(MODEL=qwen3.8-27b E_ON=E2 CODE_PATH=$W/code_ff_explore_20261113b)
note "chain start (pid $$): card $G"
( cd "$HERE" && sha256sum newmodel_e44.sh ff_explore_20261113l_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
{ job q27_source "${V[@]}" STAGE=source &&
  job q27_pilot_E "${V[@]}" STAGE=pilot BRANCH=E GPU=$G &&
  job q27_pilot_C "${V[@]}" STAGE=pilot BRANCH=C GPU=$G &&
  job q27_step0 "${V[@]}" STAGE=step0 GPU=$G &&
  job q27_step0pro "${V[@]}" STAGE=step0pro; } || give_up q27_pilot_E q27_pilot_C q27_step0 q27_step0pro
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

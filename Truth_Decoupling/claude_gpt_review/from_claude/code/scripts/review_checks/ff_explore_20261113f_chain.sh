#!/usr/bin/env bash
# Exploration chain f of 2026-11-13 (Claude; from_claude/44 section 5.6): DeepSeek Pro on Qwen3-8B's E2 run (chain c) for
# the layers written down before any E2 result was seen: the layers plan 43 section 5 sends to Pro for Qwen3-8B's round-1
# E (E1), i.e. the diagnostic layer d of its frozen screen and d+1, d+2 (no layer to be qualified met the main effect under
# AllenAI). From the screen of 13:27: d = 25, so 25 26 27. Each layer: Pro on the work dir chain c judged with AllenAI (the
# export is redone and must be identical; no GPU), then tables (strength 1, both judges; x base and the exploratory conf
# base). One layer after another (DeepSeek Pro is shared with two other chains at this time).
#   E2_PRO_LAYERS="25 26 27" ff_explore_20261113f_chain.sh
# Runner: the 13b copy next to this file. Logs and flags in logs/explore_f/.
set -uo pipefail
: "${E2_PRO_LAYERS:?}"
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/explore_f
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain f holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR -u E2_PRO_LAYERS)
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
V=(MODEL=qwen3-8b VARIANT=E2 E_ON=E2 BRANCH=E)
E2_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 conf_E2.wq.m1 conf.iso.p1 caut.wq.p1 caut_E2.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_BASES="x x_F conf conf_E2 caut caut_E2"
note "chain start (pid $$): E2_PRO_LAYERS=[$E2_PRO_LAYERS]"
for L in $E2_PRO_LAYERS; do
  [ -e "$LOGS/explore_c/q8e2_allenai_L$L.DONE" ] || { note "skip  q8e2_pro_L$L (no AllenAI pass in chain c)"; continue; }
  job q8e2_pro_L$L "${V[@]}" STAGE=pro LAYER=$L TAG=X JUDGES=pro ARMS="$E2_PROA" BASES="$E2_BASES" &&
    job q8e2_tables2_L$L "${V[@]}" STAGE=tables LAYERS=$L TAG=X STRENGTH=1 || true
done
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed"

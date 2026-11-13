#!/usr/bin/env bash
# Fixer of 2026-11-14 00:4x (Claude): chain n's AllenAI passes at layers 16 and 18 stopped at 00:33 / 00:35 only because two
# EMPTY answers (tqa:780 under E_L16_x-w-p1 and E_L16_x-rand-m1; the same at L18) had no verdict -- the AllenAI judging
# itself was complete. The runner now counts an empty answer as an abstention under both judges (two-axis standard of
# 2026-11-08, "空输出一律 C"; replaced 00:39 by an atomic mv), other missing verdicts still stop. This reruns those two
# jobs and their AllenAI tables with the chain's own runner call, log names and DONE / FAIL flags (chain n reads them only in
# pick_top, after its third third), on card 1 only (AllenAI is one 7B at a time; card 0 holds the stage-B Llama lane and
# the 27B's first half). The judging result does not depend on the card (runtime identity records the device type only).
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=$HERE/newmodel_e44.sh
LOG=$R/results_newmodels_ff_20261112/logs/explore_n
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name (fixer ff_explore_20261114n_fix.sh, pid $$)"
  [ -e "$LOG/$name.log" ] && mv -f "$LOG/$name.log" "$LOG/$name.log.prev"
  if env "${CLEAN[@]}" "$@" bash "$RUNNER" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.DONE"; note "done  $name"
  else
    echo "failed, see $name.log" > "$LOG/$name.FAIL"; note "FAIL  $name"; return 1
  fi
}
V=(MODEL=qwen3.8-27b-dm E_ON=E2 BRANCH=E CODE_PATH=$W/code_ff_explore_20261113b)
E2_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 conf_E2.wq.m1 conf.iso.p1 caut.wq.p1 caut_E2.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_BASES="x x_F conf conf_E2 caut caut_E2"
( cd "$HERE" && sha256sum newmodel_e44.sh e44_round1_tables.py ff_explore_20261114n_fix.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
for L in 16 18; do
  { job q27e_allenai_L$L "${V[@]}" STAGE=pro GPU=1 LAYER=$L TAG=X JUDGES=allenai ARMS="$E2_PROA" BASES="$E2_BASES" &&
    job q27e_atables_L$L "${V[@]}" STAGE=tables LAYERS=$L TAG=X STRENGTH=1 JUDGES=allenai; } || true
done

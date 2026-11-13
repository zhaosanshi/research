#!/usr/bin/env bash
# Starter of 2026-11-13 22:5x (Claude): the first training job of chain n (27B E, cards 0,1) and of chain o (27B C, cards
# 2,3), started while the two chains are under red-team review so that the four cards do not wait for it. Each is the
# chain's own job: the same runner call, log file and DONE / FAIL flag; NAME.RUNNING (touched first, removed last, after
# DONE or FAIL is written) tells the chain to wait for it and then skip it (DONE) or rerun it (FAIL).
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=$HERE/newmodel_e44.sh
CODE=$W/code_ff_explore_20261113b
LOGS=$R/results_newmodels_ff_20261112/logs
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR)
start() {  # start LOG_DIR NAME VAR=value ...
  local log=$1 name=$2; shift 2
  mkdir -p "$log"
  [ -e "$log/$name.DONE" ] && return 0
  [ -e "$log/$name.RUNNING" ] && { echo "$log/$name.RUNNING exists: not started twice" >&2; return 1; }
  touch "$log/$name.RUNNING"; rm -f "$log/$name.FAIL"
  echo "$(date '+%F %T') start $name (by hand: ff_explore_20261113no_start.sh, pid $$)" >> "$log/chain_jobs.log"
  if env "${CLEAN[@]}" "$@" bash "$RUNNER" > "$log/$name.log" 2>&1; then
    touch "$log/$name.DONE"; echo "$(date '+%F %T') done  $name" >> "$log/chain_jobs.log"
  else
    echo "failed, see $name.log" > "$log/$name.FAIL"; echo "$(date '+%F %T') FAIL  $name" >> "$log/chain_jobs.log"
  fi
  rm -f "$log/$name.RUNNING"
}
( cd "$HERE" && sha256sum newmodel_e44.sh ff_explore_20261113no_start.sh ) >> "$LOGS/explore_n_o_start.sha" 2>&1 || true
start "$LOGS/explore_n" q27e_run_a MODEL=qwen3.8-27b-dm E_ON=E2 BRANCH=E CODE_PATH=$CODE STAGE=run GPU=0,1 LAYERS="16 18 20" &
start "$LOGS/explore_o" q27c_run_a MODEL=qwen3.8-27b-dm E_ON=E2 BRANCH=C CODE_PATH=$CODE STAGE=run GPU=2,3 LAYERS="19 21 23" &
wait

#!/usr/bin/env bash
# Chain o of 2026-11-13 (Claude, 22:4x; 赵磊 20:4x "跑" and 22:3x "可以停其他的卡其他业务，全都咱们用"): the C lane of
# Qwen3.8-27B under the rules written down in from_claude/45 (v1), on cards 2,3 (model config qwen3.8-27b-dm: one model
# split over both cards; chain m made the source directory and step 0, truth sentence E2).
#   1. C directions at the band layers 19-37, every other layer (45 section 3: band 19-38), in three parts; per part: train
#      + round 1's C arms + reference readout (STAGE=run).
#   2. Round 1's C screen on all ten layers (45 section 3: "C 按首轮的筛选规则（43 第 5 节）"): AllenAI on the run arms
#      (STAGE=judge), the frozen shortlist rule (STAGE=screen, e44_round1_tables.py unchanged).
#   3. Every shortlisted layer (the screen's "shortlist layers" line, as chain e took them for Qwen3-8B): the twelve C_S arms
#      (w, md, rand at +-0.5 and +-0.25, base x; R5 reads toward caution at 0.25), AllenAI on the cards (TAG=S), DeepSeek Pro
#      (no card; reply cache: step 0's, the runner default), tables at 0.5 and 0.25.
# R5 (which shortlisted layers meet the frozen criterion v4 under both judges at 0.25) and R6 (E+C) are read and run after
# this chain and chain n, from their tables.
# Runner newmodel_e44.sh next to this file; code copy code_ff_explore_20261113b (chain m's; same core as the 9B's runs).
# Mechanics as chain n (RUNNING flags of jobs started by hand included). Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}                     # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore_o}
POLL=${CHAIN_POLL:-30}
G=${CHAIN_GPU:-2,3}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain o holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
running_wait() {  # a job started by hand (NAME.RUNNING): wait until it is over; a flag without a live starter = FAIL (red team 23:05)
  while [ -e "$LOG/$1.RUNNING" ]; do
    if ! pgrep -f ff_explore_20261113no_start.sh > /dev/null; then
      [ -e "$LOG/$1.RUNNING" ] || break                     # the starter ended between the two checks
      echo "stale RUNNING flag: the starter is gone; not rerun (an orphaned process may still write)" > "$LOG/$1.FAIL"
      note "FAIL  $1 (stale RUNNING flag, starter gone)"; return 1
    fi
    sleep "$POLL"
  done
}
job() {  # job NAME VAR=value ...: one runner call in a clean environment; skipped when NAME.DONE exists
  local name=$1; shift
  running_wait "$name" || return 1
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  [ -e "$LOG/$name.log" ] && mv -f "$LOG/$name.log" "$LOG/$name.log.prev"   # keep the log of a failed earlier try
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

CODE=$W/code_ff_explore_20261113b
V=(MODEL=qwen3.8-27b-dm E_ON=E2 BRANCH=C CODE_PATH=$CODE)
PARTS=("19 21 23" "25 27 29" "31 33 35 37")
KEYS=(a b c)
C_LAYERS="${PARTS[*]}"
C_S="x.w.p0.5 x.w.m0.5 x.md.p0.5 x.md.m0.5 x.rand.p0.5 x.rand.m0.5 x.w.p0.25 x.w.m0.25 x.md.p0.25 x.md.m0.25 x.rand.p0.25 x.rand.m0.25"
[ -f "$R/${CODE#$W/}/scripts/review_checks/newmodel_e44.sh" ] || [ -n "${CHAIN_RUNNER:-}" ] || { echo "the 13b copy has no runner" >&2; exit 3; }

pro_lane() {  # pro_lane LAYER: DeepSeek Pro (no card) once the layer's AllenAI pass is done, then its tables
  local L=$1 S
  wait_done q27c_allenai_L$L || { give_up q27c_pro_L$L q27c_tables_L${L}_s0.5 q27c_tables_L${L}_s0.25; return 0; }
  job q27c_pro_L$L "${V[@]}" STAGE=pro LAYER=$L TAG=S JUDGES=pro ARMS="$C_S" BASES=x \
    || { give_up q27c_tables_L${L}_s0.5 q27c_tables_L${L}_s0.25; return 0; }
  for S in 0.5 0.25; do job q27c_tables_L${L}_s$S "${V[@]}" STAGE=tables LAYERS=$L TAG=S STRENGTH=$S || true; done
}

note "chain start (pid $$): card $G, layers [$C_LAYERS]"
( cd "$HERE" && sha256sum newmodel_e44.sh e44_round1_tables.py ff_explore_20261113o_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
ok=1
for i in 0 1 2; do job q27c_run_${KEYS[$i]} "${V[@]}" STAGE=run GPU=$G LAYERS="${PARTS[$i]}" || ok=0; done
if [ $ok = 1 ] && job q27c_judge "${V[@]}" STAGE=judge GPU=$G LAYERS="$C_LAYERS" && job q27c_screen "${V[@]}" STAGE=screen; then
  SHORT=$(sed -n 's/^shortlist layers: *//p' "$LOG/q27c_screen.log")
  note "C shortlist (round 1's screen): [$SHORT]"
  for L in $SHORT; do pro_lane "$L" & done
  for L in $SHORT; do
    { job q27c_extra_L$L "${V[@]}" STAGE=extra GPU=$G LAYERS=$L ARMS="$C_S" &&
      job q27c_allenai_L$L "${V[@]}" STAGE=pro GPU=$G LAYER=$L TAG=S JUDGES=allenai ARMS="$C_S" BASES=x; } || give_up q27c_allenai_L$L
  done
  wait
else
  [ $ok = 1 ] || note "a training part failed: the screen needs all ten layers, judge and screen not tried"
  give_up q27c_judge q27c_screen
fi
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

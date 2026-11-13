#!/usr/bin/env bash
# Chain n of 2026-11-13 (Claude, 22:4x; 赵磊 20:4x "跑" and 22:3x "可以停其他的卡其他业务，全都咱们用"): the E lane of
# Qwen3.8-27B under the rules written down in from_claude/45 (v1), on cards 0,1 (model config qwen3.8-27b-dm: one model
# split over both cards; chain m made the source directory and step 0, truth sentence E2).
#   1. E2 directions at the band layers 16-32, every other layer (45 section 3), in thirds. Per third: train + round 1's arms
#      + reference readout (STAGE=run); the six extra arms (conf.w.p1 and its mean-difference and random controls,
#      x.rand.m1, x_F.w.p1); then per layer AllenAI on the E2 judged set of the other models (TAG=X: 15 arms on 6 prompts,
#      as chain c) and its AllenAI tables.
#   2. R3, first half: AllenAI dR of conf.w.p1 against conf on all nine layers; the three largest (ties: the lower layer)
#      go to DeepSeek Pro (ff_rule_pick.py top; this rule is in 45 section 3, written before this chain ran).
#   3. Those three layers: strengths 2 and 3 on the must-answer base with their controls (chain g's arms; TAG=D, one work
#      dir per layer holds strengths 1 / 2 / 3), AllenAI on the cards; DeepSeek Pro (no card) on TAG=X, then on TAG=D
#      (reply cache seeded from that layer's TAG=X work dir, as chain g); tables.
#   4. R3, second half: the largest AllenAI + Pro dR at strength 1 (ties: the lower layer) -> r3_choice.json / .md in the
#      log dir; then the dose summary of the three layers (ff_explore_dose_table.py).
# Runner newmodel_e44.sh next to this file; code copy code_ff_explore_20261113b (chain m's; same core as the 9B's runs).
# Mechanics as chains b-m: clean environment per job, DONE flags skip, FAIL marks the rest of a group "not tried", one chain
# n at a time (lock directory). A job whose NAME.RUNNING flag exists was started by hand (ff_explore_20261113no_start.sh)
# with the same runner call, log and flags: the chain waits for it to end and then skips it (DONE) or reruns it (FAIL).
# Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}                     # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore_n}
POLL=${CHAIN_POLL:-30}
G=${CHAIN_GPU:-0,1}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain n holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
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
tjob() {  # tjob NAME COMMAND...: a read-only table command (tx) with the same DONE / FAIL flags; stdout to NAME.md
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if "$@" > "$LOG/$name.md" 2> "$LOG/$name.log"; then
    touch "$LOG/$name.DONE"; note "done  $name"
  else
    echo "failed, see $name.log" > "$LOG/$name.FAIL"; note "FAIL  $name"; return 1
  fi
}

CODE=$W/code_ff_explore_20261113b
RUN=$W/results_newmodels_ff_20261112/e44/qwen3.8-27b-dm
LOGC=$W/${LOG#$R/}                                                # the log dir as the container sees it
V=(MODEL=qwen3.8-27b-dm E_ON=E2 BRANCH=E CODE_PATH=$CODE)
THIRDS=("16 18 20" "22 24 26" "28 30 32")
KEYS=(a b c)
E_LAYERS="${THIRDS[*]}"
E_EXTRA_A="x.rand.m1 x_F.w.p1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 conf_E2.wq.m1 conf.iso.p1 caut.wq.p1 caut_E2.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_BASES="x x_F conf conf_E2 caut caut_E2"
NEWARMS="conf.w.p2 conf.w.p3 conf.rand.p2 conf.rand.m2 conf.rand.p3 conf.rand.m3 conf.md.p2 conf.md.p3"
DOSE="conf.w.p1 conf.rand.p1 conf.rand.m1 conf.md.p1 $NEWARMS"
DBASES="conf conf_E2"
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec python3)
tx() {  # the read-only table scripts in the 13b copy (CHAIN_TX_DRY=1: print the command only, for dry runs)
  if [ -n "${CHAIN_TX_DRY:-}" ]; then echo "tx $*"; else "${TX[@]}" "$@"; fi
}
for f in newmodel_e44.sh e44_round1_tables.py ff_explore_dose_table.py ff_rule_pick.py; do   # fail now, not after hours
  [ -n "${CHAIN_TX_DRY:-}" ] || [ -f "$R/${CODE#$W/}/scripts/review_checks/$f" ] || { echo "the 13b copy has no $f" >&2; exit 3; }
  [ -n "${CHAIN_TX_DRY:-}" ] || cmp -s "$HERE/$f" "$R/${CODE#$W/}/scripts/review_checks/$f" \
    || { echo "$f next to this chain differs from the 13b copy that tx runs" >&2; exit 3; }
done
case "$LOG" in "$R"/*) ;; *) [ -n "${CHAIN_TX_DRY:-}" ] || { echo "LOG must be under $R (the container reads r3_choice there)" >&2; exit 2; } ;; esac

third() {  # third KEY "LAYERS": train + round 1's arms + readout, the six extra arms, then AllenAI (TAG=X) and its tables per layer
  local q=$1 layers=$2 L names="q27e_extra_$1"
  for L in $layers; do names="$names q27e_allenai_L$L q27e_atables_L$L"; done
  job q27e_run_$q "${V[@]}" STAGE=run GPU=$G LAYERS="$layers" || { give_up $names; return 0; }
  job q27e_extra_$q "${V[@]}" STAGE=extra GPU=$G LAYERS="$layers" ARMS="$E_EXTRA_A" || { give_up $names; return 0; }
  for L in $layers; do
    { job q27e_allenai_L$L "${V[@]}" STAGE=pro GPU=$G LAYER=$L TAG=X JUDGES=allenai ARMS="$E2_PROA" BASES="$E2_BASES" &&
      job q27e_atables_L$L "${V[@]}" STAGE=tables LAYERS=$L TAG=X STRENGTH=1 JUDGES=allenai; } || give_up q27e_atables_L$L
  done
}
pick_top() {  # R3 first half -> TOP3 (needs the AllenAI pass of all nine layers)
  local L f=""
  for L in $E_LAYERS; do f="$f q27e_allenai_L$L"; done
  wait_done $f || return 1
  tjob r3_top3 tx scripts/review_checks/ff_rule_pick.py top --run "$RUN" --layers $E_LAYERS --k 3 || return 1
  TOP3=$(sed -n 's/^top: //p' "$LOG/r3_top3.md")
  [ "$(echo $TOP3 | wc -w)" = 3 ] || { echo "r3_top3.md does not name three layers" > "$LOG/r3_top3.FAIL"; rm -f "$LOG/r3_top3.DONE"; return 1; }
  note "R3 top three by AllenAI (to Pro): $TOP3"
}
dose_gpu() {  # strengths 2 and 3 on the three layers, then AllenAI (TAG=D) per layer, on the cards
  local L names=""
  for L in $TOP3; do names="$names q27e_dallenai_L$L"; done
  job q27e_dextra "${V[@]}" STAGE=extra GPU=$G LAYERS="$TOP3" ARMS="$NEWARMS" || { give_up $names; return 0; }
  for L in $TOP3; do
    job q27e_dallenai_L$L "${V[@]}" STAGE=pro GPU=$G LAYER=$L TAG=D JUDGES=allenai ARMS="$DOSE" BASES="$DBASES" || true
  done
}
dtables() {  # conf base, strengths 1 2 3, both judges (read-only)
  local L=$1 S J
  for S in 1 2 3; do for J in allenai pro; do
    tx scripts/review_checks/e44_round1_tables.py cells "$RUN" E "$RUN/judge_tune_proD_E_L$L" --judge "$J" --layers "$L" --base conf --strength "$S" || return 1
  done; done
}
pro_lane() {  # pro_lane LAYER: Pro on TAG=X and its tables (both judges); then, once the layer's TAG=D AllenAI pass is done, Pro on TAG=D
  local L=$1
  job q27e_pro_L$L "${V[@]}" STAGE=pro LAYER=$L TAG=X JUDGES=pro ARMS="$E2_PROA" BASES="$E2_BASES" \
    || { give_up q27e_tables_L$L q27e_dpro_L$L q27e_dtables_L$L; return 0; }
  job q27e_tables_L$L "${V[@]}" STAGE=tables LAYERS=$L TAG=X STRENGTH=1 || true   # read-only: never blocks the Pro D pass
  { wait_done q27e_dallenai_L$L &&
    job q27e_dpro_L$L "${V[@]}" STAGE=pro LAYER=$L TAG=D JUDGES=pro ARMS="$DOSE" BASES="$DBASES" CACHE_FROM=$RUN/judge_tune_proX_E_L$L/judge2/cache &&
    tjob q27e_dtables_L$L dtables "$L"; } || give_up q27e_dpro_L$L q27e_dtables_L$L
}
r3_choice() {  # R3 second half, once Pro has judged TAG=X on the three layers
  local L f=""
  for L in $TOP3; do f="$f q27e_pro_L$L"; done
  wait_done $f || { give_up r3_choice; return 0; }
  tjob r3_choice tx scripts/review_checks/ff_rule_pick.py pick --run "$RUN" --layers $TOP3 --out "$LOGC/r3_choice.json" || return 0
  note "R3 choice: layer $(sed -n 's/^chosen: //p' "$LOG/r3_choice.md")"
}
dose_summary() {
  local L f="" args=()
  for L in $TOP3; do f="$f q27e_dtables_L$L"; args+=(--work "27B:$L:$RUN/judge_tune_proD_E_L$L"); done
  wait_done $f || { give_up dose_summary; return 0; }
  tjob dose_summary tx scripts/review_checks/ff_explore_dose_table.py --e-on E2 "${args[@]}" || true
}

note "chain start (pid $$): card $G, layers [$E_LAYERS]"
( cd "$HERE" && sha256sum newmodel_e44.sh e44_round1_tables.py ff_explore_dose_table.py ff_rule_pick.py ff_explore_20261113n_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
for i in 0 1 2; do third "${KEYS[$i]}" "${THIRDS[$i]}"; done
if pick_top; then
  dose_gpu &
  for L in $TOP3; do pro_lane "$L" & done
  r3_choice &
  wait
  dose_summary
else
  note "R3 first half not possible (a layer's AllenAI pass failed or the pick failed): Pro, dose and choice not tried"
fi
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

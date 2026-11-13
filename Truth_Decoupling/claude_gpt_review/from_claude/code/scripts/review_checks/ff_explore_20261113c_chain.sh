#!/usr/bin/env bash
# Exploration chain c of 2026-11-13 (Claude; from_claude/44 section 5.5). Qwen3-8B with the truth sentence E2 instead of E1
# (VARIANT=E2, its step 0 reused byte for byte; the only change against its round 1, as for the 9B): the E family trained on
# ALL 36 layers (no layer region is known for this model yet; Llama's E2 side branch was trained on all 32), with round 1's
# arms, then the six arms the criterion and the must-answer base need (conf.w.p1 and its mean-difference and random
# controls, x.rand.m1, x_F.w.p1), then AllenAI on the 9B's E2 judged set (15 arms on 6 prompts), one job per layer, and
# AllenAI-only tables per layer (strength 1; x base and the exploratory conf base). DeepSeek Pro is not run here: its layers
# are chosen afterwards from these tables (by hand, written down before the Pro pass), as for the other models.
# Four quarters of nine layers, each on a GPU as soon as the job that holds it now is over (DONE or FAIL):
#   layers 0-8   GPU 1  after the 09:38 chain's q8_run_C_b
#   layers 9-17  GPU 3  after the 09:38 chain's q8_run_E_b, chain b's AllenAI passes and chain d's AllenAI passes (red
#                       team 11:56: one of our GPU jobs at a time there; start chains b and d before this one)
#   layers 18-26 GPU 2  after the 09:38 chain's q8_judge_E and q8_screen_E
#   layers 27-35 GPU 0  after the 09:38 chain's q8_judge_C and q8_screen_C
# Runner: the copy next to this file (same commands as the 09:38 copy when RUN_DIR / SOURCE_DIR are unset). Logs and flags
# in logs/explore_c/; the other chains' flags are only read. Same mechanics as those chains.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}                     # overrides: dry runs only
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/explore_c
POLL=${CHAIN_POLL:-30}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain c holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME VAR=value ...: one runner call in a clean environment; skipped when NAME.DONE exists
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
wait_over() {  # wait_over DIR NAME...: until each NAME in DIR has DONE or FAIL (the GPU it held is free either way)
  local d=$1 f; shift
  for f in "$@"; do
    until [ -e "$d/$f.DONE" ] || [ -e "$d/$f.FAIL" ]; do sleep "$POLL"; done
  done
}

Q8=qwen3-8b
V=(MODEL=$Q8 VARIANT=E2 E_ON=E2)
E2_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 conf_E2.wq.m1 conf.iso.p1 caut.wq.p1 caut_E2.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_BASES="x x_F conf conf_E2 caut caut_E2"
E_EXTRA_A="x.rand.m1 x_F.w.p1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"

note "chain start (pid $$)"
job q8e2_seed "${V[@]}" STAGE=seed || exit 1

quarter() {  # quarter NAME GPU LAYERS: train + run arms, extra arms, then AllenAI and AllenAI tables, one layer at a time
  local q=$1 g=$2 L names="q8e2_extra_$1"
  for L in $3; do names="$names q8e2_allenai_L$L q8e2_tables_L$L"; done
  job q8e2_run_$q "${V[@]}" STAGE=run BRANCH=E GPU=$g LAYERS="$3" || { give_up $names; return 0; }
  job q8e2_extra_$q "${V[@]}" STAGE=extra BRANCH=E GPU=$g LAYERS="$3" ARMS="$E_EXTRA_A" || { give_up $names; return 0; }
  for L in $3; do
    { job q8e2_allenai_L$L "${V[@]}" STAGE=pro BRANCH=E LAYER=$L TAG=X JUDGES=allenai GPU=$g ARMS="$E2_PROA" BASES="$E2_BASES" &&
      job q8e2_tables_L$L "${V[@]}" STAGE=tables BRANCH=E LAYERS=$L TAG=X STRENGTH=1 JUDGES=allenai; } || give_up q8e2_tables_L$L
  done
}
LLB=""; for L in 8 9 10 11 12 13 14 15 16 17 18; do LLB="$LLB ll_e2_allenai_L$L"; done
{ wait_over "$LOGS/explore" q8_run_C_b;                   quarter a 1 "$(seq -s ' ' 0 8)"; } &
{ wait_over "$LOGS/explore" q8_run_E_b; wait_over "$LOGS/explore_b" $LLB; wait_over "$LOGS/explore_d" llc_allenai_L19 llc_allenai_L31
                                                          quarter b 3 "$(seq -s ' ' 9 17)"; } &
{ wait_over "$LOGS/explore" q8_judge_E q8_screen_E;       quarter c 2 "$(seq -s ' ' 18 26)"; } &
{ wait_over "$LOGS/explore" q8_judge_C q8_screen_C;       quarter d 0 "$(seq -s ' ' 27 35)"; } &
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

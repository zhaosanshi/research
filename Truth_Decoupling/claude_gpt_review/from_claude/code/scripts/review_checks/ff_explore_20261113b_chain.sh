#!/usr/bin/env bash
# Exploration chain b of 2026-11-13 (Claude; from_claude/44 section 5.5). Llama's E2 directions were trained on 11-11 (side
# branch results_e44_factorial_das/llama3.1-8b_side_E2, all 32 layers; from_claude/33 section 2.2) and judged by AllenAI only.
# Nothing is retrained: on layers 8-18 (the 9B's E2 layers) this chain generates the arms the 9B's E2 run has and Llama's
# lacks -- the must-answer base with the fixed write (conf.w.p1) and its mean-difference and random controls, plus the frozen
# criterion's x.rand.m1 and x_F.w.p1 -- then judges the 9B's E2 set (15 arms on 6 prompts) per layer, AllenAI and then
# DeepSeek Pro, and makes the tables (strength 1; x base and the exploratory conf base), exactly as for the 9B.
# Generation, export and judging run in the code copy that made Llama's run (code_e45_20261112; the core checks its identity
# against the run's frozen answers and directions) with that run's source directory (results_qa_das_v1u/llama3.1-8b/full;
# the export checks its hashes). The tables need e44_round1_tables.py, which only the 11-12 copies have (read-only; the
# 13b copy next to this file, so that nothing reads from the copy the 09:38 chain runs in).
# GPU 3 order (red team 11:56): the 09:38 chain's q8_run_E_b may still be generating when this chain starts (about 21 GB;
# the Llama generation adds about 19 GB, the other user's vLLM holds 5.3 of 73), then this chain, then chain d, then chain
# c's layers 9-17; chains c and d wait for this chain's AllenAI flags. Never restart this chain on GPU 3 while c or d hold
# it (CHAIN_GPU picks another card).
# The runner is the copy next to this file (RUN_DIR / SOURCE_DIR added 11:35); logs and flags in logs/explore_b/, apart
# from the 09:38 chain, which keeps running with its own runner. Same mechanics as that chain: clean environment per job,
# DONE flags skip, FAIL marks the rest of a group "not tried", one chain at a time (lock directory).
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}                     # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore_b}
POLL=${CHAIN_POLL:-30}
G=${CHAIN_GPU:-3}                                               # the GPU of the generation and the AllenAI passes
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain b holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME VAR=value ...: one runner call in a clean environment; skipped when NAME.DONE exists
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

LL=llama3.1-8b
RUN=$W/results_e44_factorial_das/llama3.1-8b_side_E2
SRCD=$W/results_qa_das_v1u/llama3.1-8b/full
CACHE=$W/results_e44_factorial_das/llama3.1-8b/judge_tune_step0/judge2/cache   # Llama's step-0 Pro cache (the base prompts)
OLD=$W/code_e45_20261112                                                      # the code copy of Llama's 11-11 run
NEW=$W/code_ff_explore_20261113b                                              # tables only
LAYERS_E2="8 9 10 11 12 13 14 15 16 17 18"
EXTRA="x.rand.m1 x_F.w.p1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 conf_E2.wq.m1 conf.iso.p1 caut.wq.p1 caut_E2.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_BASES="x x_F conf conf_E2 caut caut_E2"
V=(MODEL=$LL RUN_DIR=$RUN SOURCE_DIR=$SRCD E_ON=E2 BRANCH=E)

gpu_lane() {  # generation of every missing arm, then AllenAI layer by layer (one GPU)
  local L names=""
  for L in $LAYERS_E2; do names="$names ll_e2_allenai_L$L"; done
  job ll_e2_extra "${V[@]}" CODE_PATH=$OLD STAGE=extra GPU=$G LAYERS="$LAYERS_E2" ARMS="$EXTRA" || { give_up $names; return; }
  for L in $LAYERS_E2; do
    job ll_e2_allenai_L$L "${V[@]}" CODE_PATH=$OLD STAGE=pro GPU=$G LAYER=$L TAG=X JUDGES=allenai ARMS="$E2_PROA" BASES="$E2_BASES" || true
  done
}
pro_lane() {  # pro_lane LAYER...: DeepSeek Pro per layer once its AllenAI pass is done (no GPU), then that layer's tables
  local L
  for L in "$@"; do
    { wait_done ll_e2_allenai_L$L &&
      job ll_e2_pro_L$L "${V[@]}" CODE_PATH=$OLD STAGE=pro LAYER=$L TAG=X JUDGES=pro ARMS="$E2_PROA" BASES="$E2_BASES" CACHE_FROM=$CACHE &&
      job ll_e2_tables_L$L "${V[@]}" CODE_PATH=$NEW STAGE=tables LAYERS=$L TAG=X STRENGTH=1; } || give_up ll_e2_pro_L$L ll_e2_tables_L$L
  done
}

note "chain start (pid $$)"
gpu_lane &
pro_lane 8 12 16 & pro_lane 9 13 17 & pro_lane 10 14 18 & pro_lane 11 15 &
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

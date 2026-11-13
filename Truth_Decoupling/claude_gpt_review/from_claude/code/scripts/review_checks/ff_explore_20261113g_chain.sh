#!/usr/bin/env bash
# Exploration chain g of 2026-11-13 (Claude; from_claude/44 section 5.8, written before it ran). 赵磊 18:0x: "if the dose along
# the direction found from the prompt goes up, can it get more right than changing the prompt?" Only Llama's E1 direction
# was ever pushed past strength 1 on the must-answer base (+36 against the sentence's +21, from_claude/42 row 4). This chain
# does it for the E2 direction: strengths 2 and 3 on the must-answer base (conf), with the random directions (both signs) and
# the mean-difference direction of the same strength. Nothing is retrained.
#   Llama: the 11-11 E2 side branch (results_e44_factorial_das/llama3.1-8b_side_E2), layers 10 11, card 0; generation, export
#          and judging in the code copy that made that run (code_e45_20261112), as chain b.
#   9B:    the E2 variant run (VARIANT=E2), layers 11 14 15, card 1; code copy code_ff_explore_20261113b (same core).
# Judged set per layer (TAG=D, work dir judge_tune_proD_E_L<layer>): the 8 new arms plus the 4 strength-1 arms that already
# exist (so one work dir holds strengths 1 / 2 / 3), on the prompts conf and conf_E2 (the prompt row of the table). AllenAI
# on the generating card, then DeepSeek Pro (no GPU; three lanes; reply cache seeded from that layer's TAG=X work dir).
# Tables: e44_round1_tables.py cells on the conf base at strengths 1 / 2 / 3 (read-only, 13b copy) and the cross-model
# summary ff_explore_dose_table.py (logs/explore_g/dose_summary.md).
# Mechanics as chains b-f: clean environment per job, DONE flags skip, FAIL marks the rest of a group "not tried", one chain
# at a time (lock directory). Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}                     # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore_g}
POLL=${CHAIN_POLL:-30}
GL=${CHAIN_GPU_LLAMA:-0}
GQ=${CHAIN_GPU_9B:-1}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain g holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
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

OLD=$W/code_e45_20261112                       # the code copy of Llama's 11-11 run (its identity)
NEW=$W/code_ff_explore_20261113b               # the 9B's runs (same core as the copy that made them) and every table
LLRUN=$W/results_e44_factorial_das/llama3.1-8b_side_E2
Q9RUN=$W/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2
LL=(MODEL=llama3.1-8b RUN_DIR=$LLRUN SOURCE_DIR=$W/results_qa_das_v1u/llama3.1-8b/full E_ON=E2 BRANCH=E CODE_PATH=$OLD)
Q9=(MODEL=qwen3.5-9b VARIANT=E2 E_ON=E2 BRANCH=E CODE_PATH=$NEW)
LL_LAYERS="10 11"
Q9_LAYERS="11 14 15"
NEWARMS="conf.w.p2 conf.w.p3 conf.rand.p2 conf.rand.m2 conf.rand.p3 conf.rand.m3 conf.md.p2 conf.md.p3"
DOSE="conf.w.p1 conf.rand.p1 conf.rand.m1 conf.md.p1 $NEWARMS"
BASES="conf conf_E2"
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$NEW -w $NEW truth_dec python3)
tx() {  # the read-only table scripts in the 13b copy (CHAIN_TX_DRY=1: print the command only, for dry runs)
  if [ -n "${CHAIN_TX_DRY:-}" ]; then echo "tx $*"; else "${TX[@]}" "$@"; fi
}
for f in e44_round1_tables.py ff_explore_dose_table.py; do   # fail now, not after hours of generation and judging
  [ -n "${CHAIN_TX_DRY:-}" ] || [ -f "$R/${NEW#$W/}/scripts/review_checks/$f" ] || { echo "the 13b copy has no $f" >&2; exit 3; }
done

vars_of() {  # vars_of ll|q9: V = that model's runner variables, RUNV = its run directory
  case "$1" in
    ll) V=("${LL[@]}"); RUNV=$LLRUN ;;
    q9) V=("${Q9[@]}"); RUNV=$Q9RUN ;;
    *) echo "unknown model key $1" >&2; return 1 ;;
  esac
}
gpu_lane() {  # gpu_lane KEY GPU "LAYERS": generation of the new arms, then AllenAI layer by layer (one card)
  local key=$1 g=$2 layers=$3 L names=""
  vars_of "$key" || return 1
  for L in $layers; do names="$names ${key}_allenai_L$L"; done
  job ${key}_extra "${V[@]}" STAGE=extra GPU=$g LAYERS="$layers" ARMS="$NEWARMS" || { give_up $names; return; }
  for L in $layers; do
    job ${key}_allenai_L$L "${V[@]}" STAGE=pro GPU=$g LAYER=$L TAG=D JUDGES=allenai ARMS="$DOSE" BASES="$BASES" || true
  done
}
tables_job() {  # tables_job NAME RUN LAYER: conf base, strengths 1 2 3, both judges (read-only)
  local name=$1 run=$2 L=$3
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if ( for S in 1 2 3; do for J in allenai pro; do
         tx scripts/review_checks/e44_round1_tables.py cells "$run" E "$run/judge_tune_proD_E_L$L" \
           --judge "$J" --layers "$L" --base conf --strength "$S" || exit 1
       done; done ) > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.DONE"; note "done  $name"
  else
    echo "failed, see $name.log" > "$LOG/$name.FAIL"; note "FAIL  $name"; return 1
  fi
}
pro_lane() {  # pro_lane KEY:LAYER ...: for each, wait for its AllenAI pass, then DeepSeek Pro (no GPU), then its tables
  local item key L
  for item in "$@"; do
    key=${item%%:*}; L=${item#*:}
    vars_of "$key" || return 1
    { wait_done ${key}_allenai_L$L &&
      job ${key}_pro_L$L "${V[@]}" STAGE=pro LAYER=$L TAG=D JUDGES=pro ARMS="$DOSE" BASES="$BASES" \
        CACHE_FROM=$RUNV/judge_tune_proX_E_L$L/judge2/cache &&
      tables_job ${key}_tables_L$L "$RUNV" "$L"; } || give_up ${key}_pro_L$L ${key}_tables_L$L
  done
}
summary_job() {  # the cross-model table, once every layer's tables are done
  local f="" L
  for L in $LL_LAYERS; do f="$f ll_tables_L$L"; done
  for L in $Q9_LAYERS; do f="$f q9_tables_L$L"; done
  wait_done $f || { give_up dose_summary; return; }
  [ -e "$LOG/dose_summary.DONE" ] && return 0
  note "start dose_summary"
  local args=()
  for L in $LL_LAYERS; do args+=(--work "Llama:$L:$LLRUN/judge_tune_proD_E_L$L"); done
  for L in $Q9_LAYERS; do args+=(--work "9B:$L:$Q9RUN/judge_tune_proD_E_L$L"); done
  if tx scripts/review_checks/ff_explore_dose_table.py --e-on E2 "${args[@]}" > "$LOG/dose_summary.md" 2> "$LOG/dose_summary.log"; then
    touch "$LOG/dose_summary.DONE"; note "done  dose_summary"
  else
    echo "failed, see dose_summary.log" > "$LOG/dose_summary.FAIL"; note "FAIL  dose_summary"
  fi
}

note "chain start (pid $$): Llama layers [$LL_LAYERS] card $GL, 9B layers [$Q9_LAYERS] card $GQ"
( cd "$HERE" && sha256sum newmodel_e44.sh e44_round1_tables.py ff_explore_dose_table.py ff_explore_20261113g_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
gpu_lane ll "$GL" "$LL_LAYERS" &
gpu_lane q9 "$GQ" "$Q9_LAYERS" &
pro_lane ll:10 q9:14 & pro_lane ll:11 q9:15 & pro_lane q9:11 &
wait
summary_job
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

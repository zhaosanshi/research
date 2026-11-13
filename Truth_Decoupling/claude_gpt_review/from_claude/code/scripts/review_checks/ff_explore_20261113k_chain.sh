#!/usr/bin/env bash
# Exploration chain k of 2026-11-13 (Claude, 21:0x; 赵磊 20:2x "好，干吧"): the equal-N comparison of from_claude/39 for the
# 9B's E+C (chain h, TAG ec9b) through ff_explore_ec_select.sh: E2 layer 15 alone on the must-answer prompt (strengths 1, 2;
# reproduces the driver's files), judged by both judges (Pro cache from chain g's layer-15 work dir, which judged the same
# answers), the model's confidence in its own answers, then E+C against "keep the N most confident answers" at E+C's own N.
# Card 1, shared with chain j (one process each at a time). Mechanics as chains b-j.
# Order (red team 20:56): the confidence pass runs before any judging, so a failed check costs no judge calls. Written
# before any run: the confidence pass scores in batches of 8 (as for Llama); if a target misses the 98% greedy check
# there (the 9B's hybrid layers make padded batches differ slightly from single rows), it is redone once in batches of 1
# (no padding) with the same 98% threshold; if that fails too, the comparison is not reported.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
SEL=${CHAIN_SEL:-$HERE/ff_explore_ec_select.sh}                   # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore_k}
G=${CHAIN_GPU:-1}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain k holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u TABLE_CODE -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u ES -u KS -u TAG
       -u CACHE_FROM -u FF_ROOT -u SOURCE_DIR -u SCORE_BATCH)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if env "${CLEAN[@]}" "$@" bash "$SEL" > "$LOG/$name.log" 2>&1; then
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
FFE=$W/results_newmodels_ff_20261112/e44
Q9E=$FFE/qwen3.5-9b_E2
V=(MODEL=qwen3.5-9b CODE_PATH=$W/code_ff_explore_20261113b E_RUN=$Q9E C_RUN=$FFE/qwen3.5-9b E_ON=E2 C_E_ON=E1 EL=15 CL=16
   "ES=1 2" "KS=0.25 0.5 1 2" TAG=ec9b)
note "chain start (pid $$): card $G"
( cd "$HERE" && sha256sum ff_explore_ec_select.sh ff_explore_ec_confidence.py ff_explore_ec_compare.py ff_explore_20261113k_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
{ job genconf "${V[@]}" STAGE=genconf GPU=$G &&
  { job confidence "${V[@]}" STAGE=confidence GPU=$G ||
    { grep -q "teacher-forced arg-max reproduces only" "$LOG/confidence.log" &&
      job confidence_b1 "${V[@]}" STAGE=confidence GPU=$G SCORE_BATCH=1; }; } &&
  job judgeconf "${V[@]}" STAGE=judgeconf GPU=$G "CACHE_FROM=$Q9E/judge_tune_proD_E_L15/judge2/cache" &&
  job compare "${V[@]}" STAGE=compare; } || give_up confidence judgeconf compare
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

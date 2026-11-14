#!/usr/bin/env bash
# Stage-B equal-N comparison for the 27B (Claude, 2026-11-14 17:xx; from_claude/45 R6; as ff_test_select_20261114.sh did
# for Llama and the 9B): on the 529 test questions, the 27B's v1 E+C (chain p, TAG t27<CL>, C grid 0.25 0.5 1 2 all judged)
# against "E alone on the must-answer prompt, then keep the N answers the model itself is most confident in", at E+C's own N,
# both judges, through ff_explore_ec_select.sh SPLIT=test.
#   genconf     E alone on the must-answer prompt at strengths 1 2 through the engine; must reproduce chain p's driver test
#               files exactly (their judged answers stand in: CONF_WORK = the 27B's stage-B E work dir judge_test_testD_E_L<EL>)
#   confidence  the model's confidence in its own answers; EMPTY answers (the 27B sometimes stops at once: 3 in baseline x,
#               a few in the E+C files) are not scored and count as abstentions (from_claude/45 section 3.5); batch 8, and
#               once more in batches of 1 only if a target misses the 98% greedy check (decided before any run)
#   compare     once chain p's Pro passes of the E set (t27_ep) and of the E+C set (t27_ecp) are DONE
# Cards: CHAIN_GPU (default 2,3; the 27B config splits one model over the visible cards). Mechanics as chains b-s.
# Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
SEL=${CHAIN_SEL:-$HERE/ff_explore_ec_select.sh}                   # overrides: dry runs only
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/test_sel27
PLOG=$LOGS/explore_p
POLL=${CHAIN_POLL:-60}
G=${CHAIN_GPU:-2,3}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u TABLE_CODE -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u ES -u KS -u TAG
       -u CACHE_FROM -u FF_ROOT -u SOURCE_DIR -u SCORE_BATCH -u SPLIT -u CONF_WORK)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  [ -e "$LOG/$name.log" ] && mv -f "$LOG/$name.log" "$LOG/$name.log.prev"
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
upstream_done() {  # upstream_done NAME: 0 when chain p's NAME.DONE; 1 when FAIL, or when chain p has ended without it
  local f=$1
  until [ -e "$PLOG/$f.DONE" ]; do
    [ -e "$PLOG/$f.FAIL" ] && return 1
    if [ ! -d "$PLOG/.lock_chain" ]; then [ -e "$PLOG/$f.DONE" ] && return 0; return 1; fi
    sleep "$POLL"
  done
}
CODE=$W/code_ff_explore_20261113b
RUN=$W/results_newmodels_ff_20261112/e44/qwen3.8-27b-dm
SRC=$W/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
if [ -z "${CHAIN_SEL:-}" ]; then
  for f in ff_explore_ec_select.sh ff_explore_ec_confidence.py ff_test_confidence.py ff_explore_ec_compare.py e44_round1_tables.py e44_split_test.py; do
    cmp -s "$HERE/$f" "$R/${CODE#$W/}/scripts/review_checks/$f" || { echo "${CODE##*/}: $f missing or different" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
  done
fi
EL=$(sed -n 's/^chosen: //p' "$LOGS/explore_n/r3_choice.md" 2>/dev/null)
CL=$(sed -n 's/^r6: //p' "$PLOG/r5_r6.md" 2>/dev/null)
for v in "$EL" "$CL"; do
  case "$v" in ""|*[!0-9]*) echo "cannot read the R3 layer ($EL) or the R6 layer ($CL)" | tee -a "$LOG/chain_jobs.log" >&2; exit 3 ;; esac
done
V=(MODEL=qwen3.8-27b-dm CODE_PATH=$CODE SOURCE_DIR=$SRC E_RUN=$RUN C_RUN=$RUN E_ON=E2 C_E_ON=E2 EL=$EL CL=$CL
   "ES=1 2" "KS=0.25 0.5 1 2" SPLIT=test CONF_WORK=$RUN/judge_test_testD_E_L$EL TAG=t27$CL)
note "chain start (pid $$): 27B E$EL + C$CL v1 test (TAG t27$CL), cards $G"
( cd "$HERE" && sha256sum ff_explore_ec_select.sh ff_explore_ec_confidence.py ff_test_confidence.py ff_explore_ec_compare.py \
  e44_round1_tables.py ff_test_select27_20261114.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
{ upstream_done t27_egen && upstream_done t27_ec1 &&
  job genconf "${V[@]}" STAGE=genconf GPU=$G &&
  { [ -e "$LOG/confidence_b1.DONE" ] ||          # a finished batch-1 fallback is not redone at 8
    job confidence "${V[@]}" STAGE=confidence GPU=$G ||
    { grep -q "teacher-forced arg-max reproduces only" "$LOG/confidence.log" &&
      job confidence_b1 "${V[@]}" STAGE=confidence GPU=$G SCORE_BATCH=1; }; } &&
  upstream_done t27_ep && upstream_done t27_ecp &&
  job compare "${V[@]}" STAGE=compare; } || give_up genconf confidence compare
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

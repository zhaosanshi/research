#!/usr/bin/env bash
# Stage-B equal-N comparison of 2026-11-14 (Claude, 09:xx; from_claude/45 R6 "同作答数下比'E + 按模型自己的把握挑题'"; procedure
# of from_claude/39, as chain k ran it on the 9B's tune questions): on the 529 test questions, E+C against "E alone on the
# must-answer prompt, then keep the N answers the model itself is most confident in", at E+C's own N, both judges.
#   Llama: E2 L11 (side_E2 run), E+C with C19 (R6, TAG tll19) and C31 (secondary, TAG tll31) -- chain test_b's E+C runs.
#   9B:    E2 L15 (VARIANT E2 run), E+C with C16 (R6, TAG t9b16).
# Per model, through ff_explore_ec_select.sh SPLIT=test:
#   genconf     E alone on the must-answer prompt at strengths 1 2 through the engine; must reproduce chain test_b's
#               driver files of the same arms exactly (their judged answers stand in for these: CONF_WORK = the stage-B
#               E work dir judge_test_testD_E_L<EL>, which both judges judge in chain test_b)
#   confidence  per E+C C layer: the model's confidence in its own answers (batch 8; if a target misses the 98% greedy
#               check, once more in batches of 1 -- decided before any run, as in chain k)
#   compare     per E+C C layer, read-only, once chain test_b's Pro passes of the E set and of that E+C set are DONE
# Cards: Llama GPU_LL (default 1), 9B GPU_Q9 (default 3), shared with chain p's 27B jobs (72 GB cards; a 27B job holds
# about 32 GB on each of its two cards; ours about 20 GB). Mechanics as chains b-p: clean environment per job, DONE flags
# skip, FAIL marks the rest "not tried", one chain at a time (lock directory). Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
SEL=${CHAIN_SEL:-$HERE/ff_explore_ec_select.sh}                   # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/test_sel}
TB=${CHAIN_TB:-$R/results_newmodels_ff_20261112/logs/test_b}
POLL=${CHAIN_POLL:-60}
GL=${CHAIN_GPU_LLAMA:-1}
GQ=${CHAIN_GPU_9B:-3}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u TABLE_CODE -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u ES -u KS -u TAG
       -u CACHE_FROM -u FF_ROOT -u SOURCE_DIR -u SCORE_BATCH -u SPLIT -u CONF_WORK)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME VAR=value ...: one ff_explore_ec_select.sh call in a clean environment; skipped when NAME.DONE exists
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
upstream_done() {  # upstream_done NAME: 0 when chain test_b's NAME.DONE; 1 when FAIL, or when test_b has ended without it
  local f=$1
  until [ -e "$TB/$f.DONE" ]; do
    [ -e "$TB/$f.FAIL" ] && return 1
    if [ ! -d "$TB/.lock_chain" ]; then [ -e "$TB/$f.DONE" ] && return 0; return 1; fi
    sleep "$POLL"
  done
}

OLD=$W/code_e45_20261112
NEW=$W/code_ff_explore_20261113b
LLE=$W/results_e44_factorial_das/llama3.1-8b_side_E2
LLM=$W/results_e44_factorial_das/llama3.1-8b
Q9E=$W/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2
Q9=$W/results_newmodels_ff_20261112/e44/qwen3.5-9b
if [ -z "${CHAIN_SEL:-}" ]; then   # the scripts that run inside each code copy must be the ones next to this chain
  for c in "$OLD" "$NEW"; do
    for f in ff_explore_ec_confidence.py ff_test_confidence.py e44_split_test.py; do
      cmp -s "$HERE/$f" "$R/${c#$W/}/scripts/review_checks/$f" || { echo "${c##*/}: $f missing or different" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
    done
  done
  for f in ff_explore_ec_compare.py e44_round1_tables.py ff_explore_ec_select.sh; do
    cmp -s "$HERE/$f" "$R/${NEW#$W/}/scripts/review_checks/$f" || { echo "${NEW##*/}: $f missing or different" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
  done
fi
vars_of() {  # vars_of ll|q9: V (the selection runner's variables), PAIRS ("CL:TAG ..."), KEYP (prefix of test_b's job names)
  case "$1" in
    ll) V=(MODEL=llama3.1-8b CODE_PATH=$OLD TABLE_CODE=$NEW SOURCE_DIR=$W/results_qa_das_v1u/llama3.1-8b/full E_RUN=$LLE
           C_RUN=$LLM E_ON=E2 C_E_ON=E1 EL=11 "ES=1 2" "KS=0.25 0.5 1 2" SPLIT=test CONF_WORK=$LLE/judge_test_testD_E_L11)
        PAIRS="19:tll19 31:tll31" ;;
    q9) V=(MODEL=qwen3.5-9b CODE_PATH=$NEW SOURCE_DIR=$W/results_newmodels_ff_20261112/source/qwen3.5-9b E_RUN=$Q9E C_RUN=$Q9
           E_ON=E2 C_E_ON=E1 EL=15 "ES=1 2" "KS=0.25 0.5 1 2" SPLIT=test CONF_WORK=$Q9E/judge_test_testD_E_L15)
        PAIRS="16:t9b16" ;;
    *) echo "unknown model key $1" >&2; return 1 ;;
  esac
}
lane() {  # lane KEY GPU: genconf once, then per C layer the confidence pass (GPU) and, after the Pro passes, compare
  local k=$1 g=$2 p cl tag first names=""
  vars_of "$k" || return 1
  for p in $PAIRS; do names="$names ${k}_conf_${p#*:} ${k}_cmp_${p#*:}"; done
  first=${PAIRS%% *}
  # red team 10:14 item 1: the inputs of genconf / confidence come from chain test_b's GPU lane (driver test files of the E
  # arms, the choice file and gen1 files of each E+C run): wait for those jobs instead of assuming them
  { upstream_done ${k}_egen && upstream_done ${k}_ec1_C${first%%:*} &&
    job ${k}_genconf "${V[@]}" CL=${first%%:*} TAG=${first#*:} STAGE=genconf GPU=$g; } || { give_up $names; return 0; }
  for p in $PAIRS; do
    cl=${p%%:*}; tag=${p#*:}
    upstream_done ${k}_ec1_C$cl || { give_up ${k}_conf_$tag ${k}_cmp_$tag; continue; }
    [ -e "$LOG/${k}_conf1_$tag.DONE" ] && continue     # red team item 5: a finished batch-1 fallback is not redone at 8
    { job ${k}_conf_$tag "${V[@]}" CL=$cl TAG=$tag STAGE=confidence GPU=$g ||
      { grep -q "teacher-forced arg-max reproduces only" "$LOG/${k}_conf_$tag.log" &&
        job ${k}_conf1_$tag "${V[@]}" CL=$cl TAG=$tag STAGE=confidence GPU=$g SCORE_BATCH=1; }; } || give_up ${k}_cmp_$tag
  done
  for p in $PAIRS; do
    cl=${p%%:*}; tag=${p#*:}
    { [ -e "$LOG/${k}_conf_$tag.DONE" ] || [ -e "$LOG/${k}_conf1_$tag.DONE" ]; } || { give_up ${k}_cmp_$tag; continue; }
    { upstream_done ${k}_ep && upstream_done ${k}_ecp_C$cl &&
      job ${k}_cmp_$tag "${V[@]}" CL=$cl TAG=$tag STAGE=compare; } || give_up ${k}_cmp_$tag
  done
}

note "chain start (pid $$): Llama card $GL, 9B card $GQ"
( cd "$HERE" && sha256sum ff_explore_ec_select.sh ff_explore_ec_confidence.py ff_test_confidence.py e44_split_test.py \
  ff_explore_ec_compare.py e44_round1_tables.py ff_test_select_20261114.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
lane ll "$GL" &
lane q9 "$GQ" &
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

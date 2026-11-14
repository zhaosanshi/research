#!/usr/bin/env bash
# Chain s of 2026-11-14 (Claude, 11:1x): the equal-N comparison of from_claude/39 for the 27B's v1 E+C on the tune
# questions (chain p, TAG ec27<CL>, C grid 0.25 0.5 1 2 all judged), through ff_explore_ec_select.sh, exactly as chain k
# ran it for the 9B: E (the R3 layer) alone on the must-answer prompt at strengths 1, 2 (reproduces the driver's files of
# chain n), the model's confidence in its own answers, both judges (Pro cache from chain n's TAG=D work dir of that layer,
# which judged the same answers), then E+C against "keep the N most confident answers" at E+C's own N.
# Order as chain k (red team 11-13 20:56): the confidence pass before any judging; batch 8, and once more in batches of 1
# only if a target misses the 98% greedy check (decided before any run). The 27B's tune targets hold no empty answer
# (checked 11-14 11:1x); the confidence script stops loudly on one.
# Cards: CHAIN_GPU (default 2,3; the 27B config splits one model over the visible cards; chain r's Llama job shares card 2).
# Mechanics as chains b-r. Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
SEL=${CHAIN_SEL:-$HERE/ff_explore_ec_select.sh}                   # overrides: dry runs only
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/explore_s
G=${CHAIN_GPU:-2,3}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain s holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
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
CODE=$W/code_ff_explore_20261113b
RUN=$W/results_newmodels_ff_20261112/e44/qwen3.8-27b-dm
SRC=$W/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
EL=$(sed -n 's/^chosen: //p' "$LOGS/explore_n/r3_choice.md" 2>/dev/null)
CL=$(sed -n 's/^r6: //p' "$LOGS/explore_p/r5_r6.md" 2>/dev/null)
for v in "$EL" "$CL"; do
  case "$v" in ""|*[!0-9]*) echo "cannot read the R3 layer ($EL) or the R6 layer ($CL)" | tee -a "$LOG/chain_jobs.log" >&2; exit 3 ;; esac
done
for f in ec27${CL}_d; do [ -e "$LOGS/explore_p/$f.DONE" ] || { echo "chain p has no $f.DONE (the v1 tune E+C is not judged)" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }; done
V=(MODEL=qwen3.8-27b-dm CODE_PATH=$CODE SOURCE_DIR=$SRC E_RUN=$RUN C_RUN=$RUN E_ON=E2 C_E_ON=E2 EL=$EL CL=$CL
   "ES=1 2" "KS=0.25 0.5 1 2" TAG=ec27$CL)
note "chain start (pid $$): 27B E$EL + C$CL v1 tune (TAG ec27$CL), cards $G"
( cd "$HERE" && sha256sum ff_explore_ec_select.sh ff_explore_ec_confidence.py ff_explore_ec_compare.py ff_explore_20261114s_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
{ job genconf "${V[@]}" STAGE=genconf GPU=$G &&
  { [ -e "$LOG/confidence_b1.DONE" ] ||          # a finished batch-1 fallback is not redone at 8 (red team 11-14 10:14)
    job confidence "${V[@]}" STAGE=confidence GPU=$G ||
    { grep -q "teacher-forced arg-max reproduces only" "$LOG/confidence.log" &&
      job confidence_b1 "${V[@]}" STAGE=confidence GPU=$G SCORE_BATCH=1; }; } &&
  job judgeconf "${V[@]}" STAGE=judgeconf GPU=$G "CACHE_FROM=$RUN/judge_tune_proD_E_L$EL/judge2/cache" &&
  job compare "${V[@]}" STAGE=compare; } || give_up confidence judgeconf compare
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

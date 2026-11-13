#!/usr/bin/env bash
# Exploration chain i of 2026-11-13 (Claude; 赵磊 20:2x "好，干吧"): Llama's E+C with the E2 sentence, by the same rule as the
# 9B's chain h (from_claude/44 section 5.8), through ff_explore_ec.sh:
#   E = Llama's E2 direction of the 11-11 side branch (llama3.1-8b_side_E2), layer 11 (the best must-answer layer at
#       strength 1 under both judges together: Pro +29, AllenAI +29), strengths 1 and 2;
#   C = Llama's round-1 C direction (llama3.1-8b), toward confident, k = 0.25 / 0.5 / 1 / 2 chosen per E strength by rule
#       abstentions before any judging; two C layers, each its own TAG: 19 (the 9B's rule: the lowest C layer sent to Pro
#       above the E layer) and 31 (Llama's frozen C candidate, the layer of its E1+C in from_claude/37).
# Model steps and judging in the code copy of Llama's 11-11 run (code_e45_20261112: the core checks its identity); the
# read-only helpers from the 13b copy (TABLE_CODE). Card 0: generation of both TAGs one after the other, then both judge
# passes side by side (AllenAI loads one 7B judge at a time; DeepSeek Pro needs no card).
# Pro reply caches: layer 19 merges the E2 layer-11 work dir with the round-1 C layer-19 work dirs (a test merge found no
# conflicting key); layer 31 seeds from the E2 layer-11 work dir alone (every round-1 C layer-31 cache conflicts with it on
# one or two keys, and a conflicting key would stop the job).
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
EC=${CHAIN_EC:-$HERE/ff_explore_ec.sh}                           # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore_i}
G=${CHAIN_GPU:-0}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain i holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u TABLE_CODE -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u ES -u KS -u TAG
       -u CACHE_FROM -u FF_ROOT -u SOURCE_DIR)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME VAR=value ...: one ff_explore_ec.sh call in a clean environment; skipped when NAME.DONE exists
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if env "${CLEAN[@]}" "$@" bash "$EC" > "$LOG/$name.log" 2>&1; then
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

FA=$W/results_e44_factorial_das
LLE=$FA/llama3.1-8b_side_E2
LLM=$FA/llama3.1-8b
V=(MODEL=llama3.1-8b CODE_PATH=$W/code_e45_20261112 TABLE_CODE=$W/code_ff_explore_20261113b
   SOURCE_DIR=$W/results_qa_das_v1u/llama3.1-8b/full E_RUN=$LLE C_RUN=$LLM E_ON=E2 C_E_ON=E1 EL=11 "ES=1 2" "KS=0.25 0.5 1 2")
C19="$LLE/judge_tune_proX_E_L11/judge2/cache $LLM/judge_tune_proS_C_L19/judge2/cache $LLM/judge_tune_proB_C_L19/judge2/cache $LLM/judge_tune_proA_C_L19/judge2/cache"
C31="$LLE/judge_tune_proX_E_L11/judge2/cache"

note "chain start (pid $$): card $G"
( cd "$HERE" && sha256sum ff_explore_ec.sh ff_explore_ec_check.py ff_explore_ec_describe.py ff_explore_20261113i_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
gen_ok=1
{ job s19 "${V[@]}" CL=19 TAG=ecll19 STAGE=selfcheck GPU=$G &&
  job g1_19 "${V[@]}" CL=19 TAG=ecll19 STAGE=gen1 GPU=$G &&
  job g2_19 "${V[@]}" CL=19 TAG=ecll19 STAGE=gen2 GPU=$G; } || { give_up g1_19 g2_19 j19 d19; gen_ok=0; }
ok31=1
{ job s31 "${V[@]}" CL=31 TAG=ecll31 STAGE=selfcheck GPU=$G &&
  job g1_31 "${V[@]}" CL=31 TAG=ecll31 STAGE=gen1 GPU=$G &&
  job g2_31 "${V[@]}" CL=31 TAG=ecll31 STAGE=gen2 GPU=$G; } || { give_up g1_31 g2_31 j31 d31; ok31=0; }
if [ "$gen_ok" = 1 ]; then
  { job j19 "${V[@]}" CL=19 TAG=ecll19 STAGE=judge GPU=$G "CACHE_FROM=$C19" && job d19 "${V[@]}" CL=19 TAG=ecll19 STAGE=describe; } || give_up j19 d19 &
fi
if [ "$ok31" = 1 ]; then
  { job j31 "${V[@]}" CL=31 TAG=ecll31 STAGE=judge GPU=$G "CACHE_FROM=$C31" && job d31 "${V[@]}" CL=31 TAG=ecll31 STAGE=describe; } || give_up j31 d31 &
fi
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

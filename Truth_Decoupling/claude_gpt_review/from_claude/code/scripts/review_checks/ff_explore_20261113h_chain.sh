#!/usr/bin/env bash
# Exploration chain h of 2026-11-13 (Claude; from_claude/44 section 5.8, written before it ran). 赵磊 18:0x: on the plain
# prompt, where the model may say "I don't know", E alone makes it abstain and C has to rescue it -- the typical E+C case
# that the T/F 2x2 has no column for. Llama has it (from_claude/37, 39 section 6); this chain repeats the Llama procedure of
# from_claude/36 section 3 on the 9B through ff_explore_ec.sh:
#   E = the 9B's E2 direction, layer 15 (the best must-answer layer at strength 1 under both judges together: Pro +18,
#       AllenAI +18; layer 14 is +18 / +17), strengths 1 (the candidate rule's strength) and 2 (Llama's E+C strength);
#   C = the 9B's round-1 C direction, layer 16 (the lowest C layer sent to Pro above 15), toward confident,
#       k = 0.25 / 0.5 / 1 / 2, chosen per E strength by rule abstentions before any judging (ff_explore_ec_check.py).
# Card 0, shared with chain g's Llama lane: each lane runs one process at a time (a generation, or AllenAI, which loads one
# 7B judge at a time), so the card holds at most two processes, about 40 GB of 73 (red team 19:43).
# Pro reply caches: the 9B E2 layer-15 work dir (baseline x, the single E writes and their random controls) and the
# round-1 C layer-16 strength work dir (the single C writes), merged into the work dir's cache; the red team found no
# conflicting key between the two (the round-1 proA cache is left out: it holds two keys with two different replies).
# Mechanics as chains b-g: clean environment per job, DONE flags skip, FAIL marks the rest "not tried", one chain at a time.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
EC=${CHAIN_EC:-$HERE/ff_explore_ec.sh}                           # overrides: dry runs only
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/explore_h}
G=${CHAIN_GPU:-0}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain h holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u ES -u KS -u TAG -u CACHE_FROM
       -u FF_ROOT -u SOURCE_DIR)
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

FFE=$W/results_newmodels_ff_20261112/e44
Q9E=$FFE/qwen3.5-9b_E2
Q9=$FFE/qwen3.5-9b
V=(MODEL=qwen3.5-9b CODE_PATH=$W/code_ff_explore_20261113b E_RUN=$Q9E C_RUN=$Q9 E_ON=E2 C_E_ON=E1 EL=15 CL=16
   "ES=1 2" "KS=0.25 0.5 1 2" TAG=ec9b)
CACHES="$Q9E/judge_tune_proX_E_L15/judge2/cache $Q9/judge_tune_proS_C_L16/judge2/cache"

note "chain start (pid $$): card $G"
( cd "$HERE" && sha256sum ff_explore_ec.sh ff_explore_ec_check.py ff_explore_ec_describe.py e44_round1_tables.py judge2_cache_merge.py \
    ../e44_eval_extra.py ff_explore_20261113h_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true
{ job ec_selfcheck "${V[@]}" STAGE=selfcheck GPU=$G &&
  job ec_gen1 "${V[@]}" STAGE=gen1 GPU=$G &&
  job ec_gen2 "${V[@]}" STAGE=gen2 GPU=$G &&
  job ec_judge "${V[@]}" STAGE=judge GPU=$G "CACHE_FROM=$CACHES" &&
  job ec_describe "${V[@]}" STAGE=describe; } || give_up ec_gen1 ec_gen2 ec_judge ec_describe
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

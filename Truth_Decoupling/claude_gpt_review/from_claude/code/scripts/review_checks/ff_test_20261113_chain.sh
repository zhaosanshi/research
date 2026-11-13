#!/usr/bin/env bash
# Stage-B chain of 2026-11-13 (Claude, 23:xx; from_claude/45 section 2; 赵磊 20:4x "可以统一重验"): the rules v1 on the 529
# test questions for Llama (card GL) and the 9B (card GQ). One GPU lane per model: its GPU jobs one after another on that
# card, which it shares with the first half of a 27B lane (chains n / o) -- so never two of our jobs at once on a card.
# DeepSeek Pro runs in background lanes (no card) as soon as the AllenAI pass of its work dir is done; tables after Pro.
# Per model, GPU lane: freeze (11 prompts, 529 questions) -> E arms -> C arms -> AllenAI of the E set -> AllenAI of each C
# set -> for each E+C C layer: selfcheck, gen1 (+ reproduction check and the choice of k, written before any judging),
# gen2, AllenAI. A failed GPU job stops the rest of that model's GPU lane (the rest is marked "not tried").
#   Llama: the 11-11 runs (code copy code_e45_20261112: their identity; tables from code_ff_explore_20261113b); E2 at L11
#          (side_E2 run), C at 19 and 31 (round-1 run, E1); E+C with C19 (R6) and C31 (secondary, as on tune).
#   9B:    E2 at L15 (VARIANT E2 run), C at 12 16 17 (round-1 run, E1); E+C with C16 (R6).
# Runners next to this file: ff_test_eval.sh (driver steps, through e44_split_test.py) and ff_explore_ec.sh (SPLIT=test).
# Mechanics as chains b-o: clean environment per job, DONE flags skip, FAIL marks the rest of a group "not tried", one chain
# at a time (lock directory). Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
TE=${CHAIN_TEST_RUNNER:-$HERE/ff_test_eval.sh}                    # overrides: dry runs only
EC=${CHAIN_EC_RUNNER:-$HERE/ff_explore_ec.sh}
LOG=${CHAIN_LOG:-$R/results_newmodels_ff_20261112/logs/test_b}
POLL=${CHAIN_POLL:-30}
GL=${CHAIN_GPU_LLAMA:-0}
GQ=${CHAIN_GPU_9B:-2}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another stage-B chain holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u TABLE_CODE -u SOURCE_DIR -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u CLS
       -u ECLS -u ES -u KS -u TAG -u LAYER -u JUDGES -u SPLIT -u CACHE_FROM -u FF_ROOT)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME RUNNER VAR=value ...: one runner call in a clean environment; skipped when NAME.DONE exists
  local name=$1 runner=$2; shift 2
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if env "${CLEAN[@]}" "$@" bash "$runner" > "$LOG/$name.log" 2>&1; then
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

OLD=$W/code_e45_20261112
NEW=$W/code_ff_explore_20261113b
LLE=$W/results_e44_factorial_das/llama3.1-8b_side_E2
LLM=$W/results_e44_factorial_das/llama3.1-8b
Q9E=$W/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2
Q9=$W/results_newmodels_ff_20261112/e44/qwen3.5-9b
if [ -z "${CHAIN_TEST_RUNNER:-}" ]; then
  for c in "$OLD" "$NEW"; do                     # the entry script runs inside each model's code copy: the same file
    cmp -s "$HERE/e44_split_test.py" "$R/${c#$W/}/scripts/review_checks/e44_split_test.py" \
      || { echo "${c##*/}: e44_split_test.py missing or different from the one next to this chain" >&2; exit 3; }
  done
  for f in ff_explore_ec_check.py ff_explore_ec_describe.py e44_round1_tables.py ff_explore_dose_table.py; do   # table code copy
    cmp -s "$HERE/$f" "$R/${NEW#$W/}/scripts/review_checks/$f" || { echo "${NEW##*/}: $f missing or different" >&2; exit 3; }
  done
  grep -q 'SPLIT=${SPLIT:-tune}' "$EC" || { echo "$EC is not the SPLIT-aware E+C runner" >&2; exit 3; }
  # red team 23:21: the engine's selfcheck file name does not carry the split, so the test selfcheck replaces the tune
  # record of the same direction pair; every tune record must have a byte-identical backup before this chain runs
  python3 - "$R" "$LOG/selfcheck_tune_backup" <<'EOF' || exit 3
import filecmp, json, sys
from pathlib import Path
R, B = Path(sys.argv[1]), Path(sys.argv[2])
bad = []
for name, run in (("llama3.1-8b_side_E2", "results_e44_factorial_das/llama3.1-8b_side_E2"),
                  ("qwen3.5-9b_E2", "results_newmodels_ff_20261112/e44/qwen3.5-9b_E2")):
    for f in sorted((R / run / "eval_extra").glob("selfcheck_*.json")):
        if json.load(open(f)).get("split") == "tune":
            b = B / name / f.name
            if not b.exists() or not filecmp.cmp(str(f), str(b), shallow=False):
                bad.append(str(f))
sys.exit("tune selfcheck records without an identical backup in %s: %s" % (B, bad) if bad else 0)
EOF
fi
vars_of() {  # vars_of ll|q9: T (ff_test_eval.sh variables), X (ff_explore_ec.sh variables), CLS_OF, ECLS_OF, TAGP
  case "$1" in
    ll) T=(MODEL=llama3.1-8b CODE_PATH=$OLD TABLE_CODE=$NEW SOURCE_DIR=$W/results_qa_das_v1u/llama3.1-8b/full
           E_RUN=$LLE C_RUN=$LLM E_ON=E2 C_E_ON=E1 EL=11 "CLS=19 31" "ECLS=19 31" "ES=1 2" "KS=0.25 0.5 1 2")
        X=(MODEL=llama3.1-8b CODE_PATH=$OLD TABLE_CODE=$NEW SOURCE_DIR=$W/results_qa_das_v1u/llama3.1-8b/full
           E_RUN=$LLE C_RUN=$LLM E_ON=E2 C_E_ON=E1 EL=11 "ES=1 2" "KS=0.25 0.5 1 2" SPLIT=test)
        CLS_OF="19 31"; ECLS_OF="19 31"; TAGP=tll; CRUN_OF=$LLM ;;
    q9) T=(MODEL=qwen3.5-9b CODE_PATH=$NEW SOURCE_DIR=$W/results_newmodels_ff_20261112/source/qwen3.5-9b
           E_RUN=$Q9E C_RUN=$Q9 E_ON=E2 C_E_ON=E1 EL=15 "CLS=12 16 17" "ECLS=16" "ES=1 2" "KS=0.25 0.5 1 2")
        X=(MODEL=qwen3.5-9b CODE_PATH=$NEW SOURCE_DIR=$W/results_newmodels_ff_20261112/source/qwen3.5-9b
           E_RUN=$Q9E C_RUN=$Q9 E_ON=E2 C_E_ON=E1 EL=15 "ES=1 2" "KS=0.25 0.5 1 2" SPLIT=test)
        CLS_OF="12 16 17"; ECLS_OF="16"; TAGP=t9b; CRUN_OF=$Q9 ;;
    *) echo "unknown model key $1" >&2; return 1 ;;
  esac
}
gpu_names() {  # every GPU job name of a model, in lane order
  local k=$1 L
  vars_of "$k"
  echo -n "${k}_freeze ${k}_egen ${k}_cgen ${k}_ea"
  for L in $CLS_OF; do echo -n " ${k}_ca_L$L"; done
  for L in $ECLS_OF; do echo -n " ${k}_ecs_C$L ${k}_ec1_C$L ${k}_ec2_C$L ${k}_eca_C$L"; done
  echo
}
gpu_lane() {  # gpu_lane KEY GPU: the model's GPU jobs in order; the first failure gives up the rest of the lane
  local k=$1 g=$2 L names n
  vars_of "$k" || return 1
  names=$(gpu_names "$k")
  run_or_stop() { job "$@" || { give_up $names; return 1; }; }
  run_or_stop ${k}_freeze "$TE" "${T[@]}" STAGE=freeze GPU=$g || return 0
  run_or_stop ${k}_egen   "$TE" "${T[@]}" STAGE=egen GPU=$g || return 0
  run_or_stop ${k}_cgen   "$TE" "${T[@]}" STAGE=cgen GPU=$g || return 0
  run_or_stop ${k}_ea     "$TE" "${T[@]}" STAGE=ejudge JUDGES=allenai GPU=$g || return 0
  for L in $CLS_OF; do
    run_or_stop ${k}_ca_L$L "$TE" "${T[@]}" STAGE=cjudge LAYER=$L JUDGES=allenai GPU=$g || return 0
  done
  for L in $ECLS_OF; do
    run_or_stop ${k}_ecs_C$L "$EC" "${X[@]}" CL=$L TAG=$TAGP$L STAGE=selfcheck GPU=$g || return 0
    run_or_stop ${k}_ec1_C$L "$EC" "${X[@]}" CL=$L TAG=$TAGP$L STAGE=gen1 GPU=$g || return 0
    run_or_stop ${k}_ec2_C$L "$EC" "${X[@]}" CL=$L TAG=$TAGP$L STAGE=gen2 GPU=$g || return 0
    run_or_stop ${k}_eca_C$L "$EC" "${X[@]}" CL=$L TAG=$TAGP$L STAGE=judge JUDGES=allenai GPU=$g || return 0
  done
}
pro_lanes() {  # pro_lanes KEY: one background lane per judged set: wait for its AllenAI pass, DeepSeek Pro, then tables
  local k=$1 L
  vars_of "$k" || return 1
  { wait_done ${k}_ea && job ${k}_ep "$TE" "${T[@]}" STAGE=ejudge JUDGES=pro &&
    job ${k}_et "$TE" "${T[@]}" STAGE=etables; } || give_up ${k}_ep ${k}_et &
  for L in $CLS_OF; do
    { wait_done ${k}_ca_L$L && job ${k}_cp_L$L "$TE" "${T[@]}" STAGE=cjudge LAYER=$L JUDGES=pro &&
      job ${k}_ct_L$L "$TE" "${T[@]}" STAGE=ctables LAYER=$L; } || give_up ${k}_cp_L$L ${k}_ct_L$L &
  done
  for L in $ECLS_OF; do   # E+C Pro after the same layer's C set Pro: its reply cache (baseline x, the single C writes) is
                          # merged in, so identical cells are not paid twice (red team 23:21; conflicts stop the job)
    { wait_done ${k}_eca_C$L && wait_done ${k}_cp_L$L &&
      job ${k}_ecp_C$L "$EC" "${X[@]}" CL=$L TAG=$TAGP$L STAGE=judge JUDGES=pro \
        CACHE_FROM=$CRUN_OF/judge_test_testS_C_L$L/judge2/cache &&
      job ${k}_ecd_C$L "$EC" "${X[@]}" CL=$L TAG=$TAGP$L STAGE=describe; } || give_up ${k}_ecp_C$L ${k}_ecd_C$L &
  done
  wait
}

note "chain start (pid $$): Llama card $GL, 9B card $GQ"
( cd "$HERE" && sha256sum ff_test_eval.sh ff_explore_ec.sh e44_split_test.py ff_explore_ec_check.py ff_explore_ec_describe.py \
  ff_test_20261113_chain.sh; sha256sum "$R/${OLD#$W/}/scripts/review_checks/e44_split_test.py" "$R/${OLD#$W/}/scripts/e44_eval_extra.py" \
  "$R/${OLD#$W/}/scripts/e44_factorial_das.py" "$R/${NEW#$W/}/scripts/e44_eval_extra.py" "$R/${NEW#$W/}/scripts/e44_factorial_das.py" ) \
  >> "$LOG/chain_jobs.log" 2>&1 || true
gpu_lane ll "$GL" &
gpu_lane q9 "$GQ" &
pro_lanes ll &
pro_lanes q9 &
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

#!/usr/bin/env bash
# Chain p of 2026-11-14 (Claude, 01:xx, revised 02:xx after the red team of 01:51; from_claude/45 rules v1, 赵磊 11-13
# "跑"): the 27B after its rule-based layer choices, with no hand step in between.
#   R3    chain n's choice (explore_n/r3_choice.md, "chosen: L"). Without it there is nothing to do: the chain notes it and
#         ends (also when chain n has ended without writing it).
#   R5/R6 in the background, from chain o's screen and its tables at strength 0.25 (ff_rule_c_pick.py; written down in 45
#         before any 27B result; reproduces Llama 19 31 -> 19 and 9B 12 16 17 -> 16). If anything there fails or chain o
#         ends without it, R5 / R6 are "undetermined": the E half below still runs, C and E+C wait for a rerun.
#   test  stage B for the 27B on cards 2,3 once chain o has no GPU job left (ff_test_eval.sh through e44_split_test.py, as
#         Llama / 9B): freeze the 11 prompts on the 529 test questions, the E arms at the R3 layer, AllenAI of the E set;
#         then, once R5 / R6 are known, the C arms at every R5 layer and their AllenAI, and E+C under R6 (ff_explore_ec.sh
#         SPLIT=test) -- its selfcheck only after the tune E+C generation is over (the engine names a selfcheck file by the
#         two directions only, so the test selfcheck replaces the tune one).
#   tune  E+C under R6 on cards 0,1 once chain n has no GPU job left (ff_explore_ec.sh, TAG ec27<R6>); the tune selfcheck
#         record is copied to explore_p/selfcheck_tune_backup/ right after it is made.
#   Pro   background lanes, no card. Reply caches are seeded only from a conflict-free set (ff_cache_pick.py).
# Mechanics as chains n / o / test_b: clean environment per job, DONE flags skip, FAIL marks the rest "not tried", one chain
# p at a time (lock directory). Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
TE=${CHAIN_TEST_RUNNER:-$HERE/ff_test_eval.sh}                    # overrides: dry runs only
EC=${CHAIN_EC_RUNNER:-$HERE/ff_explore_ec.sh}
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/explore_p
NLOG=$LOGS/explore_n
OLOG=$LOGS/explore_o
POLL=${CHAIN_POLL:-60}
GT=${CHAIN_GPU_TUNE:-0,1}
GB=${CHAIN_GPU_TEST:-2,3}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain p holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u TABLE_CODE -u SOURCE_DIR -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u CLS
       -u ECLS -u ES -u KS -u TAG -u LAYER -u JUDGES -u SPLIT -u CACHE_FROM -u FF_ROOT)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME RUNNER VAR=value ...
  local name=$1 runner=$2; shift 2
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  [ -e "$LOG/$name.log" ] && mv -f "$LOG/$name.log" "$LOG/$name.log.prev"
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
wait_over() {  # wait_over NAME...: each NAME of this chain DONE or FAIL
  local f
  for f in "$@"; do
    until [ -e "$LOG/$f.DONE" ] || [ -e "$LOG/$f.FAIL" ]; do sleep "$POLL"; done
  done
}
upstream_done() {  # upstream_done DIR NAME: 0 when DIR/NAME.DONE; 1 when FAIL, or when that chain has ended without it
  local d=$1 f=$2
  until [ -e "$d/$f.DONE" ]; do
    [ -e "$d/$f.FAIL" ] && return 1
    if [ ! -d "$d/.lock_chain" ]; then [ -e "$d/$f.DONE" ] && return 0; return 1; fi
    sleep "$POLL"
  done
}
upstream_over() {  # upstream_over DIR NAME...: each DONE or FAIL, or that chain has ended (its cards are free either way)
  local d=$1 f; shift
  for f in "$@"; do
    until [ -e "$d/$f.DONE" ] || [ -e "$d/$f.FAIL" ] || [ ! -d "$d/.lock_chain" ]; do sleep "$POLL"; done
  done
}

CODE=$W/code_ff_explore_20261113b
RUN=$W/results_newmodels_ff_20261112/e44/qwen3.8-27b-dm
SRC=$W/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
for f in ff_test_eval.sh ff_explore_ec.sh e44_split_test.py ff_rule_c_pick.py ff_cache_pick.py ff_explore_ec_check.py ff_explore_ec_describe.py e44_round1_tables.py; do
  [ -n "${CHAIN_TEST_RUNNER:-}" ] || cmp -s "$HERE/$f" "$R/${CODE#$W/}/scripts/review_checks/$f" \
    || { echo "$f next to this chain is missing or differs from the 13b copy" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
done
[ -n "${CHAIN_TEST_RUNNER:-}" ] || grep -q 'SPLIT=${SPLIT:-tune}' "$EC" || { echo "$EC is not the SPLIT-aware E+C runner" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
# a selfcheck made by a lane whose generation did not finish is redone on a rerun (the other lane may have replaced it)
for p in "ec27 _s _g2" "t27 _ecs _ec2"; do
  set -- $p
  for f in "$LOG"/$1*$2.DONE; do
    [ -e "$f" ] || continue
    g=${f%$2.DONE}$3.DONE; [ "$1" = t27 ] && g=$LOG/t27$3.DONE
    [ -e "$g" ] || { rm -f "$f"; note "removed $(basename "$f"): its generation is not done, the selfcheck is redone"; }
  done
done

note "chain start (pid $$): tune E+C on cards $GT, test on cards $GB"
( cd "$HERE" && sha256sum ff_test_eval.sh ff_explore_ec.sh e44_split_test.py ff_rule_c_pick.py ff_cache_pick.py e44_round1_tables.py \
  ff_explore_20261114p_chain.sh ) >> "$LOG/chain_jobs.log" 2>&1 || true

# ---- R3 (chain n), read only
upstream_done "$NLOG" r3_choice || { note "chain n has no R3 choice (failed, or ended without one): nothing to do"; exit 0; }
EL=$(sed -n 's/^chosen: //p' "$NLOG/r3_choice.md")
case "$EL" in ""|*[!0-9]*) note "cannot read the R3 layer from $NLOG/r3_choice.md: stop"; exit 3 ;; esac
note "R3 E layer (chain n): $EL"

# ---- R5 / R6 (chain o), in the background; r5_r6.DONE or r5_r6.FAIL ("undetermined") always gets written
r5r6_step() {
  [ -e "$LOG/r5_r6.DONE" ] && return 0
  rm -f "$LOG/r5_r6.FAIL"
  local L short targs=()
  undetermined() { echo "$1" > "$LOG/r5_r6.FAIL"; note "R5 / R6 undetermined: $1 (E half goes on; C and E+C need a rerun)"; }
  upstream_done "$OLOG" q27c_screen || { undetermined "chain o has no screen"; return 0; }
  short=$(sed -n 's/^shortlist layers: *//p' "$OLOG/q27c_screen.log")
  if [ -z "$short" ]; then
    printf 'r5: none\nr6: none\n' > "$LOG/r5_r6.md"; touch "$LOG/r5_r6.DONE"; note "chain o shortlisted no C layer: R5 empty"; return 0
  fi
  for L in $short; do
    upstream_done "$OLOG" q27c_tables_L${L}_s0.25 || { undetermined "chain o's tables at 0.25 are missing for layer $L"; return 0; }
    targs+=(--table "$L:$OLOG/q27c_tables_L${L}_s0.25.log")
  done
  if python3 "$HERE/ff_rule_c_pick.py" --e-layer "$EL" "${targs[@]}" --out "$LOG/r5_r6.json" > "$LOG/r5_r6.md" 2> "$LOG/r5_r6.log"; then
    touch "$LOG/r5_r6.DONE"; note "R5 / R6: $(sed -n 's/^r5: //p' "$LOG/r5_r6.md") / $(sed -n 's/^r6: //p' "$LOG/r5_r6.md")"
  else
    undetermined "ff_rule_c_pick.py failed (see r5_r6.log)"
  fi
}
read_r5r6() {  # sets R5 (layers, may be empty) and R6 (layer or none) once r5r6_step is over
  until [ -e "$LOG/r5_r6.DONE" ] || [ -e "$LOG/r5_r6.FAIL" ]; do sleep "$POLL"; done
  if [ -e "$LOG/r5_r6.DONE" ]; then
    R5=$(sed -n 's/^r5: //p' "$LOG/r5_r6.md"); R6=$(sed -n 's/^r6: //p' "$LOG/r5_r6.md")
    [ "$R5" = none ] && R5=""
  else
    R5=""; R6=none
  fi
}
seed_from() {  # seed_from DIR...: the conflict-free cache dirs among DIR..., space separated (empty when none)
  "${TX[@]}" scripts/review_checks/ff_cache_pick.py "$@" 2>> "$LOG/cache_pick.log" | tee -a "$LOG/cache_pick.log" | sed -n 's/^use: *//p'
}
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec python3)
[ -n "${CHAIN_TEST_RUNNER:-}" ] && seed_from() { echo "$*"; }      # dry runs: no container
base_vars() {  # T0 / X: the runners' variables that do not depend on R5 / R6
  T0=(MODEL=qwen3.8-27b-dm CODE_PATH=$CODE SOURCE_DIR=$SRC E_RUN=$RUN C_RUN=$RUN E_ON=E2 C_E_ON=E2 EL=$EL "ES=1 2" "KS=0.25 0.5 1 2")
  X=("${T0[@]}")
}
base_vars

tune_lane() {  # E+C on the tune questions (R6), cards GT, once chain n has no GPU job left
  read_r5r6
  [ "$R6" = none ] && { note "no R6 layer (none above E$EL, or undetermined): no tune E+C"; return 0; }
  local L=$R6 t=ec27$R6 s backup
  local names="${t}_s ${t}_g1 ${t}_g2 ${t}_a"
  upstream_over "$NLOG" q27e_dextra $(for l in $(sed -n 's/^top: //p' "$NLOG/r3_top3.md"); do echo q27e_dallenai_L$l; done)
  job ${t}_s  "$EC" "${X[@]}" CL=$L TAG=$t STAGE=selfcheck GPU=$GT || { give_up $names; return 0; }
  backup=$LOG/selfcheck_tune_backup; mkdir -p "$backup"
  for s in "$R/${RUN#$W/}"/eval_extra/selfcheck_*.json; do [ -e "$s" ] && cp -p "$s" "$backup/"; done
  job ${t}_g1 "$EC" "${X[@]}" CL=$L TAG=$t STAGE=gen1 GPU=$GT || { give_up $names; return 0; }
  job ${t}_g2 "$EC" "${X[@]}" CL=$L TAG=$t STAGE=gen2 GPU=$GT || { give_up $names; return 0; }
  job ${t}_a  "$EC" "${X[@]}" CL=$L TAG=$t STAGE=judge JUDGES=allenai GPU=$GT || return 0
}
tune_pro() {
  read_r5r6
  [ "$R6" = none ] && return 0
  local L=$R6 t=ec27$R6 seeds
  wait_done ${t}_a || { give_up ${t}_p ${t}_d; return 0; }
  seeds=$(seed_from "$RUN/judge_tune_proS_C_L$L/judge2/cache" "$RUN/judge_tune_proX_E_L$EL/judge2/cache")
  [ -n "$seeds" ] || { note "no conflict-free Pro reply cache for $t (tune needs one): Pro not tried"; give_up ${t}_p ${t}_d; return 0; }
  note "$t Pro seeded from: $seeds"
  { job ${t}_p "$EC" "${X[@]}" CL=$L TAG=$t STAGE=judge JUDGES=pro "CACHE_FROM=$seeds" &&
    job ${t}_d "$EC" "${X[@]}" CL=$L TAG=$t STAGE=describe; } || give_up ${t}_p ${t}_d
}
test_lane() {  # stage B for the 27B, cards GB, once chain o has no GPU job left
  local L short cnames=""
  upstream_over "$OLOG" q27c_screen
  short=$(sed -n 's/^shortlist layers: *//p' "$OLOG/q27c_screen.log" 2>/dev/null)
  upstream_over "$OLOG" q27c_run_c $(for l in $short; do echo q27c_extra_L$l q27c_allenai_L$l; done)
  # the E half needs only R3
  job t27_freeze "$TE" "${T0[@]}" STAGE=freeze GPU=$GB || { give_up t27_egen t27_ea; return 0; }
  job t27_egen   "$TE" "${T0[@]}" STAGE=egen GPU=$GB || { give_up t27_ea; return 0; }
  job t27_ea     "$TE" "${T0[@]}" STAGE=ejudge JUDGES=allenai GPU=$GB || true
  # C and E+C once R5 / R6 are known
  read_r5r6
  local T=("${T0[@]}" "CLS=$R5"); [ "$R6" = none ] || T+=("ECLS=$R6")
  for L in $R5; do cnames="$cnames t27_ca_L$L"; done
  [ "$R6" = none ] || cnames="$cnames t27_ecs t27_ec1 t27_ec2 t27_eca"
  if [ -n "$R5" ]; then
    job t27_cgen "$TE" "${T[@]}" STAGE=cgen GPU=$GB || { give_up $cnames; return 0; }
    for L in $R5; do job t27_ca_L$L "$TE" "${T[@]}" STAGE=cjudge LAYER=$L JUDGES=allenai GPU=$GB || true; done
  fi
  [ "$R6" = none ] && return 0
  wait_over ec27${R6}_g2                      # the tune generation is over before the test selfcheck replaces its record
  job t27_ecs "$EC" "${X[@]}" CL=$R6 TAG=t27$R6 SPLIT=test STAGE=selfcheck GPU=$GB || { give_up t27_ec1 t27_ec2 t27_eca; return 0; }
  job t27_ec1 "$EC" "${X[@]}" CL=$R6 TAG=t27$R6 SPLIT=test STAGE=gen1 GPU=$GB || { give_up t27_ec2 t27_eca; return 0; }
  job t27_ec2 "$EC" "${X[@]}" CL=$R6 TAG=t27$R6 SPLIT=test STAGE=gen2 GPU=$GB || { give_up t27_eca; return 0; }
  job t27_eca "$EC" "${X[@]}" CL=$R6 TAG=t27$R6 SPLIT=test STAGE=judge JUDGES=allenai GPU=$GB || true
}
test_pro() {  # one background lane per judged test set
  local L
  { wait_done t27_ea && job t27_ep "$TE" "${T0[@]}" STAGE=ejudge JUDGES=pro && job t27_et "$TE" "${T0[@]}" STAGE=etables; } \
    || give_up t27_ep t27_et &
  read_r5r6
  local T=("${T0[@]}" "CLS=$R5"); [ "$R6" = none ] || T+=("ECLS=$R6")
  for L in $R5; do
    { wait_done t27_ca_L$L && job t27_cp_L$L "$TE" "${T[@]}" STAGE=cjudge LAYER=$L JUDGES=pro &&
      job t27_ct_L$L "$TE" "${T[@]}" STAGE=ctables LAYER=$L; } || give_up t27_cp_L$L t27_ct_L$L &
  done
  if [ "$R6" != none ]; then
    { local seeds
      if wait_done t27_eca && wait_done t27_cp_L$R6; then
        seeds=$(seed_from "$RUN/judge_test_testS_C_L$R6/judge2/cache")
        note "t27$R6 Pro seeded from: ${seeds:-nothing (conflicts)}"
        { job t27_ecp "$EC" "${X[@]}" CL=$R6 TAG=t27$R6 SPLIT=test STAGE=judge JUDGES=pro ${seeds:+"CACHE_FROM=$seeds"} &&
          job t27_ecd "$EC" "${X[@]}" CL=$R6 TAG=t27$R6 SPLIT=test STAGE=describe; } || give_up t27_ecp t27_ecd
      else
        give_up t27_ecp t27_ecd
      fi; } &
  fi
  wait
}

r5r6_step &
tune_lane &
tune_pro &
test_lane &
test_pro &
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"

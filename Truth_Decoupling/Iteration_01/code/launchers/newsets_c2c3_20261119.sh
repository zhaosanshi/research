#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-19): work queue C2 / D2 / C3 on TriviaQA and PopQA
# (novelty/ours/2026-11-19_new_datasets_pilot_design.md). Thin wrapper: generation only through
# scripts/newsets/newsets_gen.py with the frozen TruthfulQA setup (configs/newsets/frozen_models_20261119.json);
# judging only through the existing scripts/review_checks/judge_two_axis_run.py (rubric two_axis_v6, unchanged).
# Order (GPT review 2026-11-19, section 2, first row): the tune part runs and is judged on its own, the D2 predictions
# are registered, and only then the test part starts. The tune stages never touch a test path (checked on every
# command), and the test stages refuse to start without the D2 registration marker.
# usage (host side, one model per call):  [DRY=1] newsets_c2c3_20261119.sh STAGE MODEL [GPUS]
#   repro       one batch of the TruthfulQA tune split must reproduce the saved answers (six arms)              GPU
#   tune        tune, both tasks: the six conditions x caut conf conf_E2 conf_FR conf:E, the confidence of conf
#               (for D2), and the tune judge inputs (all 192 questions x the six conditions)                    GPU
#   tune_judge  DeepSeek Pro on the two tune work dirs                                                         no GPU
#   gap         D2: the gap on the tune split of both tasks (scripts/newsets/newsets_gap.py)                    GPU
#   test        needs Iteration_01/newsets/gap/D2_REGISTERED: tune ladder arms (for the extra row), test nine arms,
#               test confidences (with the first-difference check), test judge inputs (the frozen Pro 100)     GPU
#   test_judge  DeepSeek Pro on the two test work dirs                                                         no GPU
#   val         fresh questions (split test2, novelty/ours/2026-11-20_rechosen_caps_validation.md); needs
#               Iteration_01/newsets/gap/VAL_REGISTERED: x, E, E + C at the four strengths on test2, then the Pro 100
#               judge input with the three arms the re-chosen caps can use (x, E, E + C 0.25)                     GPU
#   val_judge   DeepSeek Pro on the two test2 work dirs                                                         no GPU
# Judging: a work dir starts only off-peak and at least 40 min before the next peak (Mon-Fri 09-12 and 14-18 Beijing);
# the judge process itself runs under `timeout` that ends 5 min before that peak (exit 3 = stopped at the deadline;
# rerun later, the judge resumes from its reply cache). A judged work dir is reused only after
# scripts/newsets/newsets_votes.py verifies its votes against the current generations.
set -uo pipefail
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
V=$CODE/scripts/newsets/newsets_votes.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
STAGE=${1:?stage}; MODEL=${2:?model}; GPUS=${3:-}; DRY=${DRY:-}
host() { echo "$HOST/${1#$ROOT/}"; }
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') [$STAGE $MODEL] $*"; }
fail() { say "FAILED: $*"; exit 1; }
case "$STAGE" in tune|tune_judge|gap) TUNE_ONLY=1 ;; *) TUNE_ONLY= ;; esac
run() {  # every docker / python command goes through here: the tune stages may not name a test path
  if [ -n "$TUNE_ONLY" ] && printf '%s ' "$@" | grep -Eq '/test(/|_pro100|\b)|--split test'; then fail "a tune stage tried to touch a test path: $*"; fi
  if [ -n "$DRY" ]; then echo "+ $*"; return 0; fi
  "$@"
}
read -r PY EL CL NG <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"], m["gpus"])' "$(host "$M")" "$MODEL")" || fail "model $MODEL not in the config"
case "$STAGE" in tune_judge|test_judge|val_judge) ;; *) [ -n "$GPUS" ] || fail "stage $STAGE needs GPUS";
  [ "$(echo "$GPUS" | tr ',' '\n' | grep -c .)" = "$NG" ] || fail "$MODEL needs $NG GPU(s), got $GPUS" ;; esac
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
E="E$EL.w.p2"; C="C$CL.w"
TUNE_ARMS="x caut conf conf_E2 conf_FR conf:$E"
LADDER="x:$E+$C.m0.25 x:$E+$C.m0.5 x:$E+$C.m1 x:$E+$C.m2"
TUNE_EXTRA="x:$E $LADDER x:$C.m0.25"
TEST_ARMS="x x:$E $LADDER x:$C.m0.25 conf conf:$E"
token() { echo "$1" | sed -e 's/:/--/' -e 's/+/-and-/' -e 's/\./-/g'; }
bankf() { echo "$NS/bank/$1_bank_20261119.jsonl"; }
gen() {  # gen TASK SPLIT ARMS...
  local task=$1 split=$2; shift 2
  run "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $task --items "$(bankf $task)" --split $split \
      --out $NS/gen/$task/$MODEL/$split --arms "$@" || fail "gen $task $split"
}
conf() {  # conf TASK SPLIT ARM [REFERENCE ARM]
  local task=$1 split=$2 arm=$3 ref=${4:-}
  local f=$NS/gen/$task/$MODEL/$split/$(token "$arm").json o=$NS/conf/$task/$MODEL/$split/$(token "$arm").json
  if [ -z "$DRY" ] && [ -e "$(host "$o")" ]; then say "skip $o (exists)"; return 0; fi
  # scoring batch 8 first; 1 only if that misses the 98% reproduction check (the rule of ff_explore_ec_confidence.py)
  run "${DX[@]}" $PY -I $G conf --models $M --model $MODEL --items "$(bankf $task)" --gen-file $f --out $o \
      ${ref:+--ref-gen-file $NS/gen/$task/$MODEL/$split/$(token "$ref").json} && return 0
  say "conf $task $split $arm missed a check at scoring batch 8: once more at scoring batch 1"
  run "${DX[@]}" $PY -I $G conf --models $M --model $MODEL --items "$(bankf $task)" --gen-file $f --out $o --score-batch 1 \
      ${ref:+--ref-gen-file $NS/gen/$task/$MODEL/$split/$(token "$ref").json} || fail "conf $task $split $arm"
}
export_() {  # export TASK SPLIT WORKNAME [--pro100] ARMS...
  local task=$1 split=$2 work=$3; shift 3
  local extra=""; if [ "${1:-}" = "--pro100" ]; then extra=--pro100; shift; fi
  local out=$NS/judge/$task/$MODEL/$work/judge_input.json
  if [ -z "$DRY" ] && [ -e "$(host "$out")" ]; then say "skip $out (exists; never overwritten)"; return 0; fi
  run "${DX[@]}" python3 -I $G export --task $task --model $MODEL --items "$(bankf $task)" --split $split $extra \
      --gen-dir $NS/gen/$task/$MODEL/$split --only-arms "$@" --out $out || fail "export $task $work"
}
seconds_left() {  # seconds until 5 min before the next peak window (0 inside a peak window)
  TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time())
if peak(now):
    print(0)
else:
    t = now
    while not peak(t) and t - now < 8 * 86400:
        t += 60
    print(max(0, t - now - 300))'
}
judge() {  # judge TASK WORKNAME SPLIT
  local task=$1 work=$2 split=$3
  local W=$NS/judge/$task/$MODEL/$work GD=$NS/gen/$task/$MODEL/$split
  local J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
           -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
           -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  if [ -z "$DRY" ] && [ -e "$(host "$W")/judge2/single_pro.jsonl" ]; then
    run "${J[@]}" python3 -I $V verify "$W" --gen-dir "$GD" || fail "$W holds votes that do not verify against the current generations"
    say "skip $W (judged, votes verified)"; return 0
  fi
  local secs; secs=$(seconds_left)
  [ "$secs" -ge 2400 ] || { say "STOP: off-peak time left before the next peak is ${secs}s (< 40 min); not starting $W"; exit 3; }
  if [ -n "$DRY" ] || [ ! -e "$(host "$W")/judge2/cells.jsonl" ]; then
    run "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells "$W" --generation judge_input.json || fail "cells $W"
  fi
  say "pro on $W for at most ${secs}s"
  run "${J[@]}" timeout -k 60 --signal=INT "$secs" python3 scripts/review_checks/judge_two_axis_run.py llm "$W" --judge pro
  local rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then say "STOP: reached the deadline before the next peak; rerun later (the reply cache keeps finished requests)"; exit 3; fi
  [ "$rc" = 0 ] || fail "pro $W (exit $rc)"
  run "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py single "$W" --judge pro || fail "single $W"
  run "${J[@]}" python3 -I $V verify "$W" --gen-dir "$GD" || fail "the new votes of $W do not verify"
  say "judged $W"
}
case "$STAGE" in
  repro)
    run "${DX[@]}" $PY -I $G repro --models $M --model $MODEL --first-batches 1 --out $NS/repro/${MODEL}_b1_v2.json || fail repro ;;
  tune)
    for t in triviaqa popqa; do
      gen $t tune $TUNE_ARMS
      conf $t tune conf
      export_ $t tune tune $TUNE_ARMS
    done ;;
  tune_judge)
    for t in triviaqa popqa; do judge $t tune tune; done ;;
  gap)
    for t in triviaqa popqa; do
      run "${DX[@]}" $PY -I $CODE/scripts/newsets/newsets_gap.py --models $M --model $MODEL --task $t --items "$(bankf $t)" \
          --pro-work $NS/judge/$t/$MODEL/tune --gen-dir $NS/gen/$t/$MODEL/tune \
          --conf-file $NS/conf/$t/$MODEL/tune/conf.json --out $NS/gap/$t/$MODEL/tune_gap.json || fail "gap $t"
    done ;;
  test)
    [ -n "$DRY" ] || [ -s "$(host "$NS")/gap/D2_REGISTERED" ] || fail "the D2 predictions are not registered yet ($NS/gap/D2_REGISTERED)"
    for t in triviaqa popqa; do
      gen $t tune $TUNE_EXTRA
      gen $t test $TEST_ARMS
      conf $t test "conf:$E" conf
      conf $t test conf
      # the single-C confidence is not computed: it missed the 98% reproduction check for two Qwen models even at
      # scoring batch 1 (2026-11-19 20:5x), and neither the gate nor the strong form uses it
      export_ $t test test_pro100 --pro100 $TEST_ARMS
    done ;;
  test_judge)
    [ -n "$DRY" ] || [ -s "$(host "$NS")/gap/D2_REGISTERED" ] || fail "the D2 predictions are not registered yet"
    for t in triviaqa popqa; do judge $t test_pro100 test; done ;;
  val)
    [ -n "$DRY" ] || [ -s "$(host "$NS")/gap/VAL_REGISTERED" ] || fail "the validation predictions are not registered yet"
    for t in triviaqa popqa; do
      B2=$NS/bank/${t}_bank_test2_20261120.jsonl
      run "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $B2 --split test2 \
          --out $NS/gen/$t/$MODEL/test2 --arms x "x:$E" $LADDER || fail "gen $t test2"
      out=$NS/judge/$t/$MODEL/test2_pro100/judge_input.json
      if [ -z "$DRY" ] && [ -e "$(host "$out")" ]; then say "skip $out (exists)"; else
        run "${DX[@]}" python3 -I $G export --task $t --model $MODEL --items $B2 --split test2 --pro100 \
            --gen-dir $NS/gen/$t/$MODEL/test2 --only-arms x "x:$E" "x:$E+$C.m0.25" --out $out || fail "export $t test2"
      fi
    done ;;
  val_judge)
    [ -n "$DRY" ] || [ -s "$(host "$NS")/gap/VAL_REGISTERED" ] || fail "the validation predictions are not registered yet"
    for t in triviaqa popqa; do judge $t test2_pro100 test2; done ;;
  *) fail "unknown stage $STAGE" ;;
esac
say "done"

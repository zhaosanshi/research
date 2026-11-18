#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-19): work queue C2 / C3 on TriviaQA and PopQA
# (novelty/ours/2026-11-19_new_datasets_pilot_design.md). Thin wrapper: generation only through
# scripts/newsets/newsets_gen.py with the frozen TruthfulQA setup (configs/newsets/frozen_models_20261119.json);
# judging only through the existing scripts/review_checks/judge_two_axis_run.py (rubric two_axis_v6, unchanged).
# usage (host side, one model per call):  newsets_c2c3_20261119.sh STAGE MODEL [GPUS]
#   repro   one batch of the TruthfulQA tune split must reproduce the saved answers (3 arms)       GPU
#   c2      tune, no write: x caut conf conf_E2 conf_FR                                         GPU
#   c3tune  tune: E, E + C at 0.25 / 0.5 / 1 / 2, single C, conf + E                            GPU
#   c3test  test: x, E, E + C at the four strengths, single C, conf, conf + E                   GPU
#   conf    mean log-prob of the own answers of conf + E, conf, single C (tune and test)        GPU
#   export  judge inputs: tune = the five C2 prompts + conf:E (all 192); test = the nine arms (Pro 100)
#   gap     D2: the gap on the tune split (scripts/newsets/newsets_gap.py), after the tune work dir is judged     GPU
#   judge   cells + DeepSeek Pro + single over this model's four work dirs; refuses to start a work dir
#           in, or less than 40 min before, a peak window (Mon-Fri 09-12 and 14-18 Beijing)          no GPU
set -uo pipefail
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
STAGE=${1:?stage}; MODEL=${2:?model}; GPUS=${3:-}
host() { echo "$HOST/${1#$ROOT/}"; }
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') [$STAGE $MODEL] $*"; }
fail() { say "FAILED: $*"; exit 1; }
read -r PY EL CL NG <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"], m["gpus"])' "$(host "$M")" "$MODEL")" || fail "model $MODEL not in the config"
case "$STAGE" in export|judge) ;; *) [ -n "$GPUS" ] || fail "stage $STAGE needs GPUS";
  [ "$(echo "$GPUS" | tr ',' '\n' | grep -c .)" = "$NG" ] || fail "$MODEL needs $NG GPU(s), got $GPUS" ;; esac
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
E="E$EL.w.p2"; C="C$CL.w"
C2_ARMS="x caut conf conf_E2 conf_FR"
LADDER="x:$E+$C.m0.25 x:$E+$C.m0.5 x:$E+$C.m1 x:$E+$C.m2"
C3_TUNE="x:$E $LADDER x:$C.m0.25 conf:$E"
C3_TEST="x x:$E $LADDER x:$C.m0.25 conf conf:$E"
TUNE_PRO="$C2_ARMS conf:$E"
token() { echo "$1" | sed -e 's/:/--/' -e 's/+/-and-/' -e 's/\./-/g'; }
gen() {  # gen TASK SPLIT ARMS...
  local task=$1 split=$2; shift 2
  "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $task --items $NS/bank/${task}_bank_20261119.jsonl \
      --split $split --out $NS/gen/$task/$MODEL/$split --arms "$@" || fail "gen $task $split"
}
case "$STAGE" in
  repro)
    "${DX[@]}" $PY -I $G repro --models $M --model $MODEL --first-batches 1 --out $NS/repro/${MODEL}_b1_launcher.json || fail repro ;;
  c2)     for t in triviaqa popqa; do gen $t tune $C2_ARMS; done ;;
  c3tune) for t in triviaqa popqa; do gen $t tune $C3_TUNE; done ;;
  c3test) for t in triviaqa popqa; do gen $t test $C3_TEST; done ;;
  conf)
    for t in triviaqa popqa; do for s in tune test; do for a in "conf:$E" conf "x:$C.m0.25"; do
      f=$NS/gen/$t/$MODEL/$s/$(token "$a").json
      o=$NS/conf/$t/$MODEL/$s/$(token "$a").json
      [ -e "$(host "$o")" ] && { say "skip $o"; continue; }
      "${DX[@]}" $PY -I $G conf --models $M --model $MODEL --items $NS/bank/${t}_bank_20261119.jsonl --gen-file $f --out $o \
          || fail "conf $t $s $a"
    done; done; done ;;
  export)
    for t in triviaqa popqa; do
      "${DX[@]}" python3 -I $G export --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split tune \
          --gen-dir $NS/gen/$t/$MODEL/tune --only-arms $TUNE_PRO --out $NS/judge/$t/$MODEL/tune/judge_input.json || fail "export $t tune"
      "${DX[@]}" python3 -I $G export --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split test --pro100 \
          --gen-dir $NS/gen/$t/$MODEL/test --only-arms $C3_TEST --out $NS/judge/$t/$MODEL/test_pro100/judge_input.json || fail "export $t test"
    done ;;
  gap)
    for t in triviaqa popqa; do
      "${DX[@]}" $PY -I $CODE/scripts/newsets/newsets_gap.py --models $M --model $MODEL --task $t \
          --items $NS/bank/${t}_bank_20261119.jsonl --pro-work $NS/judge/$t/$MODEL/tune \
          --conf-file $NS/conf/$t/$MODEL/tune/conf.json --out $NS/gap/$t/$MODEL/tune_gap.json || fail "gap $t"
    done ;;
  judge)
    allowed() {  # 0 if now is off-peak and the next peak is at least 40 min away (Beijing time)
      TZ=Asia/Shanghai python3 -c '
import sys, time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = time.time()
sys.exit(0 if not any(peak(now + k * 60) for k in range(0, 41)) else 1)'
    }
    J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
       -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
       -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec python3)
    for t in triviaqa popqa; do for w in tune test_pro100; do
      W=$NS/judge/$t/$MODEL/$w
      [ -e "$(host "$W")/judge2/single_pro.jsonl" ] && { say "skip $W (judged)"; continue; }
      allowed || fail "peak window now or within 40 min: not starting $W (rerun later; the reply cache keeps finished cells)"
      [ -e "$(host "$W")/judge2/cells.jsonl" ] || "${J[@]}" scripts/review_checks/judge_two_axis_run.py cells "$W" --generation judge_input.json \
          > "$(host "$W")/cells_stdout.json" || fail "cells $W"
      "${J[@]}" scripts/review_checks/judge_two_axis_run.py llm "$W" --judge pro || fail "pro $W"
      "${J[@]}" scripts/review_checks/judge_two_axis_run.py single "$W" --judge pro > "$(host "$W")/single_pro_stdout.md" || fail "single $W"
      say "judged $W"
    done; done ;;
  *) fail "unknown stage $STAGE" ;;
esac
say "done"

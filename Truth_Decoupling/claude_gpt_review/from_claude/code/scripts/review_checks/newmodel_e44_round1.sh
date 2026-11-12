#!/usr/bin/env bash
# from_claude/43, round 1 of the free-form E44 for one model on four GPUs (Claude, 2026-11-12). Host side; every step is a
# STAGE of newmodel_e44.sh (fixed copy code_newmodels_ff_20261112). Same layout as Llama's e44_round1_all.sh wave 1:
#   gpu0: E layers 0 .. L/2-1     gpu1: E layers L/2 .. L-1     gpu2: C layers 0 .. L/2-1     gpu3: C layers L/2 .. L-1
#   then AllenAI judging of E (gpu0) and of C (gpu1) once both halves of the branch are done, then the frozen screens (CPU).
# Pro judging, the extra arms and the strength sweeps follow the routing rule of 43 section 5 and are started after the
# screens have been read (they depend on which layers the screens list).
#   usage: newmodel_e44_round1.sh MODEL          (flags MODEL_<step>.DONE / .FAIL and logs in results_newmodels_ff_20261112/logs)
set -u
MODEL=${1:?model}
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$HOST/code_newmodels_ff_20261112
LOG=$HOST/results_newmodels_ff_20261112/logs
RUNNER=$CODE/scripts/review_checks/newmodel_e44.sh
mkdir -p "$LOG"
mkdir "$LOG/.lock_round1_$MODEL" 2>/dev/null || { echo "round 1 of $MODEL already started" >&2; exit 3; }
NL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['num_layers'])" "$CODE/configs/models/$MODEL.json")
H=$((NL / 2))
step() {  # step NAME ENV...  (runs newmodel_e44.sh with the given environment)
  local name=$1; shift
  echo "== $(date '+%F %T') start $name" >> "$LOG/round1_$MODEL.chain.log"
  if env MODEL="$MODEL" "$@" bash "$RUNNER" > "$LOG/${MODEL}_$name.log" 2>&1; then
    echo "== $(date '+%F %T') done $name" >> "$LOG/round1_$MODEL.chain.log"; touch "$LOG/${MODEL}_$name.DONE"
  else
    echo "== $(date '+%F %T') FAILED $name" >> "$LOG/round1_$MODEL.chain.log"; touch "$LOG/${MODEL}_$name.FAIL"; return 1
  fi
}
branch() {  # branch E|C GPU_LOW GPU_HIGH GPU_JUDGE
  local B=$1
  step run_${B}_a STAGE=run BRANCH=$B GPU=$2 LAYERS="$(seq 0 $((H - 1)) | tr '\n' ' ')" & local p1=$!
  step run_${B}_b STAGE=run BRANCH=$B GPU=$3 LAYERS="$(seq $H $((NL - 1)) | tr '\n' ' ')" & local p2=$!
  wait $p1; local r1=$?; wait $p2; local r2=$?
  [ $r1 = 0 ] && [ $r2 = 0 ] || { echo "== $(date '+%F %T') $B: a run half failed; no judging" >> "$LOG/round1_$MODEL.chain.log"; return 1; }
  step judge_$B STAGE=judge BRANCH=$B GPU=$4 || return 1
  step screen_$B STAGE=screen BRANCH=$B || return 1
}
branch E 0 1 0 &
branch C 2 3 1 &
wait
echo "== $(date '+%F %T') round 1 waves finished" >> "$LOG/round1_$MODEL.chain.log"

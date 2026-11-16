#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-16): runs scripts/review_checks/ff_ec_probe.py (赵磊 11-16 plan, item 6: does the
# E write move the model along the C direction?) for ONE model, with the model's own interpreter, in the container
# environment of newmodel_e44.sh. Read-only on the run; writes RUN/eval_extra/probe_c_under_e_<NAME>.{json,md}.
#   usage (host):
#     MODEL=gemma4-12b GPU=2 FF_ROOT=/workspace/Truth_Decoupling/Iteration_01/results/ff/gemma4/12b EL=24 CLS="25 27" \
#     GRID=ec21 NAME=g4plain [ES="1 2"] [FIRST_BATCHES=1] bash ff_ec_probe.sh
#   GRID names the judged tune grid RUN/eval_extra/judge_tune_<GRID>_tune whose rule-detector flags group the questions.
# One line per call in <FF_ROOT on the host>/logs/probe/chain_jobs.log, the output in logs/probe/<NAME>.log.
set -uo pipefail
: "${MODEL:?}" "${GPU:?}" "${FF_ROOT:?}" "${EL:?}" "${CLS:?}" "${GRID:?}" "${NAME:?}"
ES=${ES:-1 2}
FIRST_BATCHES=${FIRST_BATCHES:-0}
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
CODE=$W/Iteration_01/code
MODELS=$W/models
case "$FF_ROOT" in "$W"/Iteration_01/results/*) ;; *) echo "FF_ROOT must be under $W/Iteration_01/results" >&2; exit 2 ;; esac
case "$NAME$GRID" in *[!A-Za-z0-9_]*) echo "NAME / GRID: letters, digits and _ only" >&2; exit 2 ;; esac
case "$GPU" in ""|*[!0-9]*) echo "GPU: one GPU index" >&2; exit 2 ;; esac
case "$FIRST_BATCHES" in ""|*[!0-9]*) echo "FIRST_BATCHES: a number" >&2; exit 2 ;; esac
RUN=$FF_ROOT/e44/$MODEL
SRC=$FF_ROOT/source/$MODEL
HRUN=$R/${RUN#"$W"/}
[ -f "$HRUN/frozen_tune.json" ] || { echo "no run $RUN" >&2; exit 3; }
[ -f "$R/Iteration_01/code/configs/models/$MODEL.json" ] || { echo "no model config $MODEL" >&2; exit 3; }
[ -f "$HRUN/eval_extra/judge_tune_${GRID}_tune/judge_input.json" ] || { echo "no tune grid export judge_tune_${GRID}_tune/judge_input.json" >&2; exit 3; }
[ -f "$R/Iteration_01/code/scripts/review_checks/ff_ec_probe.py" ] || { echo "no scripts/review_checks/ff_ec_probe.py" >&2; exit 3; }
DX=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$MODELS
    -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES="$GPU" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
PYM=$("${DX[@]}" python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$CODE/configs/models/$MODEL.json") \
  || { echo "cannot read the interpreter of $MODEL" >&2; exit 3; }
LOG=$R/${FF_ROOT#"$W"/}/logs/probe
mkdir -p "$LOG"
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
note "start $NAME (pid $$): model $MODEL, GPU $GPU, E$EL, C [$CLS], E strengths [$ES], grid $GRID, first batches $FIRST_BATCHES, interpreter $PYM"
sha256sum "$0" "$R/Iteration_01/code/scripts/review_checks/ff_ec_probe.py" >> "$LOG/chain_jobs.log" 2>&1 || true
[ -e "$LOG/$NAME.log" ] && mv -f "$LOG/$NAME.log" "$LOG/$NAME.log.prev"
# shellcheck disable=SC2086  # CLS and ES are word lists on purpose
if "${DX[@]}" "$PYM" scripts/review_checks/ff_ec_probe.py "$NAME" --source-run "$SRC" --run-dir "$RUN" --e-on E2 \
     --el "$EL" --cls $CLS --es $ES --grid "$RUN/eval_extra/judge_tune_${GRID}_tune" --first-batches "$FIRST_BATCHES" \
     > "$LOG/$NAME.log" 2>&1 < /dev/null; then
  note "done  $NAME"
else
  note "FAIL  $NAME (see $NAME.log)"; exit 1
fi

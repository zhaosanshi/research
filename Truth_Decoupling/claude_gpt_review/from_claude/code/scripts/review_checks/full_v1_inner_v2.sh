#!/usr/bin/env bash
# Runs INSIDE truth_dec: the whole QA-DAS v1 chain for one model, every phase skipped when its artifact is already
# complete (so it can be launched after an earlier partial chain), stopping after STOP_AFTER (default: everything).
# usage: full_v1_inner_v2.sh MODEL
# env: TD_RESULTS_DIR TD_DATA_DIR TD_MODEL_ROOT CUDA_VISIBLE_DEVICES DEEPSEEK_API_KEY_FILE STOP_AFTER
set -u
MODEL=$1
ROOT=/workspace/Truth_Decoupling
RES=${TD_RESULTS_DIR:-$ROOT/results_qa_das_v1}
RUN=$RES/$MODEL/full_v1
STOP=${STOP_AFTER:-judge-diagnostics}
mkdir -p "$RES/$MODEL"
cd "$ROOT/review_e25/claude_gpt_review/from_claude/code" || exit 1
is_done() {   # is_done exists:FILE | completed:FILE   (completed = JSON with "completed": true)
  local kind=${1%%:*} file=${1#*:}
  [ -f "$file" ] || return 1
  [ "$kind" = exists ] && return 0
  python3 - "$file" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("completed") else 1)
PY
}
phase() {   # phase NAME CHECK [extra args]
  local name=$1 check=$2; shift 2
  if is_done "$check"; then
    echo "== $(date '+%F %T') $name $MODEL: already complete (${check#*:}), skipped"
  else
    echo "== $(date '+%F %T') $name $MODEL (GPU ${CUDA_VISIBLE_DEVICES:-?})"
    python3 scripts/e25_qa_das.py "$name" --model "$MODEL" --run-dir "$RUN" "$@" || { echo "== $(date '+%F %T') PHASE FAILED: $name $MODEL"; exit 1; }
  fi
  if [ "$name" = "$STOP" ]; then echo "== $(date '+%F %T') STOPPED AFTER $name $MODEL (as requested)"; exit 0; fi
}
phase estimate          "exists:$RUN/bank.json" --all-layers
phase train             "completed:$RUN/training.json"
phase tune-e            "exists:$RUN/shortlist.json"
phase generate-tune     "completed:$RUN/tune.json"
phase judge-tune        "completed:$RUN/tune_scored.json"
phase freeze            "exists:$RUN/selection.json"
phase generate          "completed:$RUN/test.json"
phase judge-test        "completed:$RUN/test_scored.json"
phase diagnose          "completed:$RUN/tune_diagnostics.json"
phase judge-diagnostics "completed:$RUN/diagnostics_scored.json"
echo "== $(date '+%F %T') FULL V1 CHAIN DONE $MODEL"

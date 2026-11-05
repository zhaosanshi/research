#!/usr/bin/env bash
# Runs INSIDE truth_dec: the whole QA-DAS chain for one model and one experiment config, every phase skipped when its
# artifact is already complete, stopping after STOP_AFTER (default: everything). Same phases as full_v1_inner_v2.sh.
# usage: full_variant_inner.sh MODEL CONFIG RESULTS_DIR
# env: TD_DATA_DIR TD_MODEL_ROOT CUDA_VISIBLE_DEVICES DEEPSEEK_API_KEY_FILE STOP_AFTER
set -u
MODEL=$1; CONFIG=$2; RES=$3
ROOT=/workspace/Truth_Decoupling
RUN=$RES/$MODEL/full
STOP=${STOP_AFTER:-judge-diagnostics}
export TD_RESULTS_DIR=$RES
mkdir -p "$RES/$MODEL"
cd "$ROOT/review_e25/claude_gpt_review/from_claude/code" || exit 1
# Red team 2026-11-05: an existing run directory must belong to THIS config, or a resume would silently continue another one.
if [ -f "$RUN/bank.json" ]; then
  have=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['config'].get('name',''))" "$RUN/bank.json")
  [ "$have" = "$CONFIG" ] || { echo "== $(date '+%F %T') REFUSED: $RUN holds config '$have', not '$CONFIG'"; exit 1; }
fi
# Seed the judge cache from the v1 run: identical (question, answer) grades are reused instead of re-asked (cost, judge noise).
V1CACHE=$ROOT/results_qa_das_v1/judge_cache/judge_cache_deepseek-chat.jsonl
if [ ! -f "$RES/judge_cache/judge_cache_deepseek-chat.jsonl" ] && [ -f "$V1CACHE" ]; then
  mkdir -p "$RES/judge_cache" && cp "$V1CACHE" "$RES/judge_cache/" && echo "== $(date '+%F %T') judge cache seeded from v1 ($(wc -l < "$V1CACHE") rows)"
fi
is_done() {
  local kind=${1%%:*} file=${1#*:}
  [ -f "$file" ] || return 1
  [ "$kind" = exists ] && return 0
  python3 - "$file" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("completed") else 1)
PY
}
phase() {
  local name=$1 check=$2; shift 2
  if is_done "$check"; then
    echo "== $(date '+%F %T') $name $MODEL: already complete (${check#*:}), skipped"
  else
    echo "== $(date '+%F %T') $name $MODEL config=$CONFIG (GPU ${CUDA_VISIBLE_DEVICES:-?})"
    python3 scripts/e25_qa_das.py "$name" --model "$MODEL" --run-dir "$RUN" "$@" || { echo "== $(date '+%F %T') PHASE FAILED: $name $MODEL"; exit 1; }
  fi
  if [ "$name" = "$STOP" ]; then echo "== $(date '+%F %T') STOPPED AFTER $name $MODEL (as requested)"; exit 0; fi
}
phase estimate          "exists:$RUN/bank.json" --all-layers --config "$CONFIG"
phase train             "completed:$RUN/training.json"
phase tune-e            "exists:$RUN/shortlist.json"
phase generate-tune     "completed:$RUN/tune.json"
phase judge-tune        "completed:$RUN/tune_scored.json"
phase freeze            "exists:$RUN/selection.json"
phase generate          "completed:$RUN/test.json"
phase judge-test        "completed:$RUN/test_scored.json"
phase diagnose          "completed:$RUN/tune_diagnostics.json"
phase judge-diagnostics "completed:$RUN/diagnostics_scored.json"
echo "== $(date '+%F %T') FULL CHAIN DONE $MODEL config=$CONFIG"

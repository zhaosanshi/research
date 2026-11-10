#!/usr/bin/env bash
# AllenAI judging of a subset of finished generation files (early look while the main run continues on GPU 0).
# GPT-reviewed scripts unchanged (e89b0d8): export with an explicit --gen list, then cells / allenai / single in a
# separate working directory judge_<split>_<NAME>/ so the final full judging in judge_<split>/ is untouched.
# usage: e43_judge_partial.sh NAME SPLIT FAMILY "LAYERS" "ARMS"     (GPU 1; no Pro, no GPT)
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONDONTWRITEBYTECODE=1
NAME=$1; SPLIT=$2; FAMILY=$3; LAYERS=$4; ARMS=$5
RUN=/workspace/Truth_Decoupling/results_e43_qa_prompt_das/llama3.1-8b
SRC=/workspace/Truth_Decoupling/results_qa_das_v1u/llama3.1-8b/full
CODE=/workspace/Truth_Decoupling/review_e43_claude/claude_gpt_review/from_claude/code
MODELS=/workspace/Truth_Decoupling/models
WORK=$RUN/judge_${SPLIT}_$NAME
cd "$CODE"
step() { echo "=== $(date '+%F %T') $* ==="; }
FILES=()
for L in $LAYERS; do for A in $ARMS; do
  f="${FAMILY}_L${L}_${A}_${SPLIT}.json"
  [ -f "$RUN/gen/$f" ] || { echo "missing $f"; exit 1; }
  FILES+=("$f")
done; done
step "export ${#FILES[@]} generation files as $NAME ($SPLIT)"
python scripts/e43_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split "$SPLIT" --gen "${FILES[@]}" --name "partial_$NAME" | tail -n 3
mkdir -p "$WORK"
cp -n "$RUN/judge_$SPLIT/partial_$NAME.json" "$WORK/judge_input.json"
cmp "$RUN/judge_$SPLIT/partial_$NAME.json" "$WORK/judge_input.json"
step "cells"
python scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
step "allenai"
python scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
  --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B"
step "single allenai"
python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$WORK/single_stdout.md"
step "partial judging $NAME done"

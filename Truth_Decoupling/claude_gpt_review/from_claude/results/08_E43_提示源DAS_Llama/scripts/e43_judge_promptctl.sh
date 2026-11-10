#!/usr/bin/env bash
# AllenAI judging of the four prompt-only controls (x / x_E / x_conf / x_caut frozen answers), tune split only.
# Uses the GPT-reviewed scripts unchanged (e89b0d8): e43_export_judge.py with an empty --gen list exports just the
# prompt controls; cells / allenai / single run in a separate working directory so the later full judging in
# judge_tune/ is untouched. GPU 1 (the main run is on GPU 0). No Pro, no GPT.
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1
SPLIT=${1:-tune}
RUN=/workspace/Truth_Decoupling/results_e43_qa_prompt_das/llama3.1-8b
SRC=/workspace/Truth_Decoupling/results_qa_das_v1u/llama3.1-8b/full
CODE=/workspace/Truth_Decoupling/review_e43_claude/claude_gpt_review/from_claude/code
MODELS=/workspace/Truth_Decoupling/models
WORK=$RUN/judge_${SPLIT}_promptctl
cd "$CODE"
step() { echo "=== $(date '+%F %T') $* ==="; }

step "export prompt controls only ($SPLIT)"
python scripts/e43_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split "$SPLIT" --gen --name prompt_controls
mkdir -p "$WORK"
cp -n "$RUN/judge_$SPLIT/prompt_controls.json" "$WORK/judge_input.json"
cmp "$RUN/judge_$SPLIT/prompt_controls.json" "$WORK/judge_input.json"
step "cells"
python scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json
step "allenai"
python scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
  --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B"
step "single allenai"
python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai
step "prompt-control judging done"

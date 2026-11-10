#!/usr/bin/env bash
# E43 judging, AllenAI only (赵磊 11-10 13:00: free judge first, see whether anything moves). No Pro, no GPT.
# export (verified artifacts) -> cells -> AllenAI truth + info judges (bf16, single GPU)
# -> single-judge exit. SPLIT defaults to tune (192 selection questions); pass train for the in-sample diagnostic.
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
SPLIT=${1:-tune}
RUN=/workspace/Truth_Decoupling/results_e43_qa_prompt_das/llama3.1-8b
SRC=/workspace/Truth_Decoupling/results_qa_das_v1u/llama3.1-8b/full
CODE=/workspace/Truth_Decoupling/review_e43_claude/claude_gpt_review/from_claude/code
MODELS=/workspace/Truth_Decoupling/models
cd "$CODE"
step() { echo "=== $(date '+%F %T') $* ==="; }

step "export $SPLIT"
python scripts/e43_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split "$SPLIT" --name judge_input
step "cells $SPLIT"
python scripts/review_checks/judge_two_axis_run.py cells "$RUN/judge_$SPLIT" --generation judge_input.json
step "allenai $SPLIT"
python scripts/allenai_tqa_judge.py --cells "$RUN/judge_$SPLIT/judge2/cells.jsonl" --out "$RUN/judge_$SPLIT/judge2/allenai.jsonl" \
  --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B"
step "single allenai $SPLIT"
python scripts/review_checks/judge_two_axis_run.py single "$RUN/judge_$SPLIT" --judge allenai
step "judging $SPLIT done"

#!/usr/bin/env bash
# Small DeepSeek Pro trial (赵磊 11-10 17:21: Pro allowed; a trial run is enough if the full pass adds little).
# Purpose: calibrate the AllenAI judges' length / format bias, not to grade the whole E43 run.
# Cells: the four prompt-only controls + E L18 main arm on the 192 tune questions (5 conditions).
# GPT-reviewed scripts unchanged (e89b0d8); separate working directory; AllenAI on GPU 1 for the same cells.
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONDONTWRITEBYTECODE=1
export DEEPSEEK_API_KEY_FILE=/workspace/Truth_Decoupling/logs/.deepseek_key
RUN=/workspace/Truth_Decoupling/results_e43_qa_prompt_das/llama3.1-8b
SRC=/workspace/Truth_Decoupling/results_qa_das_v1u/llama3.1-8b/full
CODE=/workspace/Truth_Decoupling/review_e43_claude/claude_gpt_review/from_claude/code
MODELS=/workspace/Truth_Decoupling/models
WORK=$RUN/judge_tune_protrial
cd "$CODE"
step() { echo "=== $(date '+%F %T') $* ==="; }
step "export prompt controls + E_L18_main"
python scripts/e43_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split tune --gen E_L18_main_tune.json --name partial_protrial | tail -n 3
mkdir -p "$WORK"
cp -n "$RUN/judge_tune/partial_protrial.json" "$WORK/judge_input.json"
cmp "$RUN/judge_tune/partial_protrial.json" "$WORK/judge_input.json"
step "cells"
python scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
step "allenai"
python scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
  --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 1
python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$WORK/single_allenai_stdout.md"
step "pro (DeepSeek V4 Pro, two questions per cell)"
python scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
step "single pro"
python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$WORK/single_pro_stdout.md"
cat "$WORK/single_pro_stdout.md"
step "pro trial done"

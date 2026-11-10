#!/usr/bin/env bash
# E44 step 0 (prompt-only; plan 25 as revised in 26 section 6; GPT round 28: "第 0 步可以跑").
# Code = GPT's 6bfca9f tree. Freezes the 11 prompts on train (96) + tune (192), scores the reference readout for every
# prompt without any write, exports the prompt controls and judges them with AllenAI. Nothing is selected from the
# result: E1 is the pre-fixed main sentence, E2 a sensitivity control, M a diagnostic. Single GPU. No Pro here.
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONDONTWRITEBYTECODE=1
RUN=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b
SRC=/workspace/Truth_Decoupling/results_qa_das_v1u/llama3.1-8b/full
CODE=/workspace/Truth_Decoupling/review_e44_claude/claude_gpt_review/from_claude/code
MODELS=/workspace/Truth_Decoupling/models
PROMPTS="x x_E1 x_F conf caut conf_E1 conf_E2 caut_E1 caut_E2 conf_M caut_M"
COMMON=(--source-run "$SRC" --run-dir "$RUN" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
cd "$CODE"
mkdir -p "$RUN"
step() { echo "=== $(date '+%F %T') $* ==="; }

step "freeze train + tune (11 prompts)"
python scripts/e44_factorial_das.py freeze "${COMMON[@]}" --splits train tune
for SPLIT in tune train; do
  step "reference readout without any write: $SPLIT"
  python scripts/e44_factorial_das.py mc "${COMMON[@]}" --split "$SPLIT" --bases $PROMPTS
done
for SPLIT in tune train; do
  WORK=$RUN/judge_${SPLIT}_step0
  step "export prompt controls: $SPLIT"
  python scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split "$SPLIT" --gen --name step0 | tail -n 3
  mkdir -p "$WORK"
  cp --update=none "$RUN/judge_$SPLIT/step0.json" "$WORK/judge_input.json"
  cmp "$RUN/judge_$SPLIT/step0.json" "$WORK/judge_input.json"
  step "cells + allenai + single: $SPLIT"
  python scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
  python scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
    --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 1
  python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$WORK/single_allenai_stdout.md"
  cat "$WORK/single_allenai_stdout.md"
done
step "summarize"
python scripts/e44_factorial_das.py summarize --source-run "$SRC" --run-dir "$RUN"
step "step 0 done"

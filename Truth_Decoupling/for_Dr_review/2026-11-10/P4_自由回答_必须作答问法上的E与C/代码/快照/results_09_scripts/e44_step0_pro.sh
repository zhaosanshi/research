#!/usr/bin/env bash
# DeepSeek V4 Pro on the E44 step-0 prompt controls (11 prompts x 192 tune questions; no intervention).
# Authorisation: 赵磊 11-10 17:21 (Pro allowed) and 23:34 (judge the 2x2 with both judges separately).
# The cells and the AllenAI votes already exist in this working directory (e44_step0.sh); the Pro cache of the two
# E43 trials is copied in, so x / x_E1 / conf / caut answers that are byte-identical are not paid for again.
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export DEEPSEEK_API_KEY_FILE=/workspace/Truth_Decoupling/logs/.deepseek_key
RUN=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b
OLD=/workspace/Truth_Decoupling/results_e43_qa_prompt_das/llama3.1-8b/judge_tune_protrial2/judge2/cache
CODE=/workspace/Truth_Decoupling/review_e44_claude/claude_gpt_review/from_claude/code
WORK=$RUN/judge_tune_step0
cd "$CODE"
step() { echo "=== $(date '+%F %T') $* ==="; }
mkdir -p "$WORK/judge2/cache"
for f in "$OLD"/*; do                                 # regular files only (the request-log directory is not a cache)
  [ -f "$f" ] || continue
  [ -e "$WORK/judge2/cache/$(basename "$f")" ] || cp "$f" "$WORK/judge2/cache/"
done
ls -la "$WORK/judge2/cache" | tail -n +2
step "pro (two questions per unique cell)"
python scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
step "single pro"
python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$WORK/single_pro_stdout.md"
cat "$WORK/single_pro_stdout.md"
step "step-0 pro done"

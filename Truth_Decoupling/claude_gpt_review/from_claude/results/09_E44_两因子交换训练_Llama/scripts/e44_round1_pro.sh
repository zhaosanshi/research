#!/usr/bin/env bash
# DeepSeek V4 Pro judging of ONE shortlisted layer of one family (plan 27, section 4.4). One process per layer, CPU only.
#   FAMILY=E|C  LAYER=n  [TAG=A]  [ARMS="x.w.p1 ..."]  [PROMPTS="x"]  [RUN=<run dir>]
# Batch A (default) = the plain-prompt arms that decide the 2x2; the baseline prompt x was Pro-judged in step 0,
# and that cache is copied in, so the baseline is not paid for again. The judge code is GPT's reviewed runner, unchanged.
# Authorisation: 赵磊 11-10 17:21 (Pro allowed) and 23:34 (both judges, each judging its own complete 2x2).
set -euo pipefail
: "${FAMILY:?}" "${LAYER:?}"
TAG=${TAG:-A}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DEEPSEEK_API_KEY_FILE=/workspace/Truth_Decoupling/logs/.deepseek_key
ROOT=/workspace/Truth_Decoupling
MAIN=$ROOT/results_e44_factorial_das/llama3.1-8b
RUN=${RUN:-$MAIN}
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
CODE=$ROOT/review_e44_claude/claude_gpt_review/from_claude/code
if [ -z "${ARMS:-}" ]; then
  if [ "$FAMILY" = E ]; then ARMS="x.w.p1 x.w.m1 x.md.p1 x.rand.p1"; else ARMS="x.w.p1 x.w.m1 x.md.p1 x.md.m1 x.rand.p1 x.rand.m1"; fi
fi
PROMPTS=${PROMPTS:-x}
NAME=pro${TAG}_${FAMILY}_L${LAYER}
WORK=$RUN/judge_tune_$NAME
GEN=""; for a in $ARMS; do GEN="$GEN ${FAMILY}_L${LAYER}_${a//./-}_tune.json"; done
step() { echo "=== $(date '+%F %T') [$NAME] $* ==="; }
cd "$CODE"
step "export: prompts $PROMPTS; arms $ARMS"
python scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split tune --gen $GEN --prompts $PROMPTS --name "$NAME" | tail -n 3
mkdir -p "$WORK/judge2/cache"
[ -f "$WORK/judge_input.json" ] || cp "$RUN/judge_tune/$NAME.json" "$WORK/judge_input.json"
cmp "$RUN/judge_tune/$NAME.json" "$WORK/judge_input.json"
for f in "$MAIN/judge_tune_step0/judge2/cache"/*; do   # regular files only
  [ -f "$f" ] || continue
  [ -e "$WORK/judge2/cache/$(basename "$f")" ] || cp "$f" "$WORK/judge2/cache/"
done
step "cells"
python scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
step "pro (two questions per unique cell)"
python scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
step "single pro"
python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$WORK/single_pro_stdout.md"
step "$NAME done"

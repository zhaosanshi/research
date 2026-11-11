#!/usr/bin/env bash
# AllenAI only, for a named subset of generations of the main run dir (exploratory arms, e.g. strength sweeps).
#   FAMILY=E|C  NAME=dose_E  LAYERS="8 10"  ARMS="x.w.p0.5 x.w.p2"  PROMPTS="x conf"  GPU=k
set -euo pipefail
: "${FAMILY:?}" "${NAME:?}" "${LAYERS:?}" "${ARMS:?}" "${GPU:?}"
PROMPTS=${PROMPTS:-x}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT=/workspace/Truth_Decoupling
RUN=$ROOT/results_e44_factorial_das/llama3.1-8b
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
WORK=$RUN/judge_tune_$NAME
GEN=""; for l in $LAYERS; do for a in $ARMS; do GEN="$GEN ${FAMILY}_L${l}_${a//./-}_tune.json"; done; done
cd $ROOT/review_e44_claude/claude_gpt_review/from_claude/code
echo "=== $(date '+%F %T') [$NAME] export: layers $LAYERS; arms $ARMS; prompts $PROMPTS ==="
python scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split tune --gen $GEN --prompts $PROMPTS --name "$NAME" | tail -n 3
mkdir -p "$WORK"
[ -f "$WORK/judge_input.json" ] || cp "$RUN/judge_tune/$NAME.json" "$WORK/judge_input.json"
cmp "$RUN/judge_tune/$NAME.json" "$WORK/judge_input.json"
python scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
CUDA_VISIBLE_DEVICES=$GPU python scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
  --truth-model "$ROOT/models/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$ROOT/models/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 1
python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$WORK/single_allenai_stdout.md"
echo "=== $(date '+%F %T') [$NAME] done ==="

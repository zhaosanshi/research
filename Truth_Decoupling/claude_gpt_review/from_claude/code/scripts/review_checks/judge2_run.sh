#!/usr/bin/env bash
# Two-axis grading steps inside the truth_dec container (host side; see judge_two_axis_run.py for the flow).
# usage: judge2_run.sh RUN_DIR STEP [args]
#   RUN_DIR  container path of the run, e.g. /workspace/Truth_Decoupling/results_e42_20261108/llama3.1-8b/select-tune
#   STEP     cells | allenai (GPU) | dev-check (GPU; RUN_DIR is not used, pass "-") | pro | disputes | gpt | adjudicate | human
# env: GPU (default 0). Keys: logs/.deepseek_key, logs/.xiaoai_key (chmod 600) are passed as *_FILE, never on a command line.
set -euo pipefail
ROOT=/workspace/Truth_Decoupling
CODE=$ROOT/review_r23_claude/claude_gpt_review/from_claude/code
MODELS=$ROOT/models
RUN_DIR=${1:?RUN_DIR}; STEP=${2:?STEP}; shift 2
GPU=${GPU:-0}
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$MODELS -e PYTHONUNBUFFERED=1
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -e XIAOAI_API_KEY_FILE=$ROOT/logs/.xiaoai_key
    -e CUDA_VISIBLE_DEVICES=$GPU -w $CODE truth_dec)
RUN=scripts/review_checks/judge_two_axis_run.py
TRUTH=$MODELS/allenai__truthfulqa-truth-judge-llama2-7B
INFO=$MODELS/allenai__truthfulqa-info-judge-llama2-7B
case "$STEP" in
  cells)      "${DX[@]}" python3 $RUN cells "$RUN_DIR" "$@" ;;
  allenai)    "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells "$RUN_DIR/judge2/cells.jsonl" \
                 --out "$RUN_DIR/judge2/allenai.jsonl" --truth-model "$TRUTH" --info-model "$INFO" "$@" ;;
  dev-check)  "${DX[@]}" python3 scripts/allenai_tqa_judge.py --truth-model "$TRUTH" --info-model "$INFO" --dev-check "$@" ;;
  pro)        "${DX[@]}" python3 $RUN llm "$RUN_DIR" --judge pro "$@" ;;
  disputes)   "${DX[@]}" python3 $RUN disputes "$RUN_DIR" "$@" ;;
  gpt)        "${DX[@]}" python3 $RUN llm "$RUN_DIR" --judge gpt --only disputes "$@" ;;
  adjudicate) "${DX[@]}" python3 $RUN adjudicate "$RUN_DIR" "$@" ;;
  human)      "${DX[@]}" python3 $RUN human "$RUN_DIR" "$@" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac

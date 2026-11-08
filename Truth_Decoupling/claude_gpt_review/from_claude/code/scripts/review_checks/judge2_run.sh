#!/usr/bin/env bash
# Future-run wrapper; never edits a running snapshot or result directory.
# judge2_run.sh RUN_DIR STEP [--config CONTAINER_CONFIG] [step args...]
# STEP: cells | allenai (GPU) | dev-check (GPU; RUN_DIR unused, pass "-") | pro | disputes | gpt | adjudicate | human | analysis
# CODE_PATH may name an immutable container snapshot; otherwise this script's
# host directory is mapped from TD_HOST_ROOT to TD_CONTAINER_ROOT.
# Other overrides: TD_CONTAINER, TD_CONTAINER_USER, TD_MODEL_ROOT, TD_HF_HOME,
# TD_TRUTH_JUDGE_MODEL, TD_INFO_JUDGE_MODEL, GPU, and *_API_KEY_FILE (paths only).
set -euo pipefail
RUN_DIR=${1:?RUN_DIR}; STEP=${2:?STEP}; shift 2
TASK_CONTAINER_ROOT=${TD_CONTAINER_ROOT:-/workspace/Truth_Decoupling}
TASK_HOST_ROOT=${TD_HOST_ROOT:-/data/sanshizhao/Truth_Decoupling}
CONTAINER=${TD_CONTAINER:-truth_dec}
CONTAINER_USER=${TD_CONTAINER_USER:-1000:1001}
HOST_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ -n ${CODE_PATH:-} ]]; then
    CODE=$CODE_PATH
elif [[ "$HOST_CODE" == "$TASK_HOST_ROOT" || "$HOST_CODE" == "$TASK_HOST_ROOT/"* ]]; then
    CODE="$TASK_CONTAINER_ROOT${HOST_CODE#"$TASK_HOST_ROOT"}"
else
    echo 'Set CODE_PATH to the immutable container code directory (host code is outside TD_HOST_ROOT).' >&2
    exit 2
fi
MODELS=${TD_MODEL_ROOT:-$TASK_CONTAINER_ROOT/models}
HF_CACHE=${TD_HF_HOME:-$TASK_CONTAINER_ROOT/hf_cache}
GPU=${GPU-0}
CONFIG=${TD_JUDGE2_CONFIG:-$CODE/configs/judges/two_axis_v6.json}
EXTRA=()
while (( $# )); do
    case "$1" in
        --config)
            if (( $# < 2 )) || [[ -z "$2" ]]; then echo '--config requires a container path' >&2; exit 2; fi
            CONFIG=$2; shift 2 ;;
        --config=*) CONFIG=${1#--config=}; [[ -n "$CONFIG" ]] || { echo 'empty --config' >&2; exit 2; }; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done
# HOME must be writable for the container user (libraries write under ~); TD_JUDGE2_CACHE / TD_JUDGE2_HUMAN (container
# paths) share one API-reply cache and one human-verdict file across runs so identical inputs get identical verdicts.
DX=(docker exec -u "$CONTAINER_USER" -e "HOME=$HF_CACHE" -e "HF_HOME=$HF_CACHE" -e "TD_MODEL_ROOT=$MODELS" -e PYTHONUNBUFFERED=1
    -e "DEEPSEEK_API_KEY_FILE=${DEEPSEEK_API_KEY_FILE:-$TASK_CONTAINER_ROOT/logs/.deepseek_key}"
    -e "XIAOAI_API_KEY_FILE=${XIAOAI_API_KEY_FILE:-$TASK_CONTAINER_ROOT/logs/.xiaoai_key}"
    -e "TD_JUDGE2_CACHE=${TD_JUDGE2_CACHE:-}" -e "TD_JUDGE2_HUMAN=${TD_JUDGE2_HUMAN:-}"
    -e "TD_JUDGE2_CONFIG=$CONFIG" -e "CUDA_VISIBLE_DEVICES=$GPU" -w "$CODE" "$CONTAINER")
RUN=scripts/review_checks/judge_two_axis_run.py
TRUTH=${TD_TRUTH_JUDGE_MODEL:-$MODELS/allenai__truthfulqa-truth-judge-llama2-7B}
INFO=${TD_INFO_JUDGE_MODEL:-$MODELS/allenai__truthfulqa-info-judge-llama2-7B}
case "$STEP" in
    cells) CMD=(python3 "$RUN" --config "$CONFIG" cells "$RUN_DIR") ;;
    allenai) CMD=(python3 scripts/allenai_tqa_judge.py --cells "$RUN_DIR/judge2/cells.jsonl"
                    --out "$RUN_DIR/judge2/allenai.jsonl" --truth-model "$TRUTH" --info-model "$INFO") ;;
    dev-check) CMD=(python3 scripts/allenai_tqa_judge.py --truth-model "$TRUTH" --info-model "$INFO" --dev-check) ;;
    pro) CMD=(python3 "$RUN" --config "$CONFIG" llm "$RUN_DIR" --judge pro) ;;
    disputes) CMD=(python3 "$RUN" --config "$CONFIG" disputes "$RUN_DIR") ;;
    gpt) CMD=(python3 "$RUN" --config "$CONFIG" llm "$RUN_DIR" --judge gpt --only disputes) ;;
    adjudicate) CMD=(python3 "$RUN" --config "$CONFIG" adjudicate "$RUN_DIR") ;;
    human) CMD=(python3 "$RUN" --config "$CONFIG" human "$RUN_DIR") ;;
    analysis) CMD=(python3 scripts/review_checks/e42_analysis.py "$RUN_DIR" --two-axis) ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
# Bash 3.2 + nounset rejects expansion of an empty array: append only when nonempty.
if (( ${#EXTRA[@]} )); then CMD+=("${EXTRA[@]}"); fi
exec "${DX[@]}" "${CMD[@]}"

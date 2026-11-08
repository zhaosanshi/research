#!/usr/bin/env bash
# Future E42 launcher: e42_run.sh MODEL GPU PHASE [extra experiment args...]
# CODE_PATH overrides the immutable container snapshot; otherwise map this
# script's actual host directory from TD_HOST_ROOT to TD_CONTAINER_ROOT.
# Optional roots: TD_RESULTS_DIR, TD_HOST_RESULTS_ROOT, TD_MODEL_ROOT,
# TD_DATA_DIR, TD_SOURCE_ROOT, TD_HF_HOME, TD_CONTAINER, TD_CONTAINER_USER.
# E42_TAG names the run directory under TD_RESULTS_DIR/MODEL (default: the phase name; a complete run there is never
# overwritten - set E42_TAG for a re-run). SOURCE_RUN overrides the E25 v1u source run.
# Existing live launchers/results are not changed by this wrapper.
set -euo pipefail
MODEL=${1:?model}; GPU=${2:?gpu}; PHASE=${3:?phase}; shift 3
case "$PHASE" in oracle-check|oracle-tune|select-check|select-tune|judge-tune) ;; *) echo "unknown phase $PHASE" >&2; exit 2;; esac
TASK_CONTAINER_ROOT=${TD_CONTAINER_ROOT:-/workspace/Truth_Decoupling}
TASK_HOST_ROOT=${TD_HOST_ROOT:-/data/sanshizhao/Truth_Decoupling}
HOST_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ -n ${CODE_PATH:-} ]]; then
    CODE=$CODE_PATH
elif [[ "$HOST_CODE" == "$TASK_HOST_ROOT" || "$HOST_CODE" == "$TASK_HOST_ROOT/"* ]]; then
    CODE="$TASK_CONTAINER_ROOT${HOST_CODE#"$TASK_HOST_ROOT"}"
else
    echo 'Set CODE_PATH to the immutable container code directory.' >&2; exit 2
fi
RES=${TD_RESULTS_DIR:-$TASK_CONTAINER_ROOT/results_e42_20261108}
TAG=${E42_TAG:-$PHASE}
SOURCE=${SOURCE_RUN:-${TD_SOURCE_ROOT:-$TASK_CONTAINER_ROOT/results_qa_das_v1u}/$MODEL/full}
RUN=$RES/$MODEL/$TAG
if [[ -n ${TD_HOST_RESULTS_ROOT:-} ]]; then
    HOST_RESULTS=$TD_HOST_RESULTS_ROOT
elif [[ "$RES" == "$TASK_CONTAINER_ROOT/"* ]]; then
    HOST_RESULTS="$TASK_HOST_ROOT${RES#"$TASK_CONTAINER_ROOT"}"
else
    echo 'Set TD_HOST_RESULTS_ROOT when TD_RESULTS_DIR is outside TD_CONTAINER_ROOT.' >&2; exit 2
fi
mkdir -p "$HOST_RESULTS/$MODEL/$TAG"
exec docker exec -u "${TD_CONTAINER_USER:-1000:1001}" \
  -e "HOME=${TD_HF_HOME:-$TASK_CONTAINER_ROOT/hf_cache}" \
  -e "HF_HOME=${TD_HF_HOME:-$TASK_CONTAINER_ROOT/hf_cache}" -e "TD_RESULTS_DIR=$RES" \
  -e "TD_DATA_DIR=${TD_DATA_DIR:-$TASK_CONTAINER_ROOT/code/data/processed}" \
  -e "TD_MODEL_ROOT=${TD_MODEL_ROOT:-$TASK_CONTAINER_ROOT/models}" \
  -e "DEEPSEEK_API_KEY_FILE=${DEEPSEEK_API_KEY_FILE:-$TASK_CONTAINER_ROOT/logs/.deepseek_key}" \
  -e "CUDA_VISIBLE_DEVICES=$GPU" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -w "$CODE" "${TD_CONTAINER:-truth_dec}" python3 scripts/e42_qa_oracle_and_select.py "$PHASE" \
  --source-run "$SOURCE" --run-dir "$RUN" "$@"

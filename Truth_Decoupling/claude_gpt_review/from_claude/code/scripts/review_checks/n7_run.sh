#!/usr/bin/env bash
# N7 step 1 (赵磊 2026-11-08 01:55, notepad N7): the Dr.'s sentence tested one last time with a C mid-strength curve.
# n7_run.sh MODEL GPU ALPHA [extra E41 args]  — runs E41 (fixed C x E, two-pass TF protocol) with the model's
# 2x2-validated C at strength ALPHA in BOTH directions and only the 2x2-chosen E layer/strength; then the
# pre-declared E41 analysis (same number of self-selected answers vs a threshold on E's gate / confidence).
# Output (container): TD_RESULTS_DIR/E41/MODEL/e41_n7_a<ALPHA>{.json,.npz,_analysis.json,_analysis.md}
# The C names / E condition below are the ones recorded in results_e41_20261107/E41/<model>/e41_das{,_analysis}.json.
set -euo pipefail
MODEL=${1:?model}; GPU=${2:?gpu}; ALPHA=${3:?alpha}; shift 3
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
case "$MODEL" in
    llama3.1-8b) CK=Cp; CA=25; CN=26; EL=11; EA=3 ;;
    qwen3-8b)    CK=Cs; CA=33; CN=24; EL=35; EA=3 ;;
    gemma3-4b)   CK=Cp; CA=25; CN=23; EL=12; EA=3 ;;
    *) echo "no 2x2-validated C / E on record for $MODEL (see results_e41_20261107/E41)" >&2; exit 2 ;;
esac
RES=${TD_RESULTS_DIR:-$TASK_CONTAINER_ROOT/results_n7_20261108}
TAG=${N7_TAG:-n7_a$ALPHA}
HF=${TD_HF_HOME:-$TASK_CONTAINER_ROOT/hf_cache}
DX=(docker exec -u "${TD_CONTAINER_USER:-1000:1001}" -e "HOME=$HF" -e "HF_HOME=$HF" -e "TD_RESULTS_DIR=$RES"
    -e "TD_DATA_DIR=${TD_DATA_DIR:-$TASK_CONTAINER_ROOT/code/data/processed}" -e "TD_MODEL_ROOT=${TD_MODEL_ROOT:-$TASK_CONTAINER_ROOT/models}"
    -e "CUDA_VISIBLE_DEVICES=$GPU" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e "PYTHONPATH=$CODE"
    -w "$CODE" "${TD_CONTAINER:-truth_dec}")
"${DX[@]}" python3 scripts/e41_fixed_c_all_e.py --exp E2b_tqa --model "$MODEL" --tune-pairs 192 --e-estimator das \
    --das-dir "$TASK_CONTAINER_ROOT/results_das_tfce/E1/$MODEL" --das-variant tf-ce-bidir \
    --c-abst "${CK}_L${CA}_abst_a${ALPHA}" --c-ans "${CK}_L${CN}_ans_a${ALPHA}" --e-layers "$EL" --e-alphas "$EA" --tag "$TAG" "$@"
"${DX[@]}" python3 scripts/review_checks/e41_analysis.py "$RES/E41/$MODEL/e41_$TAG.json" --chosen-e "E_L${EL}_a${EA}" \
    --out "$RES/E41/$MODEL/e41_${TAG}_analysis"

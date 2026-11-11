#!/usr/bin/env bash
# E45 (from_claude/40): true/false with an Unknown option in the same pass; the main 2x2's E and C; E+C at the C strength
# that brings the answer count back (tuning half, counts only); then the pre-declared analysis. Host-side; runs in the container.
#   e45_run.sh MODEL GPU TAG [extra e45 args, e.g. --limit 48 for a smoke run: then no analysis]
set -euo pipefail
MODEL=${1:?model}; GPU=${2:?gpu}; TAG=${3:?tag}; shift 3
ROOT=/workspace/Truth_Decoupling
CODE=${CODE_PATH:-$ROOT/code_e45_20261112}      # a fixed copy, so that the four parallel runs use the same code
case "$MODEL" in      # the main 2x2's choices, as recorded in results_e41_20261107/E41/<model>/e41_das.json (Mistral: E23 report)
    llama3.1-8b) CK=Cp; CA=25; CN=26; EL=11; REF=$ROOT/results_e41_20261107/E41/llama3.1-8b/e41_das.json ;;
    qwen3-8b)    CK=Cs; CA=33; CN=24; EL=35; REF=$ROOT/results_e41_20261107/E41/qwen3-8b/e41_das.json ;;
    gemma3-4b)   CK=Cp; CA=25; CN=23; EL=12; REF=$ROOT/results_e41_20261107/E41/gemma3-4b/e41_das.json ;;
    mistral-7b)  CK=Cs; CA=18; CN=20; EL=13; REF= ;;
    *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
RES=${TD_RESULTS_DIR:-$ROOT/results_e45_20261112}
DX=(docker exec -u 1000:1001 -e "HOME=$ROOT/hf_cache" -e "HF_HOME=$ROOT/hf_cache" -e "TD_RESULTS_DIR=$RES"
    -e "TD_DATA_DIR=$ROOT/code/data/processed" -e "TD_MODEL_ROOT=$ROOT/models" -e "CUDA_VISIBLE_DEVICES=$GPU"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=16
    -e "PYTHONPATH=$CODE" -w "$CODE" truth_dec)
REFARG=()
[ -n "$REF" ] && REFARG=(--ref-e41-json "$REF")
"${DX[@]}" python3 scripts/e45_tf_idk_ec.py --exp E2b_tqa --model "$MODEL" --tune-pairs 192 \
    --das-dir "$ROOT/results_das_tfce/E1/$MODEL" --das-variant tf-ce-bidir --e-layer "$EL" --e-alpha 3 \
    --c-abst "${CK}_L${CA}_abst_a2" --c-ans "${CK}_L${CN}_ans_a2" --tag "$TAG" ${REFARG[@]+"${REFARG[@]}"} "$@"
if [ $# -eq 0 ]; then
    "${DX[@]}" python3 scripts/review_checks/e45_analysis.py "$RES/E45/$MODEL/e45_$TAG.json" --out "$RES/E45/$MODEL/e45_${TAG}_analysis"
fi

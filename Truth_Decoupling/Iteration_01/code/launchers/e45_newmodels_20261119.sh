#!/usr/bin/env bash
# E45 on the two new models (Iteration_01 work queue B, 2026-11-19, Claude Opus 5.5): true/false with an Unknown option in the
# same pass, the method of claude_gpt_review/from_claude/40 unchanged. scripts/e45_tf_idk_ec.py and
# scripts/review_checks/e45_analysis.py are byte-identical in code_e45_20261112 (the 11-12 run) and code_newmodels_20261112,
# which also holds the two models' configs and ran their true/false 2x2; the interpreter is the one the model config names
# (transformers 5), as in newmodel_tf_phase1.sh. E and C are the new models' 2x2 choices
# (results_newmodels_20261112/E23/<model>/report_2x2_das_tfce.md). Prediction file:
# Iteration_01/novelty/ours/2026-11-19_tf_unknown_ec_new_models.md. KIND Cp is the main run, Cs the supplementary one.
#   usage (host): e45_newmodels_20261119.sh MODEL GPU KIND        MODEL qwen3.5-9b | qwen3.8-27b, KIND Cp | Cs
#                 LIMIT=8 e45_newmodels_20261119.sh ...          smoke: first 8 evaluation statements, tag smoke_*, no analysis
set -euo pipefail
MODEL=${1:?model}; GPU=${2:?gpu}; KIND=${3:?Cp or Cs}
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/code_newmodels_20261112
case "$MODEL:$KIND" in
  qwen3.5-9b:Cp)  EL=31; CA=Cp_L16_abst_a1; CN=Cp_L17_ans_a2 ;;
  qwen3.5-9b:Cs)  EL=31; CA=Cs_L28_abst_a2; CN=Cs_L31_ans_a1 ;;
  qwen3.8-27b:Cp) EL=63; CA=Cp_L62_abst_a2; CN=Cp_L63_ans_a1 ;;
  qwen3.8-27b:Cs) EL=63; CA=Cs_L57_abst_a2; CN=Cs_L60_ans_a2 ;;
  *) echo "unknown model / kind: $MODEL $KIND" >&2; exit 2 ;;
esac
for f in scripts/e45_tf_idk_ec.py scripts/review_checks/e45_analysis.py; do
  cmp -s "$HOST/code_newmodels_20261112/$f" "$HOST/code_e45_20261112/$f" || { echo "$f differs from the 11-12 copy" >&2; exit 2; }
done
PY=$(docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$CODE/configs/models/$MODEL.json")
RES=$ROOT/results_e45_newmodels_20261119
TAG=nm_$KIND
EXTRA=()
if [ -n "${LIMIT:-}" ]; then TAG=smoke_nm_$KIND; EXTRA=(--limit "$LIMIT"); fi
DX=(docker exec -u 1000:1001 -e "HOME=$ROOT/hf_cache" -e "HF_HOME=$ROOT/hf_cache" -e "TD_RESULTS_DIR=$RES"
    -e "TD_DATA_DIR=$ROOT/code/data/processed" -e "TD_MODEL_ROOT=$ROOT/models" -e "CUDA_VISIBLE_DEVICES=$GPU"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=16
    -e "PYTHONPATH=$CODE" -w "$CODE" truth_dec)
echo "e45 new models: $MODEL $KIND on GPU $GPU, E layer $EL, C $CA / $CN, tag $TAG, interpreter $PY, results $RES"
"${DX[@]}" "$PY" scripts/e45_tf_idk_ec.py --exp E2b_tqa --model "$MODEL" --tune-pairs 192 \
    --das-dir "$ROOT/results_newmodels_20261112/E1/$MODEL" --das-variant tf-ce-bidir --e-layer "$EL" --e-alpha 3 \
    --c-abst "$CA" --c-ans "$CN" --tag "$TAG" ${EXTRA[@]+"${EXTRA[@]}"}
if [ -z "${LIMIT:-}" ]; then
  "${DX[@]}" "$PY" scripts/review_checks/e45_analysis.py "$RES/E45/$MODEL/e45_$TAG.json" --out "$RES/E45/$MODEL/e45_${TAG}_analysis"
fi
echo "e45 new models: $MODEL $KIND done"

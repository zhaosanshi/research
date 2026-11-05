#!/usr/bin/env bash
# Host side: tfce_C.sh MODEL GPU
# The C row of the 2x2 re-swept with C orthogonalised to the NEW E (gold-label CE DAS, variant tf-ce-bidir) of the same
# layer -- otherwise exactly round 5's c_das_tf_chain.sh (all layers, Cp + Cs, 192 tuning pairs, same rules) -- then the
# 2x2 report with E = das_tfce_full and C = das_tfce_C.
set -u
MODEL=$1; GPU=$2
ROOT=/workspace/Truth_Decoupling
RES=$ROOT/results_das_tfce
CODE=$ROOT/review_e25/claude_gpt_review/from_claude/code
HOSTLOG=/data/sanshizhao/Truth_Decoupling/results_das_tfce/$MODEL/C.log
run() {
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed \
    -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -w $CODE truth_dec python3 "$@" >> "$HOSTLOG" 2>&1
}
step() { local name=$1; shift; echo "== $(date '+%F %T') $name $MODEL (GPU $GPU)" >> "$HOSTLOG"; run "$@" || { echo "== $(date '+%F %T') STEP FAILED: $name $MODEL" >> "$HOSTLOG"; exit 1; }; }
step "C sweep, all layers, Cp + Cs, C orthogonal to the DAS-CE E of the same layer" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --tune-pairs 192 --skip-e \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf-ce-bidir --tag das_tfce_C
step "2x2 report" scripts/report_2x2.py --e-root "$RES" --e-tag das_tfce_full --c-root "$RES" --c-tag das_tfce_C --model "$MODEL" \
  --out "$RES/E23/$MODEL/report_2x2_das_tfce.md"
echo "== $(date '+%F %T') TFCE C DONE $MODEL" >> "$HOSTLOG"

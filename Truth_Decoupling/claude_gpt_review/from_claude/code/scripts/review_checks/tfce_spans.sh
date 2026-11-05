#!/usr/bin/env bash
# Host side: tfce_spans.sh MODEL GPU
# Multi-layer windows for the gold-label CE DAS directions, with EXACTLY the round-4 candidate set (das_tf_followup.sh:
# the single layers of the band plus every 3-layer window, alphas 2 3, 192 tuning pairs, same rule), then paired
# comparisons against the round-4 DAS windows (das_tf_spans) and mean-difference windows (meandiff_spans).
set -u
MODEL=$1; GPU=$2
ROOT=/workspace/Truth_Decoupling
RES=$ROOT/results_das_tfce
OLD=$ROOT/results_das_tf
CODE=$ROOT/review_e25/claude_gpt_review/from_claude/code
HOSTLOG=/data/sanshizhao/Truth_Decoupling/results_das_tfce/$MODEL/spans.log
case "$MODEL" in
  llama3.1-8b) BAND="8 9 10 11 12 13 14 15 16 17 18"; SPANS="8,9,10 9,10,11 10,11,12 11,12,13 12,13,14 13,14,15 14,15,16 15,16,17 16,17,18" ;;
  qwen3-8b)    BAND="15 16 17 18 19 20 21 22 23";     SPANS="15,16,17 16,17,18 17,18,19 18,19,20 19,20,21 20,21,22 21,22,23" ;;
  *) echo "no round-4 window set for $MODEL" >> "$HOSTLOG"; exit 1 ;;
esac
run() {
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed \
    -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -w $CODE truth_dec python3 "$@" >> "$HOSTLOG" 2>&1
}
step() { local name=$1; shift; echo "== $(date '+%F %T') $name $MODEL (GPU $GPU)" >> "$HOSTLOG"; run "$@" || { echo "== $(date '+%F %T') STEP FAILED: $name $MODEL" >> "$HOSTLOG"; exit 1; }; }
step "E23 windows (round-4 candidate set)" scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --layers $BAND --e-spans $SPANS \
  --tune-pairs 192 --skip-cp --no-cs --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf-ce-bidir --tag das_tfce_spans
for ref in das_tf_spans meandiff_spans; do
  step "compare das_tfce_spans vs $ref" scripts/e23_compare_estimators.py --root "$RES" --root-b "$OLD" --model "$MODEL" \
    --tag-a das_tfce_spans --label-a "DAS-CE windows" --tag-b "$ref" --label-b "$ref" --out "$RES/E23/$MODEL/compare_das_tfce_spans_vs_$ref.md"
done
echo "== $(date '+%F %T') TFCE SPANS DONE $MODEL" >> "$HOSTLOG"

#!/usr/bin/env bash
# Host side: tfce_chain.sh MODEL GPU
# TF-protocol DAS with the QA-DAS v1 objective (bidirectional gold-label CE at the prompt end), ONE variable changed vs the
# round-4 chain (das_tf_chain.sh steps 2-3a): --das-loss ce --das-bidir. Same 96 estimation pairs, same E23 sweep and rule.
# Results under /workspace/Truth_Decoupling/results_das_tfce/{E1,E23}/MODEL; nothing of results_das_tf is touched.
#   1. E1: DAS at every layer (variant tf-ce-bidir)
#   2. E23: all layers, E conditions only, 192 tuning pairs, alphas 2 3 (identical flags to das_tf_full)
#   3. paired comparison vs the round-4 DAS (das_tf_full) and mean difference (meandiff_full)
#   4. verify-own-answers with the new frozen selection
set -u
MODEL=$1; GPU=$2
ROOT=/workspace/Truth_Decoupling
RES=$ROOT/results_das_tfce
OLD=$ROOT/results_das_tf
CODE=$ROOT/review_e25/claude_gpt_review/from_claude/code
HOSTLOG=/data/sanshizhao/Truth_Decoupling/results_das_tfce/$MODEL/chain.log
mkdir -p "$(dirname "$HOSTLOG")"
run() {
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed \
    -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -w $CODE truth_dec python3 "$@" >> "$HOSTLOG" 2>&1
}
step() { local name=$1; shift; echo "== $(date '+%F %T') $name $MODEL (GPU $GPU)" >> "$HOSTLOG"; run "$@" || { echo "== $(date '+%F %T') STEP FAILED: $name $MODEL" >> "$HOSTLOG"; exit 1; }; }
step "1 E1 DAS tf-ce-bidir all layers" scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf \
  --das-anchor prompt --das-tags E --das-pairs 96 --das-all-layers --das-loss ce --das-bidir
step "2 E23 all layers E only" scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf-ce-bidir --tag das_tfce_full
mkdir -p /data/sanshizhao/Truth_Decoupling/results_das_tfce/E23/$MODEL
for ref in das_tf_full meandiff_full; do
  step "3 compare das_tfce_full vs $ref" scripts/e23_compare_estimators.py --root "$RES" --root-b "$OLD" --model "$MODEL" \
    --tag-a das_tfce_full --label-a "DAS-CE" --tag-b "$ref" --label-b "$ref" --out "$RES/E23/$MODEL/compare_das_tfce_vs_$ref.md"
done
step "4 verify own answers" scripts/review_checks/verify_own_answers.py "$MODEL" "$ROOT/results_qa_das_v1/$MODEL/full_v1" "$RES/E23/$MODEL" \
  "$ROOT/review_e25/checks/dasplus/verify_own_tfce_$MODEL.json"
echo "== $(date '+%F %T') TFCE CHAIN DONE $MODEL" >> "$HOSTLOG"

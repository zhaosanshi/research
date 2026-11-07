#!/bin/bash
# Mistral-7B through the same true/false chain as Gemma / Llama / Qwen (Claude, 2026-11-08, 赵磊: "用 mistral 跑上 gemma 同样的程序"):
#   1 new-objective DAS (gold-label, bidirectional CE, prompt end) at every layer     -> results_das_tfce/E1/mistral-7b
#   2 E sweep at every layer, 192 tuning pairs                                         -> results_das_tfce/E23/mistral-7b/knob_sweep_das_tfce_full
#   3 paired comparison with the old-objective DAS and mean-difference sweeps, if present
#   4 C sweep at every layer (prompt-contrast + self-report, orthogonal to the same-layer DAS-CE E) -> knob_sweep_das_tfce_C
#   5 2x2 report
# Code = GitHub HEAD snapshot in review_r22_claude (TF path identical to the 2x2 runs). usage: tfce_chain_mistral.sh GPU
set -u
MODEL=mistral-7b; GPU=$1
ROOT=/workspace/Truth_Decoupling
RES=$ROOT/results_das_tfce
OLD=$ROOT/results_das_tf
CODE=$ROOT/review_r22_claude/claude_gpt_review/from_claude/code
HOSTRES=/data/sanshizhao/Truth_Decoupling/results_das_tfce
HOSTLOG=$HOSTRES/$MODEL/chain.log
mkdir -p "$(dirname "$HOSTLOG")" "$HOSTRES/E23/$MODEL"
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
for ref in das_tf_full meandiff_full; do
  if [ -f "/data/sanshizhao/Truth_Decoupling/results_das_tf/E23/$MODEL/knob_sweep_$ref.npz" ]; then
    step "3 compare das_tfce_full vs $ref" scripts/e23_compare_estimators.py --root "$RES" --root-b "$OLD" --model "$MODEL" \
      --tag-a das_tfce_full --label-a "DAS-CE" --tag-b "$ref" --label-b "$ref" --out "$RES/E23/$MODEL/compare_das_tfce_vs_$ref.md"
  else
    echo "== $(date '+%F %T') 3 skipped: no $ref sweep for $MODEL under results_das_tf" >> "$HOSTLOG"
  fi
done
step "4 C sweep, all layers, Cp + Cs, C orthogonal to the DAS-CE E of the same layer" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --tune-pairs 192 --skip-e \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf-ce-bidir --tag das_tfce_C
step "5 2x2 report" scripts/report_2x2.py --e-root "$RES" --e-tag das_tfce_full --c-root "$RES" --c-tag das_tfce_C --model "$MODEL" \
  --out "$RES/E23/$MODEL/report_2x2_das_tfce.md"
echo "== $(date '+%F %T') TFCE CHAIN DONE $MODEL" >> "$HOSTLOG"
date > "$HOSTRES/$MODEL/CHAIN.DONE"

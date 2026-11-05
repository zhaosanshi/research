#!/usr/bin/env bash
# Host side: tfce1_chain.sh MODEL GPU
# Decomposition control asked for by the red team of the ce change (2026-11-05): the same as tfce_chain.sh steps 1-3 but
# WITHOUT --das-bidir, i.e. gold-label CE one-way (base = false statement <- true source, target True only).
#   round 4      : KL to the source continuation, one-way, 8 steps        (das_tf_full)
#   this control : gold-label CE, one-way, 1 step                          (das_tfce1_full)
#   tfce         : gold-label CE, bidirectional, 1 step                    (das_tfce_full)
# Results under /workspace/Truth_Decoupling/results_das_tfce1/{E1,E23}/MODEL.
set -u
MODEL=$1; GPU=$2
ROOT=/workspace/Truth_Decoupling
RES=$ROOT/results_das_tfce1
CODE=$ROOT/review_e25/claude_gpt_review/from_claude/code
HOSTLOG=/data/sanshizhao/Truth_Decoupling/results_das_tfce1/$MODEL/chain.log
mkdir -p "$(dirname "$HOSTLOG")" /data/sanshizhao/Truth_Decoupling/results_das_tfce1/E23/$MODEL
run() {
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed \
    -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -w $CODE truth_dec python3 "$@" >> "$HOSTLOG" 2>&1
}
step() { local name=$1; shift; echo "== $(date '+%F %T') $name $MODEL (GPU $GPU)" >> "$HOSTLOG"; run "$@" || { echo "== $(date '+%F %T') STEP FAILED: $name $MODEL" >> "$HOSTLOG"; exit 1; }; }
step "1 E1 DAS tf-ce (one-way) all layers" scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf \
  --das-anchor prompt --das-tags E --das-pairs 96 --das-all-layers --das-loss ce
step "2 E23 all layers E only" scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf-ce --tag das_tfce1_full
step "3 compare one-way vs bidirectional" scripts/e23_compare_estimators.py --root "$RES" --root-b "$ROOT/results_das_tfce" --model "$MODEL" \
  --tag-a das_tfce1_full --label-a "DAS-CE-oneway" --tag-b das_tfce_full --label-b "DAS-CE-bidir" --out "$RES/E23/$MODEL/compare_oneway_vs_bidir.md"
step "3b compare one-way vs round-4 KL" scripts/e23_compare_estimators.py --root "$RES" --root-b "$ROOT/results_das_tf" --model "$MODEL" \
  --tag-a das_tfce1_full --label-a "DAS-CE-oneway" --tag-b das_tf_full --label-b "DAS-KL" --out "$RES/E23/$MODEL/compare_oneway_vs_kl.md"
echo "== $(date '+%F %T') TFCE1 CHAIN DONE $MODEL" >> "$HOSTLOG"

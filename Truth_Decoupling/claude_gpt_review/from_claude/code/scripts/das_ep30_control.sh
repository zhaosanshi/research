#!/usr/bin/env bash
# das_ep30_control.sh <gpu> <model> <layer> [<layer> ...]
# Optimizer-step control for the estimation-set size experiment (red-team review 2026-11-02, S4): 288 pairs x 10 epochs is
# 3x the gradient steps of 96 pairs x 10 epochs. This trains 96 pairs x 30 epochs (= the same 2880 updates) at the given layers
# with every other setting identical (config E1_directions_tqa_ep30), sweeps those layers on the ORIGINAL 1:2:7 split, and
# compares with the production 96-pair / 10-epoch DAS sweep (same items, same tuning split -> paired CIs). Separate results root
# results_das_tf_ep30; production results are only read. Runs on the HOST (not inside docker).
set -u
GPU="$1"; MODEL="$2"; shift 2
LAYERS="$*"
LTAG="L$(echo "$LAYERS" | tr ' ' '-')"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf_ep30"
OLD="$ROOT/results_das_tf"
LOG="$ROOT_HOST/logs/das_ep30_control_${MODEL}.log"
DONE="$ROOT_HOST/logs/das_ep30_control_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/das_ep30_control_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
mkdir -p "$ROOT_HOST/results_das_tf_ep30/compare"

run() {
  docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$RES" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e CUDA_VISIBLE_DEVICES="$GPU" -w "$ROOT/code/experiments" truth_dec python3 "$@"
}
step() {
  local label="$1"; shift
  echo "== $(date '+%F %T') $label" >> "$LOG"
  if ! run "$@" >> "$LOG" 2>&1; then
    echo "== $(date '+%F %T') FAILED at: $label" >> "$LOG"
    touch "$FAIL"
    exit 1
  fi
}

echo "== $(date '+%F %T') ep30 control start gpu=$GPU model=$MODEL layers=$LAYERS" >> "$LOG"
# shellcheck disable=SC2086
step "E1 DAS seed 42, 96 pairs x 30 epochs at layers $LAYERS" \
  scripts/e1_directions.py --exp E1_directions_tqa_ep30 --model "$MODEL" --direction-prompt tf --das-anchor prompt \
  --das-tags E --das-pairs 96 --das-layers $LAYERS
# shellcheck disable=SC2086
step "E23 at layers $LAYERS, DAS E (30 epochs), original split tune 192 / test 529" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --layers $LAYERS --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf --tag das_tf_ep30
rm -f "$ROOT_HOST/results_das_tf_ep30/compare/${MODEL}_ep30_vs_ep10_${LTAG}.md" "$ROOT_HOST/results_das_tf_ep30/compare/${MODEL}_das_cos_ep30_vs_ep10_${LTAG}".{md,json}
step "paired comparison 30 vs 10 epochs (same items, same tuning split)" \
  scripts/e23_compare_estimators.py --root "$RES" --tag-a das_tf_ep30 --label-a DAS-96-ep30 --root-b "$OLD" --tag-b das_tf_full --label-b DAS-96-ep10 \
  --model "$MODEL" --out "$RES/compare/${MODEL}_ep30_vs_ep10_${LTAG}.md"
# shellcheck disable=SC2086
step "|cos| 30-epoch vs 10-epoch directions" \
  scripts/das_cos_dirs.py --dir-a "$OLD/E1/$MODEL" --dir-b "$RES/E1/$MODEL" --variant tf --layers $LAYERS \
  --out "$RES/compare/${MODEL}_das_cos_ep30_vs_ep10_${LTAG}.json"
echo "== $(date '+%F %T') ep30 control done" >> "$LOG"
touch "$DONE"

#!/usr/bin/env bash
# das_rank_pilot.sh <gpu> <model> <layers...>
# Pilot after the Gemma trace diagnostic (GPT round 5 §5): at the layers where no single DAS direction swaps the first
# token (loss ~4.5-5), does a low-rank SUBSPACE swap? Trains rank-4 and rank-16 DAS (same 96 pairs, forced True/False
# prompt, prompt anchor, single seed) at the given layers only. Training losses and the per-layer summary are the
# output; no evaluation sweep is run here (that needs a reviewed rank-k path in e23_knob_sweep.py).
# Separate results root results_das_tf_rank (the rank is in the file name, so the two ranks share the E1 directory).
set -u
GPU="$1"; MODEL="$2"; shift 2
LAYERS="$*"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf_rank"
LOG="$ROOT_HOST/logs/das_rank_pilot_${MODEL}.log"
DONE="$ROOT_HOST/logs/das_rank_pilot_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/das_rank_pilot_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
mkdir -p "$ROOT_HOST/results_das_tf_rank"

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

echo "== $(date '+%F %T') rank pilot start gpu=$GPU model=$MODEL layers=$LAYERS" >> "$LOG"
for R in 4 16; do
  step "rank $R: E1 DAS at layers $LAYERS (tf prompt, prompt anchor)" \
    scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf --das-anchor prompt \
    --das-tags E --das-pairs 96 --das-rank "$R" --das-layers $LAYERS
  # e1 writes summary.json once per run; keep each rank's copy
  cp "$ROOT_HOST/results_das_tf_rank/E1/$MODEL/summary.json" "$ROOT_HOST/results_das_tf_rank/E1/$MODEL/summary_r${R}.json"
done
echo "== $(date '+%F %T') rank pilot done" >> "$LOG"
touch "$DONE"

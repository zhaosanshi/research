#!/usr/bin/env bash
# das_tf_chain.sh <gpu> <model>
# Review point 1 (E estimator), round-4 plan, steps 2-4 of from_claude/01 §8: one model on one GPU.
#   step 2 : E1 -- DAS at EVERY layer, forced True/False prompt, read/write anchored at the prompt's last token
#   step 3a: E23 -- all layers, DAS E, E conditions only (192 tuning pairs, 529 test pairs)
#   step 3b: E23 -- all layers, mean-difference E, same protocol (the same-code baseline)
#   step 4 : read-layer / write-layer reports for both sweeps
# Runs on the HOST (not inside docker). Results go to the separate root results_das_tf; log and flags in logs/.
set -u
GPU="$1"; MODEL="$2"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf"
LOG="$ROOT_HOST/logs/das_tf_${MODEL}.log"
DONE="$ROOT_HOST/logs/das_tf_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/das_tf_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
mkdir -p "$ROOT_HOST/results_das_tf"

run() {
  docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$RES" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e CUDA_VISIBLE_DEVICES="$GPU" -w "$ROOT/code/experiments" truth_dec python3 "$@"
}
step() {  # step <label> <python args...>; any failure stops the chain loudly
  local label="$1"; shift
  echo "== $(date '+%F %T') $label" >> "$LOG"
  if ! run "$@" >> "$LOG" 2>&1; then
    echo "== $(date '+%F %T') FAILED at: $label" >> "$LOG"
    touch "$FAIL"
    exit 1
  fi
}

echo "== $(date '+%F %T') chain start gpu=$GPU model=$MODEL" >> "$LOG"
step "step 2: E1 DAS all layers (tf prompt, prompt anchor)" \
  scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf --das-anchor prompt \
  --das-tags E --das-pairs 96 --das-all-layers
step "step 3a: E23 all layers, DAS E, E conditions only" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf --tag das_tf_full
step "step 3b: E23 all layers, mean-diff E, E conditions only" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator meandiff --tag meandiff_full
step "step 4a: read/write report, DAS" \
  scripts/report_read_write_layers.py --e1-dir "$RES/E1/$MODEL" \
  --sweep "$RES/E23/$MODEL/knob_sweep_das_tf_full.json" --out "$RES/E23/$MODEL/read_write_das_tf"
step "step 4b: read/write report, mean-diff" \
  scripts/report_read_write_layers.py --e1-dir "$RES/E1/$MODEL" \
  --sweep "$RES/E23/$MODEL/knob_sweep_meandiff_full.json" --out "$RES/E23/$MODEL/read_write_meandiff"
echo "== $(date '+%F %T') chain done" >> "$LOG"
touch "$DONE"

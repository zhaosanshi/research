#!/usr/bin/env bash
# c_das_tf_chain.sh <gpu> <model>
# The C row of the 2x2 under the new E (GPT review round 5, section 1: the E direction enters C through the
# orthogonalisation step, so the old C results cannot be combined with the DAS E).
#   step C: E23 -- all layers, C conditions only (Cp = prompt-contrast, Cs = self-report), each layer's C
#           orthogonalised against the DAS (forced True/False prompt, prompt anchor) E direction at that layer.
#           Same protocol as das_tf_chain.sh step 3a (192 tuning pairs, 529 test pairs); the E conditions are
#           not repeated here because that step already ran them.
# Runs on the HOST (not inside docker). Output: results_das_tf/E23/<model>/knob_sweep_das_tf_C.{json,md,npz}.
set -u
GPU="$1"; MODEL="$2"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf"
LOG="$ROOT_HOST/logs/c_das_tf_${MODEL}.log"
DONE="$ROOT_HOST/logs/c_das_tf_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/c_das_tf_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"

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

echo "== $(date '+%F %T') C chain start gpu=$GPU model=$MODEL" >> "$LOG"
step "step C: E23 all layers, C conditions only (Cp + Cs), C orthogonal to the DAS-tf E of the same layer" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --tune-pairs 192 --skip-e \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf --tag das_tf_C
echo "== $(date '+%F %T') C chain done" >> "$LOG"
touch "$DONE"

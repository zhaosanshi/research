#!/usr/bin/env bash
# das_est_size_chain.sh <gpu> <model>
# Estimation-set size experiment (赵磊 2026-11-02: under the 1:2:7 split DAS trains on 96 pairs -- is it data-starved?).
#   split: 96 est : 192 tune : 529 test  ->  288 est (calib+dev + the old 192 tuning pairs) : 192 tune (redrawn from the old test
#          pairs) : 337 test.  Both estimators (DAS and mean-difference) see the same 288 pairs. Nothing else changes.
#   step 2 : E1 -- DAS at EVERY layer on 288 pairs (tf prompt, prompt anchor; identical to das_tf_chain.sh otherwise)
#   step 3a: E23 -- all layers, DAS E, new split
#   step 3b: E23 -- all layers, mean-difference E, new split
#   step 4 : report: the old 96-pair sweeps re-cut to the new split; four sweeps on identical tuning/test pairs, paired CIs
#   step 5 : |cos| between the 96-pair and 288-pair DAS directions per layer
# Runs on the HOST (not inside docker). Results go to the separate root results_das_tf_r288; log and flags in logs/.
set -u
GPU="$1"; MODEL="$2"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf_r288"
OLD="$ROOT/results_das_tf"
LOG="$ROOT_HOST/logs/das_est_size_${MODEL}.log"
DONE="$ROOT_HOST/logs/das_est_size_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/das_est_size_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
mkdir -p "$ROOT_HOST/results_das_tf_r288/compare"

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
step "step 2: E1 DAS all layers, 288 estimation pairs (tf prompt, prompt anchor)" \
  scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf --das-anchor prompt \
  --das-tags E --est-extra-pairs 192 --das-pairs 288 --das-all-layers
step "step 3a: E23 all layers, DAS E, 288 estimation pairs, tune 192 / test 337" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --est-extra-pairs 192 --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf --tag das_tf_full
step "step 3b: E23 all layers, mean-diff E, 288 estimation pairs, same split" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --est-extra-pairs 192 --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator meandiff --tag meandiff_full
rm -f "$ROOT_HOST/results_das_tf_r288/compare/${MODEL}_est_size".{md,json} "$ROOT_HOST/results_das_tf_r288/compare/${MODEL}_das_cos_96_vs_288".{md,json}   # the reports refuse to overwrite
step "step 4: report, four sweeps on the same tuning/test pairs" \
  scripts/report_est_size.py --root-old "$OLD" --root-new "$RES" --model "$MODEL" --out "$RES/compare/${MODEL}_est_size.md"
step "step 5: |cos| between 96-pair and 288-pair DAS directions" \
  scripts/das_cos_dirs.py --dir-a "$OLD/E1/$MODEL" --dir-b "$RES/E1/$MODEL" --variant tf --out "$RES/compare/${MODEL}_das_cos_96_vs_288.json"
echo "== $(date '+%F %T') chain done" >> "$LOG"
touch "$DONE"

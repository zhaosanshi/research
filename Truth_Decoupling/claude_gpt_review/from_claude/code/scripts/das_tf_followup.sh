#!/usr/bin/env bash
# das_tf_followup.sh <gpu> <model> <band_lo> <band_hi> <anchor_ablation 0|1>
# Follow-up to das_tf_chain.sh after the round-4 single-layer results. Same reviewed code, no new experiment code.
#   step 5a/5b: E23 on the layer band [lo, hi] PLUS every consecutive 3-layer window inside it, DAS E and mean-diff E,
#               same protocol (192 tuning pairs / 529 test pairs, E conditions only). Reason: the previously chosen
#               operating point was a 3-layer window (Llama E_L11-12-13_a3), the round-4 sweep had single layers only.
#   step 6a-6c (anchor_ablation=1): retrain DAS at every layer with the OLD read/write anchor (--das-anchor current,
#               variant tf-cur) in a SEPARATE results root, then the same all-layer single-layer E23 and read/write
#               report. One variable changed vs the main run: the anchor. Tells which of the two round-1/2 fixes mattered.
# Runs on the HOST (not inside docker). Log and flags in logs/.
set -u
GPU="$1"; MODEL="$2"; LO="$3"; HI="$4"; ANCHOR="${5:-0}"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf"
RES_CUR="$ROOT/results_das_tf_cur"
LOG="$ROOT_HOST/logs/das_tf_followup_${MODEL}.log"
DONE="$ROOT_HOST/logs/das_tf_followup_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/das_tf_followup_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
mkdir -p "$ROOT_HOST/results_das_tf" "$ROOT_HOST/results_das_tf_cur"

run() {  # run <results_root> <python args...>
  local resdir="$1"; shift
  docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$resdir" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e CUDA_VISIBLE_DEVICES="$GPU" -w "$ROOT/code/experiments" truth_dec python3 "$@"
}
step() {  # step <label> <results_root> <python args...>; any failure stops the chain loudly
  local label="$1"; local resdir="$2"; shift 2
  echo "== $(date '+%F %T') $label" >> "$LOG"
  if ! run "$resdir" "$@" >> "$LOG" 2>&1; then
    echo "== $(date '+%F %T') FAILED at: $label" >> "$LOG"
    touch "$FAIL"
    exit 1
  fi
}

BAND=$(seq -s ' ' "$LO" "$HI")
SPANS=""
for ((l = LO; l + 2 <= HI; l++)); do SPANS="$SPANS $l,$((l + 1)),$((l + 2))"; done

echo "== $(date '+%F %T') follow-up start gpu=$GPU model=$MODEL band=$LO-$HI spans=[$SPANS ] anchor_ablation=$ANCHOR" >> "$LOG"
step "step 5a: E23 band $LO-$HI + 3-layer windows, DAS E" "$RES" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --layers $BAND --e-spans $SPANS --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator das --das-dir "$RES/E1/$MODEL" --das-variant tf --tag das_tf_spans
step "step 5b: E23 band $LO-$HI + 3-layer windows, mean-diff E" "$RES" \
  scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --layers $BAND --e-spans $SPANS --tune-pairs 192 --skip-cp --no-cs \
  --e-estimator meandiff --tag meandiff_spans
if [ "$ANCHOR" = "1" ]; then
  step "step 6a: E1 DAS all layers, tf prompt, OLD anchor (current) -> variant tf-cur" "$RES_CUR" \
    scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf --das-anchor current \
    --das-tags E --das-pairs 96 --das-all-layers
  step "step 6b: E23 all layers, DAS(tf-cur) E, E conditions only" "$RES_CUR" \
    scripts/e23_knob_sweep.py --exp E2b_tqa --model "$MODEL" --all-layers --tune-pairs 192 --skip-cp --no-cs \
    --e-estimator das --das-dir "$RES_CUR/E1/$MODEL" --das-variant tf-cur --tag das_tfcur_full
  step "step 6c: read/write report, DAS(tf-cur)" "$RES_CUR" \
    scripts/report_read_write_layers.py --e1-dir "$RES_CUR/E1/$MODEL" \
    --sweep "$RES_CUR/E23/$MODEL/knob_sweep_das_tfcur_full.json" --out "$RES_CUR/E23/$MODEL/read_write_das_tfcur"
fi
echo "== $(date '+%F %T') follow-up done" >> "$LOG"
touch "$DONE"

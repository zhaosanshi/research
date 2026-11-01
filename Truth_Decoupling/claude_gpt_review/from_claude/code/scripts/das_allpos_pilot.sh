#!/usr/bin/env bash
# das_allpos_pilot.sh <gpu> <model> <layers...>
# Second Gemma pilot (one variable: the WRITE POSITIONS). The trace diagnostic showed that no last-token direction
# swaps the first token before layer 23 (DAS loss 3.6-5.0), and the rank-4/16 pilot moved that only slightly; the
# loss drops sharply right after layer 23, Gemma-3's fourth full-attention layer. Hypothesis: before that layer the
# truth evidence sits on the statement tokens, not on the last token. Test: train rank-1 DAS that clamps EVERY base
# position's projection to the source's last-token projection (--das-positions all, the CDAS one-to-many form,
# already in core.directions) at the given layers. If the all-positions swap fits where the last-token swap did not,
# the evidence is distributed over positions and a last-token E cannot reach it. Training losses are the output; no
# evaluation sweep is run (an all-positions amplify needs a per-position centre, to be designed and reviewed first).
# Separate results root results_das_tf_allpos; variant suffix tf-allpos is derived from the flags.
set -u
GPU="$1"; MODEL="$2"; shift 2
LAYERS="$*"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf_allpos"
LOG="$ROOT_HOST/logs/das_allpos_pilot_${MODEL}.log"
DONE="$ROOT_HOST/logs/das_allpos_pilot_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/das_allpos_pilot_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
mkdir -p "$ROOT_HOST/results_das_tf_allpos"

run() {
  docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$RES" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e CUDA_VISIBLE_DEVICES="$GPU" -w "$ROOT/code/experiments" truth_dec python3 "$@"
}
echo "== $(date '+%F %T') all-positions pilot start gpu=$GPU model=$MODEL layers=$LAYERS" >> "$LOG"
if run scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf --das-anchor prompt \
     --das-tags E --das-pairs 96 --das-positions all --das-layers $LAYERS >> "$LOG" 2>&1; then
  echo "== $(date '+%F %T') all-positions pilot done" >> "$LOG"; touch "$DONE"
else
  echo "== $(date '+%F %T') all-positions pilot FAILED" >> "$LOG"; touch "$FAIL"; exit 1
fi

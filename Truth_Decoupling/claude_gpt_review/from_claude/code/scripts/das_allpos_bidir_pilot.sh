#!/usr/bin/env bash
# das_allpos_bidir_pilot.sh <gpu> <model> <layers...>
# Third Gemma pilot (one variable: the OBJECTIVE). The one-directional all-positions DAS (das_allpos_pilot.sh) fitted
# loss 0.11-0.19 at layers 18-24 but learned a distributed "say True" lever: after the swap every statement answers
# True whatever the source, and the direction reads truth at chance on the tokens. CDAS closes that shortcut with the
# bidirectional objective (D+: base<-source must match the source; D-: source<-base must match the base), which is
# --das-bidir in core.directions. Same 96 pairs, forced True/False prompt, prompt anchor, rank 1, all positions.
# Variant suffix derives to tf-bidir-allpos; same results root as the one-directional pilot (file names differ).
set -u
GPU="$1"; MODEL="$2"; shift 2
LAYERS="$*"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf_allpos"
LOG="$ROOT_HOST/logs/das_allpos_bidir_pilot_${MODEL}.log"
DONE="$ROOT_HOST/logs/das_allpos_bidir_pilot_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/das_allpos_bidir_pilot_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"

run() {
  docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$RES" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e CUDA_VISIBLE_DEVICES="$GPU" -w "$ROOT/code/experiments" truth_dec python3 "$@"
}
echo "== $(date '+%F %T') all-positions BIDIRECTIONAL pilot start gpu=$GPU model=$MODEL layers=$LAYERS" >> "$LOG"
if run scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf --das-anchor prompt \
     --das-tags E --das-pairs 96 --das-positions all --das-bidir --das-layers $LAYERS >> "$LOG" 2>&1; then
  cp "$ROOT_HOST/results_das_tf_allpos/E1/$MODEL/summary.json" "$ROOT_HOST/results_das_tf_allpos/E1/$MODEL/summary_bidir.json"
  echo "== $(date '+%F %T') pilot done" >> "$LOG"; touch "$DONE"
else
  echo "== $(date '+%F %T') pilot FAILED" >> "$LOG"; touch "$FAIL"; exit 1
fi

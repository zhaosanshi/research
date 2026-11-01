#!/usr/bin/env bash
# diag_trace_allpos_run.sh <gpu> <model> <layer> [<layer> ...]
# The trace diagnostic for the ALL-POSITIONS DAS directions (das_allpos_pilot.sh output, variant tf-allpos, root
# results_das_tf_allpos): swap written at every position (as trained), both directions (false<-true and true<-false),
# on all 96 training pairs and all 192 tuning pairs; token-level readout of the direction (last token vs mean / max
# over positions). No sweep cross-check (no sweep exists for these directions). The amplify rows use the last-token
# centre at every position and are exploratory only.
# Output: results_das_tf_allpos/diag/<model>/trace_direction_allpos_L<l1>_L<l2>...{json,md}; log and flags in logs/.
set -u
GPU="$1"; MODEL="$2"; shift 2
LAYERS="$*"
VARIANT="${VARIANT:-tf-allpos}"            # e.g. VARIANT=tf-bidir-allpos for the bidirectional pilot's directions
TAG="trace_direction_${VARIANT//-/_}$(printf '_L%s' $LAYERS)"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf_allpos"
LOG="$ROOT_HOST/logs/diag_${TAG}_${MODEL}.log"
DONE="$ROOT_HOST/logs/diag_${TAG}_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/diag_${TAG}_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
echo "== $(date '+%F %T') diag (all positions) start gpu=$GPU model=$MODEL layers=$LAYERS" >> "$LOG"
if docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$RES" -e TD_DATA_DIR="$ROOT/code/data/processed" \
     -e CUDA_VISIBLE_DEVICES="$GPU" -w "$ROOT/code/experiments" truth_dec python3 scripts/diag_trace_direction.py \
     --model "$MODEL" --layers $LAYERS --das-dir "$RES/E1/$MODEL" --das-variant "$VARIANT" --positions all \
     --out "$RES/diag/$MODEL/$TAG" >> "$LOG" 2>&1; then
  echo "== $(date '+%F %T') diag done" >> "$LOG"; touch "$DONE"
else
  echo "== $(date '+%F %T') diag FAILED" >> "$LOG"; touch "$FAIL"; exit 1
fi

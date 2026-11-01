#!/usr/bin/env bash
# diag_trace_run.sh <gpu> <model> <layer> [<layer> ...]
# Runs scripts/diag_trace_direction.py on the HOST via docker (same environment as the chains): the DAS-tf directions
# of results_das_tf/E1/<model>, cross-checked against that model's knob_sweep_das_tf_full.json.
# Output: results_das_tf/diag/<model>/trace_direction_L<l1>_L<l2>...{json,md}; log and .DONE/.FAIL in logs/.
set -u
GPU="$1"; MODEL="$2"; shift 2
LAYERS="$*"
TAG="trace_direction$(printf '_L%s' $LAYERS)"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf"
LOG="$ROOT_HOST/logs/diag_${TAG}_${MODEL}.log"
DONE="$ROOT_HOST/logs/diag_${TAG}_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/diag_${TAG}_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
echo "== $(date '+%F %T') diag start gpu=$GPU model=$MODEL layers=$LAYERS" >> "$LOG"
if docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$RES" -e TD_DATA_DIR="$ROOT/code/data/processed" \
     -e CUDA_VISIBLE_DEVICES="$GPU" -w "$ROOT/code/experiments" truth_dec python3 scripts/diag_trace_direction.py \
     --model "$MODEL" --layers $LAYERS --das-dir "$RES/E1/$MODEL" --das-variant tf \
     --sweep-json "$RES/E23/$MODEL/knob_sweep_das_tf_full.json" --out "$RES/diag/$MODEL/$TAG" >> "$LOG" 2>&1; then
  echo "== $(date '+%F %T') diag done" >> "$LOG"; touch "$DONE"
else
  echo "== $(date '+%F %T') diag FAILED" >> "$LOG"; touch "$FAIL"; exit 1
fi

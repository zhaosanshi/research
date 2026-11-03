#!/usr/bin/env bash
# das_seed_check.sh <gpu> <model> <layer> [<layer> ...]
# Diagnostic for the estimation-set size question (赵磊 2026-11-02): is the 96-pair DAS direction stable across random inits?
# Trains seeds 43 and 44 at the given layers with the production settings (tf prompt, prompt anchor, 96 pairs) under a SEPARATE
# results root (results_das_tf_seedchk), so the production directions (seed 42, results_das_tf) are not touched, then reports
# |cos| between seed 42 and seeds 43/44 and between 43 and 44. Runs on the HOST (not inside docker).
set -u
GPU="$1"; MODEL="$2"; shift 2
LAYERS="$*"
LTAG="L$(echo "$LAYERS" | tr ' ' '-')"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf_seedchk"
OLD="$ROOT/results_das_tf"
LOG="$ROOT_HOST/logs/das_seed_check_${MODEL}.log"
DONE="$ROOT_HOST/logs/das_seed_check_${MODEL}.DONE"
FAIL="$ROOT_HOST/logs/das_seed_check_${MODEL}.FAIL"
rm -f "$DONE" "$FAIL"
mkdir -p "$ROOT_HOST/results_das_tf_seedchk/compare"

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

echo "== $(date '+%F %T') seed check start gpu=$GPU model=$MODEL layers=$LAYERS" >> "$LOG"
# shellcheck disable=SC2086
step "E1 DAS seeds 43 44 at layers $LAYERS (96 pairs)" \
  scripts/e1_directions.py --exp E1_directions_tqa --model "$MODEL" --direction-prompt tf --das-anchor prompt \
  --das-tags E --das-pairs 96 --das-layers $LAYERS --das-seeds 43 44
rm -f "$ROOT_HOST/results_das_tf_seedchk/compare/${MODEL}_seed_cos_${LTAG}".{md,json}   # the report refuses to overwrite
# shellcheck disable=SC2086
step "|cos| seed 42 (production) vs seeds 43/44" \
  scripts/das_cos_dirs.py --dir-a "$OLD/E1/$MODEL" --dir-b "$RES/E1/$MODEL" --variant tf --layers $LAYERS \
  --out "$RES/compare/${MODEL}_seed_cos_${LTAG}.json"
echo "== $(date '+%F %T') seed check done" >> "$LOG"
touch "$DONE"

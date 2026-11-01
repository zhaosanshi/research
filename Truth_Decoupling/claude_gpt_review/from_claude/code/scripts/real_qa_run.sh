#!/usr/bin/env bash
# real_qa_run.sh <gpu> <model> generate|judge [limit]
# Real question answering under the already-reported E / C configurations (GPT review round 8). Runs on the HOST via
# docker exec, like the other chains. The configurations are FIXED here (the ones reported in 汇总.md section 4):
#   llama3.1-8b : E = DAS-tf window 10-11-12 alpha 2 ; C = prompt-contrast (Cp) layer 28 alpha 2 towards abstention
#   qwen3-8b    : E = DAS-tf layer 18 alpha 3        ; C = self-report (Cs) layer 30 alpha 2 towards abstention
# generate: GPU; judge: API only (DeepSeek key read from a chmod-600 file through DEEPSEEK_API_KEY_FILE, never on the
# command line). With [limit] the tag becomes pilot<limit> and the questions / outputs / audit are printed in full.
set -u
GPU="$1"; MODEL="$2"; PHASE="$3"; LIMIT="${4:-}"
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
RES="$ROOT/results_das_tf"
case "$MODEL" in
  llama3.1-8b) CFG="--e-layers 10 11 12 --e-alpha 2 --c-kind Cp --c-layer 28 --c-alpha 2" ;;
  qwen3-8b)    CFG="--e-layers 18 --e-alpha 3 --c-kind Cs --c-layer 30 --c-alpha 2" ;;
  *) echo "no fixed configuration for model $MODEL" >&2; exit 2 ;;
esac
TAG="${TAG:-fixed}"
if [ -n "$LIMIT" ]; then TAG="pilot${LIMIT}"; fi
LOG="$ROOT_HOST/logs/real_qa_${PHASE}_${MODEL}_${TAG}.log"
DONE="$LOG.DONE"; FAIL="$LOG.FAIL"
rm -f "$DONE" "$FAIL"
echo "== $(date '+%F %T') real_qa $PHASE start gpu=$GPU model=$MODEL tag=$TAG cfg=[$CFG]" >> "$LOG"
if docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$RES" -e TD_DATA_DIR="$ROOT/code/data/processed" \
     -e CUDA_VISIBLE_DEVICES="$GPU" -e PYTHONUNBUFFERED=1 -e DEEPSEEK_API_KEY_FILE="$ROOT/logs/.deepseek_key" \
     -w "$ROOT/code/experiments" truth_dec python3 scripts/e24_real_qa.py --model "$MODEL" --phase "$PHASE" $CFG --tag "$TAG" \
     ${LIMIT:+--limit "$LIMIT"} >> "$LOG" 2>&1; then
  echo "== $(date '+%F %T') real_qa $PHASE done" >> "$LOG"; touch "$DONE"
else
  echo "== $(date '+%F %T') real_qa $PHASE FAILED" >> "$LOG"; touch "$FAIL"; exit 1
fi

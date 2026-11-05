#!/bin/bash
# job_doorsearch.sh -- look for ANY open first-token door in Qwen3-8B and gemma-3-4b-it (English, plain contaminant):
#   (a) Qwen3 with thinking ENABLED (no --no-think): X20k m00/m03
#   (b) role-formatted text pasted by the real user: U20k_m00_ptr / U20k_m00_pfk
#   (c) 64k context: X64k_m03 / X64k_c03
#   (d) other role words in the contaminant: Human, USER  (X20k_m03), forced continuation with the same word
# 3 seeds each, GPU 0.  Output results/doorsearch_<model>_<tag>.jsonl
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
D="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=en -e CONTAM_STYLE=plain -e CUDA_VISIBLE_DEVICES=0 -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
probe() {  # model extra-env extra-args tag conds...
  local m=$1 env=$2 extra=$3 tag=$4; shift 4
  for s in 1 2 3; do
    $D $env $IMG python turn_boundary_probe.py --model $MD/$m $extra $FILL --device auto --max-forced 48 --conditions "$@" \
        --filler-seed $s --out results/doorsearch_${m}_$tag.jsonl >> results/doorsearch_${m}_$tag.log 2>&1 || log "FAILED $m $tag seed $s"
  done
  log "$m $tag done ($(wc -l < results/doorsearch_${m}_$tag.jsonl) records)"
}
log "door search start"
probe Qwen3-8B "" "--force-first user" think X20k_m00 X20k_m03
probe Qwen3-8B "" "--no-think --force-first user" paste U20k_m00_ptr U20k_m00_pfk
probe gemma-3-4b-it "" "--force-first user" paste U20k_m00_ptr U20k_m00_pfk
probe Qwen3-8B "" "--no-think --force-first user" len64 X64k_m03 X64k_c03
probe gemma-3-4b-it "" "--force-first user" len64 X64k_m03 X64k_c03
probe Qwen3-8B "-e CONTAM_ROLE_WORD=Human" "--no-think --force-first Human user" human X20k_m03
probe gemma-3-4b-it "-e CONTAM_ROLE_WORD=Human" "--force-first Human user" human X20k_m03
probe Qwen3-8B "-e CONTAM_ROLE_WORD=USER" "--no-think --force-first USER user" upper X20k_m03
probe gemma-3-4b-it "-e CONTAM_ROLE_WORD=USER" "--force-first USER user" upper X20k_m03
log "DOORSEARCH DONE"

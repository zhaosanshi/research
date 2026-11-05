#!/bin/bash
# job_g3.sh -- GPU 3 queue (2026-10-23): (A) gemma-3-27b-it coherence smoke (sdpa vs eager), then
# (B) own-voice meta probes: each model's slot_start X20k meta cells with filler it generated itself.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_g3 'bash job_g3.sh 2>&1 | tee results/job_g3.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} -e CUDA_VISIBLE_DEVICES=3"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
XM="X20k_m00 X20k_m10 X20k_m13"

log "A: gemma-3-27b-it smoke (sdpa vs eager; short + ~10k prompt)"
$DOCKER $IMG python gemma27_smoke.py > results/gemma27_smoke.log 2>&1 || log "smoke FAILED -- see results/gemma27_smoke.log"
log "A done"

log "B: own-voice meta probes (Qwen3-8B, Qwen3-14B, gemma-3-4b-it, gemma-3-12b-it)"
for m in Qwen3-8B Qwen3-14B gemma-3-4b-it gemma-3-12b-it; do
  nt=""; case $m in Qwen3*) nt="--no-think";; esac
  for s in 1 2 3 4 5; do
    $DOCKER $IMG python turn_boundary_probe.py --model $M/$m $nt --filler-meta results/filler_meta_$m.jsonl \
        --filler-coding results/filler_coding.jsonl --force-first user --max-forced 48 --conditions $XM \
        --filler-seed $s --out results/xmodel_ownvoice_$m.jsonl >> results/xmodel_ownvoice_$m.log 2>&1 \
        || { log "$m seed $s FAILED -- see results/xmodel_ownvoice_$m.log"; exit 1; }
  done
  log "own-voice $m done ($(wc -l < results/xmodel_ownvoice_$m.jsonl) records)"
done
log "GPU3 QUEUE DONE"

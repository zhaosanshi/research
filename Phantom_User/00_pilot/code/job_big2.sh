#!/bin/bash
# job_big2.sh -- redo of the two big models after the torch._native Triton RoPE bug fix (2026-10-23).
#   A (GPU 3):    wait for the own-voice queue (job_g3.sh) and the 8B spot check, then gemma-3-27b-it on ONE GPU.
#   B (GPUs 0,1): wait for the 70B probe (job_70b.sh), then Qwen3-32B with device_map auto (now correct).
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_big2 'bash job_big2.sh 2>&1 | tee results/job_big2.log'
# Waiter patterns name other scripts' files, never this script's own command line (see the GPU server pitfalls).
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-meta results/filler_meta.jsonl --filler-coding results/filler_coding.jsonl"
X="X20k_m00 X20k_m10 X20k_m13 X20k_c10 X20k_c13"
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 30; done; }

(
  log "A: waiting for job_g3.sh and the 8B spot check to free GPU 3"
  wait_gone 'bash job_g3.sh'
  wait_gone 'spotcheck_router_off'
  log "A: gemma-3-27b-it on GPU 3 (single device), 5 seeds"
  for s in 1 2 3 4 5; do
    $DOCKER -e CUDA_VISIBLE_DEVICES=3 $IMG python turn_boundary_probe.py --model $M/gemma-3-27b-it $FILL \
        --force-first user --max-forced 48 --conditions $X --filler-seed $s \
        --out results/xmodel_v1_gemma-3-27b-it.jsonl >> results/xmodel_v1_gemma-3-27b-it.log 2>&1 \
        || { log "A: gemma-3-27b-it seed $s FAILED -- see results/xmodel_v1_gemma-3-27b-it.log"; exit 1; }
    log "A: gemma-3-27b-it seed $s done"
  done
  log "A done"
) &

(
  log "B: waiting for job_70b.sh to free GPUs 0,1,2"
  wait_gone 'bash job_70b.sh'
  log "B: Qwen3-32B on GPUs 0,1 (device_map auto), 5 seeds"
  for s in 1 2 3 4 5; do
    $DOCKER -e CUDA_VISIBLE_DEVICES=0,1 $IMG python turn_boundary_probe.py --model $M/Qwen3-32B --device auto --no-think $FILL \
        --force-first user --max-forced 48 --conditions $X --filler-seed $s \
        --out results/xmodel_v1_Qwen3-32B.jsonl >> results/xmodel_v1_Qwen3-32B.log 2>&1 \
        || { log "B: Qwen3-32B seed $s FAILED -- see results/xmodel_v1_Qwen3-32B.log"; exit 1; }
    log "B: Qwen3-32B seed $s done"
  done
  log "B done"
) &

wait
log "BIG2 DONE"

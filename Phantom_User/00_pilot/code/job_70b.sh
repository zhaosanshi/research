#!/bin/bash
# job_70b.sh -- Llama-3.3-70B-Instruct X20k slot_start probe (Llama filler, 5 seeds) on GPUs 0,1,2 (device_map auto).
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus70b 'bash job_70b.sh 2>&1 | tee results/job_70b.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} -e CUDA_VISIBLE_DEVICES=0,1,2"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}/Llama-3.3-70B-Instruct
FILL="--filler-meta results/filler_meta.jsonl --filler-coding results/filler_coding.jsonl"
X="X20k_m00 X20k_m10 X20k_m13 X20k_c10 X20k_c13"
OUT=results/xmodel_v1_Llama-3.3-70B-Instruct.jsonl

log "70B X20k probe start (GPUs 0,1,2)"
for s in 1 2 3 4 5; do
  log "seed $s"
  $DOCKER $IMG python turn_boundary_probe.py --model $M --device auto $FILL --force-first user --max-forced 48 \
      --conditions $X --filler-seed $s --out $OUT >> results/xmodel_v1_Llama-3.3-70B-Instruct.log 2>&1 \
      || { log "seed $s FAILED -- see results/xmodel_v1_Llama-3.3-70B-Instruct.log"; exit 1; }
  log "seed $s done ($(wc -l < $OUT) records so far)"
done
log "70B PROBE DONE"

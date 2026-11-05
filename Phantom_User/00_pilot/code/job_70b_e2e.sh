#!/bin/bash
# job_70b_e2e.sh -- Llama-3.3-70B-Instruct temperature-1 sampling at slot_start (no forcing), all 4 GPUs,
# same cells/flags as the small-model xe2e runs: 20k tokens, 5 orders, 20 000 first-token draws + 32 full answers each.
#   m13 = self-talk + wrapper + contamination k=3,  m10 = self-talk + wrapper clean,  c13 = coding + wrapper + contamination.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus70e 'bash job_70b_e2e.sh 2>&1 | tee results/job_70b_e2e.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} -e CUDA_VISIBLE_DEVICES=0,1,2,3"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}/Llama-3.3-70B-Instruct
E2E="python e2e_sample.py --model $M --device auto --filler-meta results/filler_meta.jsonl --filler-coding results/filler_coding.jsonl --target-tokens 20000 --seeds 1 2 3 4 5 --n-first 20000 --n-full 32 --batch 8 --max-new 200 --max-hit-cont 3"

run_cell() {  # tag kind wrapper contam
  log "cell $1 start"
  $DOCKER $IMG $E2E --kind $2 --wrapper $3 --contam-k $4 --tag xe2e_70B_$1 --out results/e2e_x_Llama-3.3-70B-Instruct_$1.jsonl \
      >> results/e2e_x_Llama-3.3-70B-Instruct_$1.log 2>&1 || { log "cell $1 FAILED -- see results/e2e_x_Llama-3.3-70B-Instruct_$1.log"; exit 1; }
  log "cell $1 done"
}

run_cell m13 meta 1 3
run_cell m10 meta 1 0
run_cell c13 coding 1 3
log "70B E2E DONE"

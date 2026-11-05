#!/bin/bash
# job_heads.sh -- mechanism block, probe 2: per-head direct logit attribution + attention patterns + ablations
# at the slot, 4 cross cells x 3 filler orders, instruct vs base, Llama vs Qwen, one model per GPU.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_heads 'bash job_heads.sh 2>&1 | tee results/job_heads.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
COMMON="--filler-meta results/filler_meta.jsonl --filler-coding results/filler_coding.jsonl --conditions X20k_m10 X20k_m13 X20k_c10 X20k_c13 --seeds 1 2 3"

run() {  # gpu name model-dir tokenizer-dir extra-flags
  local gpu=$1 name=$2 mdir=$3 tdir=$4; shift 4
  log "start $name on GPU $gpu"
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpu $IMG python head_attrib.py --model $M/$mdir --tokenizer $M/$tdir $COMMON "$@" \
      --tag $name --out results/heads_v1_$name.jsonl > results/heads_v1_$name.log 2>&1 \
      && log "done $name ($(wc -l < results/heads_v1_$name.jsonl) records)" || log "FAILED $name -- see results/heads_v1_$name.log"
}

run 0 Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct &
run 1 Llama-3.1-8B-Base Llama-3.1-8B Llama-3.1-8B-Instruct &
run 2 Qwen3-8B Qwen3-8B Qwen3-8B --no-think &
run 3 Qwen3-8B-Base Qwen3-8B-Base Qwen3-8B --no-think &
wait
log "HEADS DONE"

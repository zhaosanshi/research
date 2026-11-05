#!/bin/bash
# job_followup.sh -- after job_heads.sh: (A) induction/previous-token scores of every head, 4 models (seconds each);
# (B) contamination variants (which token before 'user', which role word, wrapper or not) on X20k_m13 contexts.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_follow 'bash job_followup.sh 2>&1 | tee results/job_followup.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-meta results/filler_meta.jsonl --filler-coding results/filler_coding.jsonl"
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 20; done; }

log "waiting for job_heads.sh"
wait_gone 'bash job_heads.sh'

run() {  # gpu name model-dir tokenizer-dir extra-flags
  local gpu=$1 name=$2 mdir=$3 tdir=$4; shift 4
  log "GPU $gpu: induction test $name"
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpu $IMG python induction_test.py --model $M/$mdir --tokenizer $M/$tdir \
      --out results/induction_$name.json > results/induction_$name.log 2>&1 || log "FAILED induction $name"
  log "GPU $gpu: contamination variants $name"
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpu $IMG python contam_variants.py --model $M/$mdir --tokenizer $M/$tdir $FILL "$@" \
      --seeds 1 2 3 --out results/contam_variants_$name.jsonl > results/contam_variants_$name.log 2>&1 \
      && log "done $name" || log "FAILED variants $name -- see results/contam_variants_$name.log"
}

run 0 Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct &
run 1 Llama-3.1-8B-Base Llama-3.1-8B Llama-3.1-8B-Instruct &
run 2 Qwen3-8B Qwen3-8B Qwen3-8B --no-think &
run 3 Qwen3-8B-Base Qwen3-8B-Base Qwen3-8B --no-think &
wait
log "FOLLOWUP DONE"

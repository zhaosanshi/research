#!/bin/bash
# job_en_plain_xmodel.sh -- paper setting (English, wrapper=0 cells, plain contaminant CONTAM_STYLE=plain) across models,
# plus high-volume first-token sampling on Llama-3.1-8B so that spontaneous fabrications at ~1e-6 per slot are observed
# rather than only inferred (n-first 1M per context = 5M draws per cell; the draws come from one forward pass each).
# Queue L (GPUs 0,1,2): 70B probe -> Qwen3-32B probe (GPUs 0,1) -> 70B e2e c03 -> m03.
# Queue S (GPU 3): small-model probes -> Llama-8B 5M-draw sampling m03, c03 -> gemma-3-27b-it probe -> Qwen3-8B / gemma-4b e2e m03.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_plain_x 'bash job_en_plain_xmodel.sh 2>&1 | tee results/job_en_plain_xmodel.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=en -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
M8=$MD/Llama-3.1-8B-Instruct
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
X="X20k_c00 X20k_c03 X20k_m00 X20k_m03"
E2E_FLAGS="--target-tokens 20000 --seeds 1 2 3 4 5 --n-full 32 --batch 8 --max-new 200 --max-hit-cont 3 --wrapper 0"

nt() { case ${1##*/} in Qwen3*) echo "--no-think";; esac; }

probe() {  # gpus model out-name seeds...
  local gpus=$1 m=$2 name=$3; shift 3
  for s in "$@"; do
    $DOCKER -e CUDA_VISIBLE_DEVICES=$gpus $IMG python turn_boundary_probe.py --model $m $(nt $m) $FILL --device auto \
        --force-first user --max-forced 48 --conditions $X --filler-seed $s --out results/$name.jsonl \
        >> results/$name.log 2>&1 || log "FAILED $name seed $s -- see results/$name.log"
  done
  log "$name done ($(wc -l < results/$name.jsonl 2>/dev/null) records)"
}
e2e() {  # gpus model cell n-first
  local gpus=$1 m=$2 cell=$3 nfirst=$4 kind contam
  local out=results/en_plain_e2e_x_${m##*/}_$cell.jsonl
  case $cell in m*) kind=meta;; c*) kind=coding;; esac
  contam=${cell:2:1}
  log "e2e ${m##*/} $cell (n-first $nfirst) start"
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpus $IMG python e2e_sample.py --model $m $(nt $m) --device auto $FILL $E2E_FLAGS \
      --n-first $nfirst --kind $kind --contam-k $contam --tag en_plain_x_${m##*/}_$cell --out $out > ${out%.jsonl}.log 2>&1 \
      && log "e2e ${m##*/} $cell done" || log "FAILED e2e ${m##*/} $cell -- see ${out%.jsonl}.log"
}

for f in results/filler_en_coding.jsonl results/filler_en_general.jsonl; do [ -s "$f" ] || { log "missing $f -- abort"; exit 1; }; done
log "plain cross-model start"
( probe 0,1,2 $MD/Llama-3.3-70B-Instruct en_plain_xmodel_Llama-3.3-70B-Instruct 1 2 3 4 5
  probe 0,1 $MD/Qwen3-32B en_plain_xmodel_Qwen3-32B 1 2 3 4 5
  e2e 0,1,2 $MD/Llama-3.3-70B-Instruct c03 20000
  e2e 0,1,2 $MD/Llama-3.3-70B-Instruct m03 20000
  log "L queue done" ) &
( for m in Qwen3-4B Qwen3-8B Qwen3-14B gemma-3-4b-it gemma-3-12b-it; do probe 3 $MD/$m en_plain_xmodel_$m 1 2 3 4 5; done
  e2e 3 $M8 m03 1000000
  e2e 3 $M8 c03 1000000
  probe 3 $MD/gemma-3-27b-it en_plain_xmodel_gemma-3-27b-it 1 2 3 4 5
  e2e 3 $MD/Qwen3-8B m03 20000
  e2e 3 $MD/gemma-3-4b-it m03 20000
  log "S queue done" ) &
wait
log "EN PLAIN XMODEL DONE"

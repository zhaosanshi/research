#!/bin/bash
# job_en_xmodel.sh -- English-materials (PROBE_LANG=en) rerun of the cross-model block (RESULTS.md §8) on
# Qwen3 4B/8B/14B/32B, gemma-3 4b/12b/27b and Llama-3.3-70B, so the "family difference" can be checked in English.
# Mirrors the Chinese pilot cell by cell: X20k slot_start probe with forced-'user' continuation (Llama-generated
# English filler, 5 seeds), own-voice filler + meta-cell probe for the four small Qwen/Gemma models, and T=1 sampling
# (20k tokens, 5 orders x 20 000 first-token draws + 32 full answers) for the cells the pilot sampled (+ 4B/27B/32B m13).
# Stage 0:  L = Llama-3.3-70B probe on GPUs 0,1,2  |  S = GPU 3 queue: five small-model probes, then own-voice filler +
#           meta probe (Qwen3-8B/14B, gemma-3-4b/12b), then the small-model e2e cells.
# Stage 1:  (after the 70B probe) Qwen3-32B probe on GPUs 0,1  |  gemma-3-27b-it probe on GPU 2.
# Stage 2:  70B e2e m13 -> m10 -> c13 on GPUs 0,1,2 (S keeps running on GPU 3).
# Stage 3:  Qwen3-32B e2e m13 on GPUs 0,1  |  gemma-3-27b-it e2e m13 on GPU 2.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_en_x 'bash job_en_xmodel.sh 2>&1 | tee results/job_en_xmodel.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=en -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
X="X20k_c00 X20k_c10 X20k_c13 X20k_m00 X20k_m10 X20k_m13"
XM="X20k_m00 X20k_m10 X20k_m13"
E2E_FLAGS="--target-tokens 20000 --seeds 1 2 3 4 5 --n-first 20000 --n-full 32 --batch 8 --max-new 200 --max-hit-cont 3"

nt() { case ${1##*/} in Qwen3*) echo "--no-think";; esac; }   # Qwen3: enable_thinking=False everywhere

probe() {  # gpus model out-name conds(quoted) seeds... : slot_start probe + forced-'user' continuation, device_map auto
  local gpus=$1 m=$2 name=$3 conds=$4; shift 4
  local fill=${FILL_OVERRIDE:-$FILL}
  for s in "$@"; do
    $DOCKER -e CUDA_VISIBLE_DEVICES=$gpus $IMG python turn_boundary_probe.py --model $m $(nt $m) $fill --device auto \
        --force-first user --max-forced 48 --conditions $conds --filler-seed $s --out results/$name.jsonl \
        >> results/$name.log 2>&1 || log "FAILED $name seed $s -- see results/$name.log"
  done
  log "$name done ($(wc -l < results/$name.jsonl 2>/dev/null) records)"
}
filler() {  # gpu model : 400 English everyday pairs in the model's own voice
  local gpu=$1 m=$2
  local out=results/filler_en_general_${m##*/}.jsonl   # separate statement: a single `local a=$1 b=${a}` expands b with the OLD a
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpu $IMG python make_filler.py --model $m $(nt $m) --kind meta --n 400 --out $out \
      > ${out%.jsonl}.log 2>&1 && log "filler ${m##*/} done ($(wc -l < $out) pairs)" || log "FAILED filler ${m##*/} -- see ${out%.jsonl}.log"
}
e2e() {  # gpus model cell : T=1 sampling, Llama English filler; cell = m13 | m10 | c13
  local gpus=$1 m=$2 cell=$3 kind contam
  local out=results/en_e2e_x_${m##*/}_$cell.jsonl
  case $cell in m*) kind=meta;; c*) kind=coding;; esac
  contam=${cell:2:1}
  log "e2e ${m##*/} $cell start"
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpus $IMG python e2e_sample.py --model $m $(nt $m) --device auto $FILL $E2E_FLAGS \
      --kind $kind --wrapper 1 --contam-k $contam --tag en_x_${m##*/}_$cell --out $out > ${out%.jsonl}.log 2>&1 \
      && log "e2e ${m##*/} $cell done" || log "FAILED e2e ${m##*/} $cell -- see ${out%.jsonl}.log"
}

for f in results/filler_en_coding.jsonl results/filler_en_general.jsonl; do [ -s "$f" ] || { log "missing $f -- abort"; exit 1; }; done

log "stage 0: Llama-3.3-70B probe (GPUs 0,1,2) | small-model queue S (GPU 3)"
( probe 0,1,2 $MD/Llama-3.3-70B-Instruct en_xmodel_Llama-3.3-70B-Instruct "$X" 1 2 3 4 5 ) &
P70=$!
( for m in Qwen3-4B Qwen3-8B Qwen3-14B gemma-3-4b-it gemma-3-12b-it; do probe 3 $MD/$m en_xmodel_$m "$X" 1 2 3 4 5; done
  log "S: small-model probes done; own-voice filler + meta probe"
  for m in Qwen3-8B Qwen3-14B gemma-3-4b-it gemma-3-12b-it; do
    filler 3 $MD/$m
    FILL_OVERRIDE="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general_$m.jsonl" \
        probe 3 $MD/$m en_xmodel_ownvoice_$m "$XM" 1 2 3 4 5
  done
  log "S: own-voice done; e2e cells"
  e2e 3 $MD/Qwen3-8B m13;  e2e 3 $MD/Qwen3-8B m10
  e2e 3 $MD/gemma-3-4b-it m13;  e2e 3 $MD/gemma-3-4b-it m10
  e2e 3 $MD/Qwen3-14B m13;  e2e 3 $MD/gemma-3-12b-it m13
  e2e 3 $MD/Qwen3-4B m13
  log "S queue done" ) &
PS=$!
wait $P70
log "stage 0 done (70B probe); stage 1: Qwen3-32B probe (GPUs 0,1) | gemma-3-27b-it probe (GPU 2)"
( probe 0,1 $MD/Qwen3-32B en_xmodel_Qwen3-32B "$X" 1 2 3 4 5 ) &
P32=$!
( probe 2 $MD/gemma-3-27b-it en_xmodel_gemma-3-27b-it "$X" 1 2 3 4 5 ) &
P27=$!
wait $P32 $P27
log "stage 1 done; ALL PROBES DONE except own-voice; stage 2: 70B e2e on GPUs 0,1,2"
for cell in m13 m10 c13; do e2e 0,1,2 $MD/Llama-3.3-70B-Instruct $cell; done
log "stage 2 done (70B e2e)"
wait $PS
log "stage 3: Qwen3-32B e2e m13 (GPUs 0,1) | gemma-3-27b-it e2e m13 (GPU 2)"
( e2e 0,1 $MD/Qwen3-32B m13 ) &
( e2e 2 $MD/gemma-3-27b-it m13 ) &
wait
log "EN XMODEL DONE"

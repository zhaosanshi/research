#!/bin/bash
# job_en_xmodel_part2.sh -- remainder of job_en_xmodel.sh after the probe stages (relaunched 2026-10-24 09:10 because the
# first version expanded the e2e output name from an unset variable -> all cells would have shared one file).
# All eight X20k probes and the Qwen3-8B/14B own-voice runs are already in results/ (en_xmodel_*.jsonl).
#   L' (GPUs 0,1,2): Llama-3.3-70B e2e m13 -> m10 -> c13, then Qwen3-32B e2e m13 on GPUs 0,1.
#   S' (GPU 3):      own-voice filler + meta probe for gemma-3-4b-it and gemma-3-12b-it, then the small-model e2e cells
#                    (Qwen3-8B m13/m10, gemma-3-4b-it m13/m10, Qwen3-14B m13, gemma-3-12b-it m13, Qwen3-4B m13),
#                    then gemma-3-27b-it e2e m13.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_en_x2 'bash job_en_xmodel_part2.sh 2>&1 | tee results/job_en_xmodel_part2.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=en -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
XM="X20k_m00 X20k_m10 X20k_m13"
XF="X20k_c10 X20k_c13 X20k_m10 X20k_m13"   # cells for the forced-'User' (capitalised) check
E2E_FLAGS="--target-tokens 20000 --seeds 1 2 3 4 5 --n-first 20000 --n-full 32 --batch 8 --max-new 200 --max-hit-cont 3"

nt() { case ${1##*/} in Qwen3*) echo "--no-think";; esac; }   # Qwen3: enable_thinking=False everywhere

probe() {  # gpus model out-name conds(quoted) seeds... : slot_start probe + forced role-word continuation, device_map auto
  # FILL_OVERRIDE (filler flags) and FORCE_WORD (default 'user') are read from the environment of the caller.
  local gpus=$1 m=$2 name=$3 conds=$4; shift 4
  local fill=${FILL_OVERRIDE:-$FILL} word=${FORCE_WORD:-user}
  for s in "$@"; do
    $DOCKER -e CUDA_VISIBLE_DEVICES=$gpus $IMG python turn_boundary_probe.py --model $m $(nt $m) $fill --device auto \
        --force-first $word --max-forced 48 --conditions $conds --filler-seed $s --out results/$name.jsonl \
        >> results/$name.log 2>&1 || log "FAILED $name seed $s -- see results/$name.log"
  done
  log "$name done ($(wc -l < results/$name.jsonl 2>/dev/null) records)"
}
filler() {  # gpu model : 400 English everyday pairs in the model's own voice
  local gpu=$1 m=$2
  local out=results/filler_en_general_${m##*/}.jsonl
  if [ -s "$out" ] && [ "$(wc -l < $out)" -ge 400 ]; then log "filler ${m##*/} already present ($(wc -l < $out) pairs), skipping"; return 0; fi
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpu $IMG python make_filler.py --model $m $(nt $m) --kind meta --n 400 --out $out \
      > ${out%.jsonl}.log 2>&1 && log "filler ${m##*/} done ($(wc -l < $out) pairs)" || log "FAILED filler ${m##*/} -- see ${out%.jsonl}.log"
}
e2e() {  # gpus model cell : T=1 sampling, Llama English filler; cell = m13 | m10 | c13
  local gpus=$1 m=$2 cell=$3 kind contam
  local out=results/en_e2e_x_${m##*/}_$cell.jsonl
  case $cell in m*) kind=meta;; c*) kind=coding;; esac
  contam=${cell:2:1}
  log "e2e ${m##*/} $cell start -> $out"
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpus $IMG python e2e_sample.py --model $m $(nt $m) --device auto $FILL $E2E_FLAGS \
      --kind $kind --wrapper 1 --contam-k $contam --tag en_x_${m##*/}_$cell --out $out > ${out%.jsonl}.log 2>&1 \
      && log "e2e ${m##*/} $cell done" || log "FAILED e2e ${m##*/} $cell -- see ${out%.jsonl}.log"
}

for f in results/filler_en_coding.jsonl results/filler_en_general.jsonl; do [ -s "$f" ] || { log "missing $f -- abort"; exit 1; }; done

log "part 2 start: L' = forced-'User' 70B/32B, then 70B e2e on GPUs 0,1,2 ; S' = forced-'User' 8B, own-voice gemma, small-model e2e on GPU 3"
( FORCE_WORD=User probe 0,1,2 $MD/Llama-3.3-70B-Instruct en_xmodel_forceUser_Llama-3.3-70B-Instruct "$XF" 1 2 3 4 5
  FORCE_WORD=User probe 0,1 $MD/Qwen3-32B en_xmodel_forceUser_Qwen3-32B "$XF" 1 2 3 4 5
  for cell in m13 m10 c13; do e2e 0,1,2 $MD/Llama-3.3-70B-Instruct $cell; done
  log "L': 70B e2e done; Qwen3-32B e2e m13 on GPUs 0,1"
  e2e 0,1 $MD/Qwen3-32B m13
  log "L' queue done" ) &
( FORCE_WORD=User probe 3 $MD/Llama-3.1-8B-Instruct en_xmodel_forceUser_Llama-3.1-8B-Instruct "$XF" 1 2 3 4 5
  for m in gemma-3-4b-it gemma-3-12b-it; do
    filler 3 $MD/$m
    FILL_OVERRIDE="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general_$m.jsonl" \
        probe 3 $MD/$m en_xmodel_ownvoice_$m "$XM" 1 2 3 4 5
  done
  log "S': own-voice done; e2e cells"
  e2e 3 $MD/Qwen3-8B m13;  e2e 3 $MD/Qwen3-8B m10
  e2e 3 $MD/gemma-3-4b-it m13;  e2e 3 $MD/gemma-3-4b-it m10
  e2e 3 $MD/Qwen3-14B m13;  e2e 3 $MD/gemma-3-12b-it m13
  e2e 3 $MD/Qwen3-4B m13
  e2e 3 $MD/gemma-3-27b-it m13
  log "S' queue done" ) &
wait
log "EN XMODEL PART 2 DONE"

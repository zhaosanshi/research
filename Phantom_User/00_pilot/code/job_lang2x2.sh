#!/bin/bash
# job_lang2x2.sh -- diagnostic for the Chinese-pilot vs English gap: language x content 2x2 on Llama-3.1-8B-Instruct.
# The pilot (Chinese, AI-self talk) gave P(user)~5e-3 in the contaminated cell; the English everyday materials give ~2e-5.
# Missing corners: English AI-self talk, and Chinese everyday talk.  Each corner: own filler, X20k_m10 + X20k_m13, 5 seeds.
# Waits for job_en.sh to finish (log line), then uses GPU 0 and GPU 1.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_2x2 'bash job_lang2x2.sh 2>&1 | tee results/job_lang2x2.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}/Llama-3.1-8B-Instruct
COND="X20k_m10 X20k_m13"

log "waiting for job_en.sh (EN JOB DONE in results/job_en.log)"
until grep -q "EN JOB DONE" results/job_en.log 2>/dev/null; do sleep 60; done
log "job_en.sh finished; starting the 2x2 corners"

corner() {  # gpu lang topics filler-coding out-name
  local gpu=$1 lang=$2 topics=$3 fcod=$4 name=$5
  local fill=results/filler_${lang}_${topics}.jsonl
  log "GPU $gpu: filler $lang/$topics"
  $DOCKER -e CUDA_VISIBLE_DEVICES=$gpu -e PROBE_LANG=$lang $IMG python make_filler.py --model $M --kind meta --topics $topics --n 400 \
      --out $fill > results/filler_${lang}_${topics}.log 2>&1 || { log "FAILED filler $lang/$topics"; return 1; }
  log "GPU $gpu: filler done ($(wc -l < $fill) pairs); probing $COND"
  for s in 1 2 3 4 5; do
    $DOCKER -e CUDA_VISIBLE_DEVICES=$gpu -e PROBE_LANG=$lang $IMG python turn_boundary_probe.py --model $M \
        --filler-coding $fcod --filler-meta $fill --force-first user --max-forced 48 --conditions $COND \
        --filler-seed $s --out results/$name.jsonl >> results/$name.log 2>&1 || log "FAILED $name seed $s"
  done
  log "$name done ($(wc -l < results/$name.jsonl) records)"
}

( corner 0 en selftalk results/filler_en_coding.jsonl en_selftalk_cross_Llama-3.1-8B-Instruct ) &
( corner 1 zh everyday results/filler_coding.jsonl zh_everyday_cross_Llama-3.1-8B-Instruct ) &
wait
log "2x2 DONE"

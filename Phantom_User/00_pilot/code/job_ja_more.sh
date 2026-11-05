#!/bin/bash
# job_ja_more.sh -- whole-conversation Japanese (lang/ja.json, filler_ja_general.jsonl) on the remaining models, after
# Qwen3-8B and DeepSeek-V4-Flash both showed their plain-'user' gate opening in Japanese (RESULTS.md §15.13, §16.1).
# Contaminant = template word 'user' (plain), cells X20k_m00 + X20k_m03, 3 seeds, forced user / User / ユーザー.
# Launch:  tmux new-session -d -s opus_ja 'bash job_ja_more.sh 2>&1 | tee results/job_ja_more.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
D="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=ja -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_ja_general.jsonl"
probe() {  # gpus model nt
  local gpus=$1 m=$2 nt=$3
  for s in 1 2 3; do
    $D -e CUDA_VISIBLE_DEVICES=$gpus $IMG python turn_boundary_probe.py --model $MD/$m $nt $FILL --device auto \
        --force-first user User ユーザー --max-forced 48 --conditions X20k_m00 X20k_m03 --filler-seed $s \
        --out results/lang_ja_user_$m.jsonl >> results/lang_ja_user_$m.log 2>&1 || log "FAILED ja $m seed $s"
  done
  log "ja $m done ($(wc -l < results/lang_ja_user_$m.jsonl) records)"
}
log "ja extension start"
( probe 0,1,2 Llama-3.3-70B-Instruct ""; probe 0,1 Qwen3-32B --no-think; log "big queue done" ) &
( probe 3 Qwen3-14B --no-think; probe 3 gemma-3-12b-it ""; probe 3 gemma-3-27b-it ""; log "GPU3 queue done" ) &
wait
log "JA MORE DONE"

#!/bin/bash
# job_roleword.sh -- does the role word inside the contaminant matter? Chinese materials, plain contaminant, role word
# 用户 vs user; P(用户), P(user) and forced continuations for both words.  One model per GPU.
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
D="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=zh -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-coding results/filler_coding.jsonl --filler-meta results/filler_meta.jsonl"
run() {  # gpu model nt
  local gpu=$1 m=$2 nt=$3
  for w in 用户 user; do
    local tag=yonghu; [ "$w" = user ] && tag=user
    for s in 1 2 3; do
      $D -e CUDA_VISIBLE_DEVICES=$gpu -e CONTAM_ROLE_WORD=$w $IMG python turn_boundary_probe.py --model $MD/$m $nt $FILL --device auto \
          --force-first 用户 user --max-forced 48 --conditions X20k_m00 X20k_m03 X20k_c03 --filler-seed $s \
          --out results/zh_roleword_${tag}_$m.jsonl >> results/zh_roleword_${tag}_$m.log 2>&1 || log "FAILED $m $w seed $s"
    done
    log "$m role word $w done ($(wc -l < results/zh_roleword_${tag}_$m.jsonl) records)"
  done
}
( run 1 Llama-3.1-8B-Instruct "" ) &
( run 2 Qwen3-8B --no-think ) &
( run 3 gemma-3-4b-it "" ) &
wait
log "ROLEWORD DONE"

#!/bin/bash
# job_rolewords_multi.sh -- 2026-10-24: try the role word 'user' in other languages as the contaminant
# (English materials, plain).  For each word: clean + contaminated general cells, 3 seeds, P(first token of word),
# forced continuation with the word and with 'user'.  One model per GPU.
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T') ] $*"; }
D="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=en -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
WORDS=(ユーザー 사용자 Benutzer Utilisateur Usuario Пользователь 用户)
TAGS=(ja ko de fr es ru zh)
run() {  # gpu model nt
  local gpu=$1 m=$2 nt=$3
  for i in "${!WORDS[@]}"; do
    local w=${WORDS[$i]} tag=${TAGS[$i]}
    for s in 1 2 3; do
      $D -e CUDA_VISIBLE_DEVICES=$gpu -e CONTAM_ROLE_WORD=$w $IMG python turn_boundary_probe.py --model $MD/$m $nt $FILL --device auto \
          --force-first "$w" user --max-forced 48 --conditions X20k_m00 X20k_m03 --filler-seed $s \
          --out results/rolewords_${tag}_$m.jsonl >> results/rolewords_${tag}_$m.log 2>&1 || log "FAILED $m $w seed $s"
    done
    log "$m $tag ($w) done ($(wc -l < results/rolewords_${tag}_$m.jsonl) records)"
  done
}
( run 1 Qwen3-8B --no-think ) &
( run 2 gemma-3-4b-it "" ) &
( run 3 Llama-3.1-8B-Instruct "" ) &
wait
log "ROLEWORDS MULTI DONE"

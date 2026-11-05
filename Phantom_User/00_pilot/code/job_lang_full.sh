#!/bin/bash
# job_lang_full.sh -- language-matched test (2026-10-24): the WHOLE conversation in German / Japanese (lang/de.json,
# lang/ja.json), own filler per language, paper setting (plain contaminant, wrapper=0).  Per language and model:
#   (a) contaminant role word = the template token 'user'  : X20k_m00 + X20k_m03, 5 seeds, forced 'user' + native word
#   (b) contaminant role word = native word (Benutzer / ユーザー): X20k_m03, 5 seeds, forced native word + 'user'
# Stage 1: filler (de with Llama-3.1-8B which supports German; ja with Qwen3-8B).  Stage 2: de on GPU 0, ja on GPU 1.
# Launch on the GPU server inside tmux:  tmux new-session -d -s opus_lang 'bash job_lang_full.sh 2>&1 | tee results/job_lang_full.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
D="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
nt() { case ${1##*/} in Qwen3*) echo "--no-think";; esac; }
native() { case $1 in de) echo Benutzer;; ja) echo ユーザー;; esac; }

log "stage 1: filler de (GPU0, Llama-8B) | ja (GPU1, Qwen3-8B)"
( $D -e CUDA_VISIBLE_DEVICES=0 -e PROBE_LANG=de $IMG python make_filler.py --model $MD/Llama-3.1-8B-Instruct --kind meta --n 400 \
      --out results/filler_de_general.jsonl > results/filler_de_general.log 2>&1 && log "filler de done ($(wc -l < results/filler_de_general.jsonl))" || log "FAILED filler de" ) &
( $D -e CUDA_VISIBLE_DEVICES=1 -e PROBE_LANG=ja $IMG python make_filler.py --model $MD/Qwen3-8B --no-think --kind meta --n 400 \
      --out results/filler_ja_general.jsonl > results/filler_ja_general.log 2>&1 && log "filler ja done ($(wc -l < results/filler_ja_general.jsonl))" || log "FAILED filler ja" ) &
wait
for L in de ja; do [ -s results/filler_${L}_general.jsonl ] || { log "missing filler $L -- abort"; exit 1; }; done

run_lang() {  # gpu lang
  local gpu=$1 L=$2 w=$(native $2)
  local FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_${L}_general.jsonl"
  for m in Qwen3-8B gemma-3-4b-it Llama-3.1-8B-Instruct; do
    for s in 1 2 3 4 5; do
      $D -e CUDA_VISIBLE_DEVICES=$gpu -e PROBE_LANG=$L $IMG python turn_boundary_probe.py --model $MD/$m $(nt $m) $FILL --device auto \
          --force-first user "$w" --max-forced 48 --conditions X20k_m00 X20k_m03 --filler-seed $s \
          --out results/lang_${L}_user_$m.jsonl >> results/lang_${L}_user_$m.log 2>&1 || log "FAILED $L user $m seed $s"
      $D -e CUDA_VISIBLE_DEVICES=$gpu -e PROBE_LANG=$L -e CONTAM_ROLE_WORD=$w $IMG python turn_boundary_probe.py --model $MD/$m $(nt $m) $FILL --device auto \
          --force-first "$w" user --max-forced 48 --conditions X20k_m03 --filler-seed $s \
          --out results/lang_${L}_native_$m.jsonl >> results/lang_${L}_native_$m.log 2>&1 || log "FAILED $L native $m seed $s"
    done
    log "$L $m done ($(wc -l < results/lang_${L}_user_$m.jsonl) + $(wc -l < results/lang_${L}_native_$m.jsonl) records)"
  done
}
log "stage 2: probes de (GPU0) | ja (GPU1)"
( run_lang 0 de ) &
( run_lang 1 ja ) &
wait
log "LANG FULL DONE"

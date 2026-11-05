#!/bin/bash
# job_joint.sh -- exact whole-word probabilities for the role-word variants that showed first-token mass >= 1e-8 on Qwen3-8B
# (multi-token words inflate the first-token number), plus the native words of the whole-conversation language test.
# Waits for the DeepSeek probe (DSV4 DONE) to free the GPUs.  GPU 0: Qwen3-8B English variants; GPU 1: language runs.
# Launch:  tmux new-session -d -s opus_joint 'bash job_joint.sh 2>&1 | tee results/job_joint.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "DSV4 DONE" results/job_dsv4.log 2>/dev/null; do sleep 60; done
D="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
log "joint start"
( $D -e CUDA_VISIBLE_DEVICES=0 -e PROBE_LANG=en $IMG python joint_prob.py --model $MD/Qwen3-8B --no-think \
      --filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl --cells m00 m03 --seeds 1 2 3 \
      --words user "[User]" "<user>" YONGHU Yonghu yonghu "yong hu" Operator U Prompt Me HUMAN Gebruiker Użytkownik Kullanıcı Client Input Q "👤" "người dùng" You "**User**" "### User" ">" Benutzer ユーザー 用户 \
      --out results/joint_en_Qwen3-8B.jsonl > results/joint_en_Qwen3-8B.log 2>&1 && log "joint en Qwen done" || log "FAILED joint en Qwen" ) &
( for L in de ja; do
    case $L in de) w=Benutzer;; ja) w=ユーザー;; esac
    for m in Qwen3-8B gemma-3-4b-it Llama-3.1-8B-Instruct; do
      nt=""; case $m in Qwen3*) nt="--no-think";; esac
      $D -e CUDA_VISIBLE_DEVICES=1 -e PROBE_LANG=$L $IMG python joint_prob.py --model $MD/$m $nt \
          --filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_${L}_general.jsonl --cells m00 m03 --seeds 1 2 3 \
          --words user User "$w" --out results/joint_${L}_$m.jsonl > results/joint_${L}_$m.log 2>&1 && log "joint $L $m done" || log "FAILED joint $L $m"
    done
  done ) &
wait
log "JOINT DONE"

#!/bin/bash
# job_overnight.sh -- unattended queue for the GPU server (2026-10-22).  Runs after the per-GPU queues
# (job_xm_g*, job_ask_probe_g0, job_ask_e2e_g*, job_xe2e_g*) finish.  Launch inside tmux:
#   tmux new-session -d -s opus_bug 'bash results/job_overnight.sh 2>&1 | tee results/job_overnight.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
FILL="--filler-meta results/filler_meta.jsonl --filler-coding results/filler_coding.jsonl"
X="X20k_m00 X20k_m10 X20k_m13 X20k_c10 X20k_c13"
wait_for() { while pgrep -f "$1" >/dev/null; do sleep 30; done; }

log "stage 0: waiting for the per-GPU queues (job_xm / job_ask / job_xe2e)"
wait_for 'job_(xm|ask_probe|ask_e2e|xe2e)_g'
log "stage 0 done"

log "stage 1: Qwen3-32B on GPUs 0,1 and gemma-3-27b-it on GPUs 2,3 (X20k probe, 5 seeds each)"
( for s in 1 2 3 4 5; do $DOCKER -e CUDA_VISIBLE_DEVICES=0,1 $IMG python turn_boundary_probe.py --model $M/Qwen3-32B --device auto --no-think $FILL --force-first user --max-forced 48 --conditions $X --filler-seed $s --out results/xmodel_v1_Qwen3-32B.jsonl; done > results/xmodel_v1_Qwen3-32B.log 2>&1 ) &
( for s in 1 2 3 4 5; do $DOCKER -e CUDA_VISIBLE_DEVICES=2,3 $IMG python turn_boundary_probe.py --model $M/gemma-3-27b-it --device auto $FILL --force-first user --max-forced 48 --conditions $X --filler-seed $s --out results/xmodel_v1_gemma-3-27b-it.jsonl; done > results/xmodel_v1_gemma-3-27b-it.log 2>&1 ) &
wait
log "stage 1 done"

log "stage 2: own-voice meta filler for Qwen3-8B / gemma-3-4b-it / gemma-3-12b-it / Qwen3-14B"
( $DOCKER -e CUDA_VISIBLE_DEVICES=0 $IMG python make_filler.py --model $M/Qwen3-8B --no-think --kind meta --n 320 --out results/filler_meta_Qwen3-8B.jsonl > results/filler_meta_Qwen3-8B.log 2>&1 ) &
( $DOCKER -e CUDA_VISIBLE_DEVICES=1 $IMG python make_filler.py --model $M/gemma-3-4b-it --kind meta --n 320 --out results/filler_meta_gemma-3-4b-it.jsonl > results/filler_meta_gemma-3-4b-it.log 2>&1 ) &
( $DOCKER -e CUDA_VISIBLE_DEVICES=2 $IMG python make_filler.py --model $M/gemma-3-12b-it --kind meta --n 320 --out results/filler_meta_gemma-3-12b-it.jsonl > results/filler_meta_gemma-3-12b-it.log 2>&1 ) &
( $DOCKER -e CUDA_VISIBLE_DEVICES=3 $IMG python make_filler.py --model $M/Qwen3-14B --no-think --kind meta --n 320 --out results/filler_meta_Qwen3-14B.jsonl > results/filler_meta_Qwen3-14B.log 2>&1 ) &
wait
log "stage 2 done"

log "stage 3: self-approval at answer_end on Gemma 4B/12B and Qwen3-8B (their leak side), T=1 sampling"
E2E="python e2e_sample.py $FILL --measure answer_end --target-tokens 20000 --seeds 1 2 3 4 5 --n-first 20000 --n-full 64 --batch 8 --max-new 120 --max-hit-cont 3"
( $DOCKER -e CUDA_VISIBLE_DEVICES=0 $IMG $E2E --model $M/gemma-3-4b-it --kind coding --wrapper 0 --approval-k 4 --end-style ask --tag ask_c4 --out results/e2e_ask_c4_gemma-3-4b-it.jsonl > results/e2e_ask_c4_gemma-3-4b-it.log 2>&1 ) &
( $DOCKER -e CUDA_VISIBLE_DEVICES=1 $IMG $E2E --model $M/gemma-3-4b-it --kind coding --wrapper 0 --approval-k 4 --end-style neutral --tag ask_c4n --out results/e2e_ask_c4n_gemma-3-4b-it.jsonl > results/e2e_ask_c4n_gemma-3-4b-it.log 2>&1 ) &
( $DOCKER -e CUDA_VISIBLE_DEVICES=2 $IMG $E2E --model $M/gemma-3-12b-it --kind coding --wrapper 0 --approval-k 4 --end-style ask --tag ask_c4 --out results/e2e_ask_c4_gemma-3-12b-it.jsonl > results/e2e_ask_c4_gemma-3-12b-it.log 2>&1 ) &
( $DOCKER -e CUDA_VISIBLE_DEVICES=3 $IMG $E2E --model $M/Qwen3-8B --no-think --kind coding --wrapper 0 --approval-k 4 --end-style ask --tag ask_c4 --out results/e2e_ask_c4_Qwen3-8B.jsonl > results/e2e_ask_c4_Qwen3-8B.log 2>&1 ) &
wait
log "stage 3 done"
log "ALL OVERNIGHT STAGES DONE"

#!/bin/bash
# job_dsv4.sh -- DeepSeek-V4-Flash (284B MoE, 13B active, FP8, $MODEL_ROOT/deepseek-v4-flash) slot_start gate probe via vLLM
# (transformers 4.57 in the NGC image has no deepseek_v4).  Needs all 4 GPUs (TP=4), so it waits for the language and
# variant jobs to finish.  Paper setting (English, plain contaminant); then the native <｜User｜> contaminant form.
# Launch on the GPU server inside tmux:  tmux new-session -d -s opus_dsv4 'bash job_dsv4.sh 2>&1 | tee results/job_dsv4.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for LANG FULL DONE and VARIANTS DONE"
true
log "GPUs free; starting DeepSeek-V4-Flash probe (vLLM, TP=4)"
D="docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING -e VLLM_USE_DEEP_GEMM=0 -e PROBE_LANG=en -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} --entrypoint python3 vllm/vllm-openai:v0.27.1"
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
$D dsv4_probe.py --model $MODEL_ROOT/deepseek-v4-flash --tp 4 $FILL --cells m00 m03 c00 c03 --seeds 1 2 3 \
    --out results/dsv4_probe.jsonl > results/dsv4_probe.log 2>&1 && log "dsv4 plain done ($(wc -l < results/dsv4_probe.jsonl) records)" || log "FAILED dsv4 plain -- see results/dsv4_probe.log"
$D dsv4_probe.py --model $MODEL_ROOT/deepseek-v4-flash --tp 4 $FILL --cells m03 c03 --seeds 1 2 3 --native-contam \
    --out results/dsv4_probe_native.jsonl > results/dsv4_probe_native.log 2>&1 && log "dsv4 native done ($(wc -l < results/dsv4_probe_native.jsonl) records)" || log "FAILED dsv4 native -- see results/dsv4_probe_native.log"
log "DSV4 DONE"

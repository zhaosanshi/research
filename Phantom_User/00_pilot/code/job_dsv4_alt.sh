#!/bin/bash
# job_dsv4_alt.sh -- DeepSeek-V4-Flash probe using the purpose-built image $IMG_VLLM
# (vLLM dev 20260712 + transformers 5.14 + flashinfer 0.6.14), after vllm/vllm-openai:v0.27.1 failed on this GPU
# architecture (DeepGEMM hyperconnection kernel: "Unsupported architecture" on sm_120).
# Launch:  tmux new-session -d -s opus_dsv4 'bash job_dsv4_alt.sh 2>&1 | tee results/job_dsv4.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
IMG=${IMG_VLLM:?set IMG_VLLM to a vLLM image with DeepSeek-V4 support, see README}
D="docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING -e VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE=1 -e PROBE_LANG=en -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} --entrypoint python3 $IMG"
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
log "DeepSeek-V4-Flash probe with $IMG (TP=4)"
$D dsv4_probe.py --model $MODEL_ROOT/deepseek-v4-flash --tp 4 $FILL --cells m00 m03 c00 c03 --seeds 1 2 3 \
    --out results/dsv4_probe.jsonl > results/dsv4_probe.log 2>&1 && log "dsv4 plain done ($(wc -l < results/dsv4_probe.jsonl) records)" || log "FAILED dsv4 plain -- see results/dsv4_probe.log"
if [ -s results/dsv4_probe.jsonl ]; then
  $D dsv4_probe.py --model $MODEL_ROOT/deepseek-v4-flash --tp 4 $FILL --cells m03 c03 --seeds 1 2 3 --native-contam \
      --out results/dsv4_probe_native.jsonl > results/dsv4_probe_native.log 2>&1 && log "dsv4 native done ($(wc -l < results/dsv4_probe_native.jsonl) records)" || log "FAILED dsv4 native -- see results/dsv4_probe_native.log"
fi
log "DSV4 DONE"

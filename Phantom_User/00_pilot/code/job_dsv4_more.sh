#!/bin/bash
# job_dsv4_more.sh -- DeepSeek-V4-Flash follow-ups after the paper-setting probe: (a) unseen role labels as contaminant
# (pinyin YONGHU, Operator, [User], U -- the ones that leaked on Qwen), English materials, contaminated general cell;
# (b) whole-conversation Japanese (lang/ja.json + filler_ja_general) with the template word 'user', clean + contaminated.
# Launch:  tmux new-session -d -s opus_dsv4 'bash job_dsv4_more.sh 2>&1 | tee results/job_dsv4_more.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
IMG=${IMG_VLLM:?set IMG_VLLM to a vLLM image with DeepSeek-V4 support, see README}
D="docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING -e VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE=1 -e CONTAM_STYLE=plain -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} --entrypoint python3"
log "(a) unseen labels, English"
$D -e PROBE_LANG=en $IMG dsv4_probe.py --model $MODEL_ROOT/deepseek-v4-flash --tp 4 --filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl \
    --cells m03 --seeds 1 2 3 --contam-words YONGHU Operator "[User]" U --out results/dsv4_probe_words.jsonl > results/dsv4_probe_words.log 2>&1 \
    && log "words done ($(wc -l < results/dsv4_probe_words.jsonl))" || log "FAILED words"
log "(b) whole-conversation Japanese"
$D -e PROBE_LANG=ja $IMG dsv4_probe.py --model $MODEL_ROOT/deepseek-v4-flash --tp 4 --filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_ja_general.jsonl \
    --cells m00 m03 --seeds 1 2 3 --contam-words user ユーザー --out results/dsv4_probe_ja.jsonl > results/dsv4_probe_ja.log 2>&1 \
    && log "ja done ($(wc -l < results/dsv4_probe_ja.jsonl))" || log "FAILED ja"
log "DSV4 MORE DONE"

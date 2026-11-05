#!/bin/bash
# job_en_plain.sh -- paper setting on Llama-3.1-8B: English generic materials, NO <USER_REQUEST> wrapper anywhere
# (wrapper=0 cells) and a plain contaminant ("\n\nuser\n<complaint>", CONTAM_STYLE=plain).  2026-10-24, after we decided
# that the wrapper from that session is not a public format and must not be part of the paper's experiments.
# Single GPU queue (GPU 3): cross cells -> T=1 sampling (contaminated first) -> logit lens + heads (instruct, base)
# -> length series.  Everything else (controls, cross-model) follows once the 70B/32B sampling frees GPUs 0-2.
# Launch on the GPU server inside tmux:
#   tmux new-session -d -s opus_plain 'bash job_en_plain.sh 2>&1 | tee results/job_en_plain.log'
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
GPU=${GPU:-3}
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=en -e CONTAM_STYLE=plain -e CUDA_VISIBLE_DEVICES=$GPU -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
MD=${MODEL_ROOT:?set MODEL_ROOT to the model directory}
M=$MD/Llama-3.1-8B-Instruct
MB=$MD/Llama-3.1-8B
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
X="X20k_c00 X20k_c03 X20k_m00 X20k_m03"
LEN="X8k_m03 X8k_c03 X32k_m03 X32k_c03 X64k_m03 X64k_c03"
MECH="$FILL --conditions X20k_m00 X20k_m03 X20k_c00 X20k_c03 --seeds 1 2 3"
E2E="python e2e_sample.py --model $M $FILL --target-tokens 20000 --seeds 1 2 3 4 5 --n-first 20000 --n-full 32 --batch 8 --max-new 200 --max-hit-cont 3 --wrapper 0"

probe() {  # out-name conds(quoted) seeds...
  local name=$1 conds=$2; shift 2
  for s in "$@"; do
    $DOCKER $IMG python turn_boundary_probe.py --model $M $FILL --force-first user --max-forced 48 --conditions $conds \
        --filler-seed $s --out results/$name.jsonl >> results/$name.log 2>&1 || log "FAILED $name seed $s -- see results/$name.log"
  done
  log "$name done ($(wc -l < results/$name.jsonl 2>/dev/null) records)"
}
mech() {  # script tag model-dir
  local script=$1 tag=$2 mdir=$3 out=results/en_plain_${1%.py}_$2
  $DOCKER $IMG python $script --model $mdir --tokenizer $M $MECH --tag $tag --out $out.jsonl > $out.log 2>&1 \
      && log "$script $tag done ($(wc -l < $out.jsonl) records)" || log "FAILED $script $tag -- see $out.log"
}
e2e() {  # cell kind contam
  local out=results/en_plain_e2e_$1.jsonl
  $DOCKER $IMG $E2E --kind $2 --contam-k $3 --tag en_plain_$1 --out $out > ${out%.jsonl}.log 2>&1 \
      && log "e2e $1 done" || log "FAILED e2e $1 -- see ${out%.jsonl}.log"
}

for f in results/filler_en_coding.jsonl results/filler_en_general.jsonl; do [ -s "$f" ] || { log "missing $f -- abort"; exit 1; }; done
log "plain queue start on GPU $GPU"
probe en_plain_cross_Llama-3.1-8B-Instruct "$X" 1 2 3 4 5
e2e m03 meta 3;  e2e c03 coding 3;  e2e m00 meta 0;  e2e c00 coding 0
mech logit_lens.py Llama-3.1-8B-Instruct $M
mech head_attrib.py Llama-3.1-8B-Instruct $M
mech logit_lens.py Llama-3.1-8B-Base $MB
mech head_attrib.py Llama-3.1-8B-Base $MB
probe en_plain_length_Llama-3.1-8B-Instruct "$LEN" 1 2 3
log "EN PLAIN DONE"

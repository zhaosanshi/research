#!/bin/bash
# job_en_plain_extra.sh -- RECONSTRUCTED after the fact (2026-11-05).  The run that produced
# results/en_plain_extra_Llama-3.1-8B-Instruct.jsonl on 2026-10-24 was launched by hand; this script repeats it with
# the arguments recorded in that file (condition kwargs, 5 filler seeds, paper setting) and the probe() of job_en_plain.sh.
#   U20k_{m00,c00}_{ptr,pfk} : the REAL user pastes role-formatted text into the final request (transcript / fake line)
#   T20k_{m,c}0{0,3}_ipy     : tool results sent through Llama's own 'ipython' tool role instead of the user role
cd ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory} || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
GPU=${GPU:-3}
DOCKER="docker run --rm --gpus all --ipc=host -v $PHANTOM_ROOT:$PHANTOM_ROOT -v $MODEL_ROOT:$MODEL_ROOT:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROBE_LANG=en -e CONTAM_STYLE=plain -e CUDA_VISIBLE_DEVICES=$GPU -w ${PHANTOM_ROOT:?set PHANTOM_ROOT to this code directory}"
IMG=${IMG:?set IMG to a PyTorch container image, see README}
M=${MODEL_ROOT:?set MODEL_ROOT to the model directory}/Llama-3.1-8B-Instruct
FILL="--filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl"
CONDS="U20k_m00_ptr U20k_c00_ptr U20k_m00_pfk U20k_c00_pfk T20k_m00_ipy T20k_m03_ipy T20k_c00_ipy T20k_c03_ipy"
OUT=results/en_plain_extra_Llama-3.1-8B-Instruct
for s in 1 2 3 4 5; do
  $DOCKER $IMG python turn_boundary_probe.py --model $M $FILL --force-first user --max-forced 48 --conditions $CONDS \
      --filler-seed $s --out $OUT.jsonl >> $OUT.log 2>&1 || log "FAILED seed $s -- see $OUT.log"
done
log "extra done ($(wc -l < $OUT.jsonl) records)"

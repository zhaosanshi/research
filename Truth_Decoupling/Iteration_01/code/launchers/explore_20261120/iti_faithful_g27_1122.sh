#!/bin/bash
# 2026-11-22 15:2x (Fable 5.2), exploration: the faithful ITI baseline for the third model with standard attention,
# Gemma-3-27B, after Llama-3.1-8B and Gemma-3-12B (launchers/explore_20261120/iti_faithful_1122.sh). Same fit options
# (--faithful, the 96 train questions, seed 42), same grid (K 16 32 48 64, alpha 2.5 .. 30), same rules and readings
# (iti_grid.sh, iti_chain.sh, iti_free.sh; discussion/work_queue/2026-11-22_queue.md section 4). The comparison is the
# main-table setting of this model (layer 28, strength 1, write rule "prompt": LIKE name pospromptl28).
# Why: both models done so far gave single E and N1 above ITI; with this one the baseline covers every model of the
# project on which ITI can be run as published (the two Qwen models have linear-attention layers).
# One 27B lane per card. Card A: fit, acceptance check, then K 16. Card B: the K = 0 MMLU pass (the model itself),
# then K 32. Finish acceptance and the K = 0 pass before the four K lanes start. Card C:
# K 48, once the second reading of Gemma-3-12B (its last GSM8K pass) has ended. K 64 goes to the first card whose
# lane has ended. The main and the second reading wait for the four lanes by markers; a lane that failed on a point
# is repaired by the chain's own pass over the whole grid.
# The patched iti_baseline.py of the repository (export checks from the review of 2026-11-22) must be in place at
# both paths; this script refuses otherwise, so that every file of this run comes from one version.
# usage: iti_faithful_g27_1122.sh CARD_A CARD_B CARD_C        e.g. iti_faithful_g27_1122.sh 2 3 0
# Markers: those of iti_grid.sh / iti_chain.sh / iti_free.sh for gemma3-27b, fit name faithful_seed42.
cd /data/sanshizhao/Truth_Decoupling || exit 1
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
IT=$ROOT/Iteration_01/code/scripts/newsets/iti_baseline.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
A="2.5 5 7.5 10 15 20 25 30"
F=faithful_seed42
MODEL=gemma3-27b
LIKE=pospromptl28
[ "$#" -eq 3 ] || { echo "usage: iti_faithful_g27_1122.sh CARD_A CARD_B CARD_C" >&2; exit 2; }
CA=$1; CB=$2; CC=$3
[ "$CA" != "$CB" ] && [ "$CA" != "$CC" ] && [ "$CB" != "$CC" ] || { echo "three distinct GPU indices required" >&2; exit 2; }
cards=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits) || exit 1
for card in "$CA" "$CB" "$CC"; do
  [[ $card =~ ^(0|[1-9][0-9]*)$ ]] && grep -Fxq -- "$card" <<< "$cards" || { echo "unknown GPU index: $card" >&2; exit 2; }
done
say() { echo "$(date '+%F %T') iti faithful g27: $*" >> $L/iti_all.log; }
# The detached grid, chain and free workers inherit fd 9. The lock outlives this
# launcher and is released by the OS after its last worker exits.
exec 9>"$L/ITI_LAUNCH_${MODEL}_${F}_endq.lock" || exit 1
flock -n 9 || { say "not started: this model and fit already have active workers"; exit 1; }
WANT=1e6c53d6eff1e8e3
for p in Iteration_01/code/scripts/newsets/iti_baseline.py Iteration_01/code/scripts/newsets_stage/iti_baseline.py; do
  [ "$(sha256sum < $p | cut -c1-16)" = "$WANT" ] || { say "FAILED: $p is not the patched program"; exit 1; }
done
DX() { local g=$1; shift; docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$g -w $ROOT truth_dec "$@"; }
G=Iteration_01/baselines/iti/$MODEL
mkdir -p $G
say "start: fit on card $CA"
if [ ! -e $G/fit_$F.pt ]; then
  DX $CA python3 -I $IT fit --models $M --model $MODEL --faithful --out $ROOT/$G/fit_$F.pt > $L/iti_fit_${MODEL}_$F.log 2>&1 || { say "FAILED: fit"; exit 1; }
fi
lane() { setsid nohup bash $L/iti_grid.sh "$@" > /dev/null 2>&1 < /dev/null & }
check_log=$L/iti_check_${MODEL}_${F}_20261122.txt
if ! DX "$CA" python3 -I "$IT" check --models "$M" --model "$MODEL" --fit "$ROOT/$G/fit_$F.pt" --k 48 --alpha 5 > "$check_log" 2>&1 || ! grep -qx 'check PASS' "$check_log"; then
  say "FAILED: acceptance check"; exit 1
fi
# A chain or a free worker can start before its producer clears an old marker.
# Reset only this model and fit under the lock. Keep reusable data and choices.
for marker in "$L/ITI_GRID_${MODEL}_${F}_endq_"{f16,f32,f48,f64} "$L/ITI_CHAIN_${MODEL}_${F}_endq" "$L/ITI_FREE_${MODEL}_${F}_endq"; do
  rm -f "${marker}_DONE" "${marker}_FAILED" || exit 1
done
# The budget reader of every K uses K=0. Existing, completed K=0 data are reused.
if [ ! -s "$G/mmlu_k0.json" ]; then
  bash "$L/iti_grid.sh" "$MODEL" "$CB" "$F" endq "0" "$A" f0 && [ -s "$G/mmlu_k0.json" ] || { say "FAILED: K=0 baseline"; exit 1; }
fi
say "fit, acceptance and K 0 done; K 16/32 start on cards $CA/$CB"
lane $MODEL $CB $F endq "32" "$A" f32
lane $MODEL $CA $F endq "16" "$A" f16
setsid nohup bash $L/iti_chain.sh $MODEL $F endq $CA "16 32 48 64" "$A" $LIKE f16 f32 f48 f64 > /dev/null 2>&1 < /dev/null &
setsid nohup bash $L/iti_free.sh $MODEL $F endq $CA "16 32 48 64" "$A" $LIKE $CA $CB > /dev/null 2>&1 < /dev/null &
until [ -e $L/ITI_FREE_gemma3-12b_${F}_endq_DONE ] || [ -e $L/ITI_FREE_gemma3-12b_${F}_endq_FAILED ]; do sleep 60; done
lane $MODEL $CC $F endq "48" "$A" f48
say "K 48 started on card $CC"
while :; do
  for t in f16:$CA f32:$CB f48:$CC; do
    tag=${t%%:*}; card=${t##*:}
    if [ -e $L/ITI_GRID_${MODEL}_${F}_endq_${tag}_DONE ] || [ -e $L/ITI_GRID_${MODEL}_${F}_endq_${tag}_FAILED ]; then
      lane $MODEL $card $F endq "64" "$A" f64
      say "all lanes started (K 64 on card $card)"
      exit 0
    fi
  done
  sleep 60
done

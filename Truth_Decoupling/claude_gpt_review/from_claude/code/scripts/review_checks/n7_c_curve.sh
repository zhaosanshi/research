#!/usr/bin/env bash
# N7 step 1 driver: the C mid-strength curve for one model. n7_c_curve.sh MODEL GPU [ALPHAS...]
# Default strengths 0.25 0.5 0.75 1 1.5 2 (the 2x2 used 2, where C saturates: Llama abst leaves 100 / 1058 test
# statements answered, Qwen / Gemma abst leave 0). Each strength is one E41 run + its pre-declared analysis; the
# curve table is n7_curve_table.py. Pre-declared criterion (notepad N7): the Dr.'s sentence holds only if, at >= 1
# strength on >= 2 models, E+C's self-selected answers contain more correct ones than E thresholded on its own
# confidence to the SAME count (P2), Holm-corrected within model; P1 (gate threshold) is the secondary.
set -euo pipefail
MODEL=${1:?model}; GPU=${2:?gpu}; shift 2
ALPHAS=("$@"); if (( ${#ALPHAS[@]} == 0 )); then ALPHAS=(0.25 0.5 0.75 1 1.5 2); fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
for a in "${ALPHAS[@]}"; do
    echo "[$(date +%H:%M:%S)] == $MODEL alpha $a"
    bash "$HERE/n7_run.sh" "$MODEL" "$GPU" "$a"
done
echo "[$(date +%H:%M:%S)] N7_CURVE_DONE $MODEL"

#!/usr/bin/env bash
# Work queue 2026-11-22_queue_d (Claude Opus 5.5): the GPU step of diag/selfjudge_probe.py ("score") for one model, in
# the container, on GPU 1 only, with the code copy and interpreter E45 ran with (code_e45_20261112 / container python3
# for Llama-3.1-8B; code_newmodels_20261112 / the model config's interpreter for Qwen3.5-9B). Refuses to start when
# GPU 1 plus this model would exceed 65 GB. Never writes into E45's result dirs (output: Iteration_01/newsets/selfjudge/).
#   usage (host, from /data/sanshizhao/Truth_Decoupling): selfjudge_1122.sh MODEL [extra args, e.g. --regress-limit 64 --regress-only]
set -euo pipefail
MODEL=${1:?model}; shift
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
case "$MODEL" in
  llama3.1-8b) CODE=code_e45_20261112; NEED=17000 ;;
  qwen3.5-9b)  CODE=code_newmodels_20261112; NEED=20000 ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
# Queue D/E workers share physical card 1. Hold this descriptor until all attempts finish.
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu1.lock
mkdir -p "$(dirname "$LOCK")" || { echo "cannot prepare GPU 1 reservation" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock unavailable on GPU host" >&2; exit 2; }
exec 9>"$LOCK" || { echo "cannot open GPU 1 reservation" >&2; exit 2; }
flock -n 9 || { echo "another self-judge worker holds GPU 1" >&2; exit 3; }
USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 == 1 {print $2}') || { echo "cannot query GPU 1 memory" >&2; exit 2; }
[[ $USED =~ ^[0-9]+$ ]] || { echo "cannot read GPU 1 memory" >&2; exit 2; }
if [ $((USED + NEED)) -gt 66560 ]; then echo "GPU 1 holds $USED MiB; with $NEED MiB more it exceeds 65 GB, not starting" >&2; exit 3; fi
PY=$(docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$ROOT/$CODE/configs/models/$MODEL.json")
echo "selfjudge score: $MODEL, code $CODE, interpreter $PY, GPU 1 at $USED MiB before start, args: $*"
docker exec -u 1000:1001 -e "HOME=$ROOT/hf_cache" -e "HF_HOME=$ROOT/hf_cache" -e "TD_DATA_DIR=$ROOT/code/data/processed" \
    -e "TD_MODEL_ROOT=$ROOT/models" -e "TD_RESULTS_DIR=$ROOT/Iteration_01/newsets/selfjudge/_e45_unused" -e CUDA_VISIBLE_DEVICES=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=16 \
    -e "PYTHONPATH=$ROOT/$CODE" -w "$ROOT" truth_dec "$PY" -B Iteration_01/diag/selfjudge_probe.py score "$MODEL" "$@"
echo "selfjudge score: $MODEL done"

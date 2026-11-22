#!/usr/bin/env bash
# Work queues 2026-11-22_queue_e (1d) and 2026-11-22_queue_g (G4), Claude Opus 5.5: the GPU steps of
# diag/selfjudge_probe.py ("score", "cfgcheck") for the three models of the second and third steps, in the container,
# on GPU 0 only, one model at a time, with the code copy E45 ran with for the new models (code_newmodels_20261112) and
# the model config's interpreter. Qwen3.8-27B and Gemma-3-12B use the copy's own model config; Gemma-3-27B (no
# config in the copy) uses Iteration_01/newsets/selfjudge/_configs/gemma3-27b.json (repository:
# code/configs/selfjudge/). Holds the GPU-0 reservation (Iteration_01/newsets/selfjudge/.gpu0.lock, written as the
# GPU-1 one of selfjudge_1122.sh) and refuses to start when GPU 0 plus this model would exceed the card.
# Never writes into E45's result dirs (output: Iteration_01/newsets/selfjudge/).
#   usage (host, from /data/sanshizhao/Truth_Decoupling): selfjudge_gpu0_1122.sh score|cfgcheck MODEL [extra args]
set -euo pipefail
CMD=${1:?command}; shift
MODEL=${1:?model}; shift
case "$CMD" in score|cfgcheck) ;; *) echo "unknown command $CMD" >&2; exit 2 ;; esac
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=code_newmodels_20261112
case "$MODEL" in
  qwen3.8-27b) NEED=60000; CFG=$CODE/configs/models/$MODEL.json ;;
  gemma3-12b)  NEED=28000; CFG=$CODE/configs/models/$MODEL.json ;;
  gemma3-27b)  NEED=60000; CFG=Iteration_01/newsets/selfjudge/_configs/$MODEL.json ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu0.lock
mkdir -p "$(dirname "$LOCK")" || { echo "cannot prepare GPU 0 reservation" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock unavailable on GPU host" >&2; exit 2; }
exec 9>"$LOCK" || { echo "cannot open GPU 0 reservation" >&2; exit 2; }
flock -n 9 || { echo "another self-judge worker holds GPU 0" >&2; exit 3; }
read -r USED TOTAL < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '$1 == 0 {print $2, $3}') || { echo "cannot query GPU 0 memory" >&2; exit 2; }
[[ $USED =~ ^[0-9]+$ && $TOTAL =~ ^[0-9]+$ ]] || { echo "cannot read GPU 0 memory" >&2; exit 2; }
if [ $((USED + NEED)) -gt "$TOTAL" ]; then echo "GPU 0 holds $USED of $TOTAL MiB; with $NEED MiB more it exceeds the card, not starting" >&2; exit 3; fi
[ -f "$HOST/$CFG" ] || { echo "no model config $CFG" >&2; exit 2; }
PY=$(docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$ROOT/$CFG")
echo "selfjudge $CMD: $MODEL, code $CODE, config $CFG, interpreter $PY, GPU 0 at $USED of $TOTAL MiB before start, args: $*"
docker exec -u 1000:1001 -e "HOME=$ROOT/hf_cache" -e "HF_HOME=$ROOT/hf_cache" -e "TD_DATA_DIR=$ROOT/code/data/processed" \
    -e "TD_MODEL_ROOT=$ROOT/models" -e "TD_RESULTS_DIR=$ROOT/Iteration_01/newsets/selfjudge/_e45_unused" -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=16 \
    -e "PYTHONPATH=$ROOT/$CODE" -w "$ROOT" truth_dec "$PY" -B Iteration_01/diag/selfjudge_probe.py "$CMD" "$MODEL" "$@"
echo "selfjudge $CMD: $MODEL done"

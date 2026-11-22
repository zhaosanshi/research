#!/usr/bin/env bash
# Work queue 2026-11-22_queue_e, item "1g supplement: K3" (Claude Opus 5.5): for one model, in the container, on GPU 1
# only, with the code copy and interpreter E45 ran with (as selfjudge_gate_1122.sh):
#   1. diag/selfjudge_gate_k3.py capture MODEL   (plain-prompt residual stream, every layer, last prompt token; GPU)
#   2. diag/selfjudge_gate_k3.py fit MODEL       (logistic probe on the 100 (QA)^2 tune questions; CPU, sklearn)
# Refuses to start when GPU 1 plus this model would exceed 65 GB. Hard-fails on any error.
# Output only under Iteration_01/newsets/selfjudge/<MODEL>/k3_*.
#   usage (host, from /data/sanshizhao/Truth_Decoupling): selfjudge_gate_k3_1122.sh MODEL
set -euo pipefail
MODEL=${1:?model}
ROOT=/workspace/Truth_Decoupling
case "$MODEL" in
  llama3.1-8b) CODE=code_e45_20261112; NEED=17000 ;;
  qwen3.5-9b)  CODE=code_newmodels_20261112; NEED=20000 ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
check() {
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 == 1 {print $2}')
  [ -n "$USED" ] || { echo "cannot read GPU 1 memory" >&2; exit 2; }
  if [ $((USED + NEED)) -gt 66560 ]; then echo "GPU 1 holds $USED MiB; with $NEED MiB more it exceeds 65 GB, not starting" >&2; exit 3; fi
}
PY=$(docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$ROOT/$CODE/configs/models/$MODEL.json")
DX=(docker exec -u 1000:1001 -e "HOME=$ROOT/hf_cache" -e "HF_HOME=$ROOT/hf_cache" -e "TD_DATA_DIR=$ROOT/code/data/processed"
    -e "TD_MODEL_ROOT=$ROOT/models" -e "TD_RESULTS_DIR=$ROOT/Iteration_01/newsets/selfjudge/_e45_unused" -e CUDA_VISIBLE_DEVICES=1
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=16
    -e "PYTHONPATH=$ROOT/$CODE" -w "$ROOT" truth_dec "$PY" -B)
check
echo "selfjudge k3: $MODEL, code $CODE, interpreter $PY, GPU 1 at $USED MiB before capture"
"${DX[@]}" Iteration_01/diag/selfjudge_gate_k3.py capture "$MODEL"
echo "selfjudge k3: $MODEL fit"
"${DX[@]}" Iteration_01/diag/selfjudge_gate_k3.py fit "$MODEL"
echo "selfjudge k3: $MODEL done"

BASE=/data/sanshizhao/Truth_Decoupling
cd "$BASE"; mkdir -p logs
dex() { # dex <gpu> <root> <args...>
  local gpu="$1" root="$2"; shift 2
  docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e CUDA_VISIBLE_DEVICES=$gpu -e PYTHONUNBUFFERED=1 \
    -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/$root -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed \
    -w /workspace/Truth_Decoupling/code/experiments truth_dec "$@"
}
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

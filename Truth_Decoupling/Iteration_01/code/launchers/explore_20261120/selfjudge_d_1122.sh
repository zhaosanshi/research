#!/bin/bash
# Work queue 2026-11-22_queue_e, "1 supplement", comparator D on (QA)^2 only (Claude Opus 5.5): the mean answer
# log-probability of single E's pass (plain prompt, x:E<L>.w.p<S> at "prompt") with the existing scorer
# scripts/newsets/ns_conf_pos.py (newsets_gen.py conf, writes at "prompt" as in the generation), exactly as
# launchers/explore_20261120/qaqa_strong.sh scores the strong arm (same environment, scoring batch 8, batch 1 only when
# the 98% greedy-reproduction check is missed), first-difference check against the no-write plain pass x.json.
# Output in our own dir (never into newsets/conf/): Iteration_01/newsets/selfjudge/<MODEL>/x--E<L>-w-p<S>-at-prompt.json.
# GPU 1 only; refuses to start when GPU 1 plus this model would exceed 65 GB.
#   usage (host): selfjudge_d_1122.sh MODEL E_LAYER E_STRENGTH      e.g. selfjudge_d_1122.sh llama3.1-8b 16 3.5
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
MODEL=${1:?model}; EL=${2:?layer}; S=${3:?strength}
case "$MODEL" in llama3.1-8b) NEED=17000 ;; qwen3.5-9b) NEED=20000 ;; *) echo "unknown model" >&2; exit 2 ;; esac
# Queue D/E workers share physical card 1. Hold this descriptor until all attempts finish.
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu1.lock
mkdir -p "$(dirname "$LOCK")" || { echo "cannot prepare GPU 1 reservation" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock unavailable on GPU host" >&2; exit 2; }
exec 9>"$LOCK" || { echo "cannot open GPU 1 reservation" >&2; exit 2; }
flock -n 9 || { echo "another self-judge worker holds GPU 1" >&2; exit 3; }
USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 == 1 {print $2}') || { echo "cannot query GPU 1 memory" >&2; exit 2; }
[[ $USED =~ ^[0-9]+$ ]] || { echo "cannot read GPU 1 memory" >&2; exit 2; }
if [ $((USED + NEED)) -gt 66560 ]; then echo "GPU 1 holds $USED MiB; with $NEED MiB more it exceeds 65 GB, not starting" >&2; exit 3; fi
CODE=$ROOT/Iteration_01/code
NCP=$CODE/scripts/newsets/ns_conf_pos.py
M=$CODE/configs/newsets/frozen_models_20261119.json
BANK=$ROOT/Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl
GEN=$ROOT/Iteration_01/newsets/gen_prompt/qaqafull/$MODEL/test
TOK=x--E$EL-w-p$(echo "$S" | tr . -)
OUT=$ROOT/Iteration_01/newsets/selfjudge/$MODEL/$TOK-at-prompt.json
HOUT=$HOST/Iteration_01/newsets/selfjudge/$MODEL/$TOK-at-prompt.json
[ -e "$HOUT" ] && { echo "$HOUT exists" >&2; exit 2; }
PY=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' \
     $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || exit 2
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=1 -w $ROOT truth_dec)
echo "selfjudge D: $MODEL $TOK, GPU 1 at $USED MiB before start, interpreter $PY"
CMD=("${DX[@]}" $PY -I $NCP qaqafull prompt conf --task qaqafull --models $M --model $MODEL --items $BANK
     --gen-file $GEN/$TOK.json --ref-gen-file $GEN/x.json --out $OUT)
TMP=$(mktemp)
"${CMD[@]}" > $TMP 2>&1; rc=$?
cat $TMP
if [ $rc -ne 0 ]; then
  if grep -q "teacher-forced arg-max reproduces only" $TMP; then
    echo "selfjudge D: the 98% check was missed at scoring batch 8, once more at scoring batch 1"
    "${CMD[@]}" --score-batch 1 || { rm -f $TMP; echo "selfjudge D: FAILED" >&2; exit 1; }
  else
    rm -f $TMP; echo "selfjudge D: FAILED" >&2; exit 1
  fi
fi
rm -f $TMP
[ -s "$HOUT" ] || { echo "selfjudge D: no output" >&2; exit 1; }
echo "selfjudge D: $MODEL done -> $HOUT"

#!/bin/bash
# 2026-11-22 10:1x (Opus 5.5), exploration: three checks asked for by the read-only review of the ITI baseline.
#   lane A (GPU_A): the full-stop refit (diag/iti_period_check.py) for Llama-3.1-8B and Gemma-3-12B, each compared with
#                   the existing fit; and the acceptance check of iti_baseline.py, this time with its output kept
#                   (logs/newsets_20261119/iti_check_<MODEL>_20261122.txt);
#   lane B (GPU_B): Llama-3.1-8B K 48 alpha 5 again with the current iti_baseline.py (its tune answers and its MMLU pass
#                   were written by the first version), into baselines/iti/llama3.1-8b/recheck/, compared token for
#                   token and pick for pick with the files the reading used.
# Nothing here is judged and nothing replaces a file the readings use. Output: logs/newsets_20261119/iti_recheck_1122.txt
# usage: iti_recheck_1122.sh GPU_A GPU_B      Marker: ITI_RECHECK_1122_DONE / _FAILED
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
GA=$1; GB=$2
[ -n "$GB" ] || { echo "usage: iti_recheck_1122.sh GPU_A GPU_B"; exit 2; }
OUT=$L/iti_recheck_1122.txt
rm -f $L/ITI_RECHECK_1122_DONE $L/ITI_RECHECK_1122_FAILED
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
IT=$ROOT/Iteration_01/code/scripts/newsets/iti_baseline.py
PC=$ROOT/Iteration_01/diag/iti_period_check.py
DX() { local g=$1; shift; docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$g -w $ROOT truth_dec "$@"; }
noise() { grep -v "KernelPreference\|ScaleCalculationMode\|Loading checkpoint\|^layer\|generation flags\|sklearn\|scikit\|n_iter_i\|STOP: TOTAL\|Increase the number\|https://\|^ *$\|You might also\|Please also refer"; }
echo "# ITI re-checks, $(date '+%F %T')" > $OUT
(
  set -o pipefail
  rc=0
  for model in llama3.1-8b gemma3-12b; do
    B=Iteration_01/baselines/iti/$model
    mkdir -p $B/diag
    if [ ! -e $B/diag/fit_train_seed42_fullstop.pt ]; then
      DX $GA python3 -I $PC fit --models $M --model $model --out $ROOT/$B/diag/fit_train_seed42_fullstop.pt 2>&1 | noise > $L/.iti_recheck_fit_$model.txt || rc=1
    fi
    { echo "## $model: refit with a full stop at the end of every answer"; cat $L/.iti_recheck_fit_$model.txt | tail -2
      DX $GA python3 -I $PC compare --base $ROOT/$B/fit_train_seed42.pt --other $ROOT/$B/diag/fit_train_seed42_fullstop.pt 2>&1 | noise; } > $L/.iti_recheck_cmp_$model.txt || rc=1
    DX $GA python3 -I $IT check --models $M --model $model --fit $ROOT/$B/fit_train_seed42.pt --k 48 --alpha 5 2>&1 | noise > $L/iti_check_${model}_20261122.txt || rc=1
  done
  exit $rc
) &
pa=$!
(
  B=Iteration_01/baselines/iti/llama3.1-8b
  mkdir -p $B/recheck
  [ -e $B/recheck/tune_k48_a5.json ] || DX $GB python3 -I $IT gen --models $M --model llama3.1-8b --fit $ROOT/$B/fit_train_seed42.pt --k 48 --alpha 5 \
    --split tune --out $ROOT/$B/recheck/tune_k48_a5.json > $L/.iti_recheck_gen.txt 2>&1 || exit 1
  [ -e $B/recheck/mmlu_k48_a5.json ] || DX $GB python3 -I $IT mmlu --models $M --model llama3.1-8b --fit $ROOT/$B/fit_train_seed42.pt --k 48 --alpha 5 \
    --items $ROOT/data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids $ROOT/Iteration_01/capability/sets/mmlu_14042_20261120.json \
    --out $ROOT/$B/recheck/mmlu_k48_a5.json > $L/.iti_recheck_mmlu.txt 2>&1 || exit 1
  python3 - $B <<'EOF' > $L/.iti_recheck_cmp_version.txt
import json, sys
b = sys.argv[1]
old = json.load(open(b + "/train_seed42/gen/tune_k48_a5.json")); new = json.load(open(b + "/recheck/tune_k48_a5.json"))
same = sum(x["token_ids"] == y["token_ids"] and x["finished"] == y["finished"] for x, y in zip(old["rows"], new["rows"]))
print("## Llama K 48 alpha 5 with the current iti_baseline.py (script %s) against the file the reading used (script %s)" % (new["script_sha256"][:16], old["script_sha256"][:16]))
print("tune answers: %d of %d equal token for token; same question order %s; same heads %s" % (
    same, len(old["rows"]), [r["pair_id"] for r in old["rows"]] == [r["pair_id"] for r in new["rows"]], old["setting"]["heads"] == new["setting"]["heads"]))
mo = json.load(open(b + "/train_seed42/mmlu/k48_a5.json")); mn = json.load(open(b + "/recheck/mmlu_k48_a5.json"))
po = {r["id"]: r["pick"] for r in mo["rows"]}; pn = {r["id"]: r["pick"] for r in mn["rows"]}
print("MMLU: %d of %d picks equal; accuracy %.4f then, %.4f now" % (sum(po[i] == pn[i] for i in po), len(po), mo["accuracy"], mn["accuracy"]))
EOF
) &
pb=$!
bad=0
wait $pa || bad=1
wait $pb || bad=1
cat $L/.iti_recheck_cmp_llama3.1-8b.txt $L/.iti_recheck_cmp_gemma3-12b.txt $L/.iti_recheck_cmp_version.txt >> $OUT 2>/dev/null
for model in llama3.1-8b gemma3-12b; do echo "## acceptance check, $model"; grep "^check" $L/iti_check_${model}_20261122.txt; done >> $OUT
if [ "$bad" -ne 0 ]; then touch $L/ITI_RECHECK_1122_FAILED; exit 1; fi
touch $L/ITI_RECHECK_1122_DONE

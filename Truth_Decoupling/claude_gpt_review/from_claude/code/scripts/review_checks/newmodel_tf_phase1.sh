#!/usr/bin/env bash
# from_claude/41, phase 1 (Claude, 2026-11-12). Host side; every step runs in the truth_dec container.
#   lane gpu0 : qwen3.5-9b  E1 DAS all layers -> E23 DAS-E all layers -> E23 mean-diff E all layers -> E23 C all layers -> reports
#   gpu1      : qwen3.8-27b E1 DAS layers 0-24   (chunk root chunks/e1_a; merged in phase 2)
#   gpu2      : qwen3.8-27b E1 DAS layers 25-63  (chunk root chunks/e1_b)
#   gpu3      : qwen3.8-27b E23 mean-diff E all layers
# Flags are those of the old models' chains, only the model changes: E1 = tfce_chain.sh step 1, DAS-E sweep = tfce_chain.sh
# step 2, mean-diff sweep = das_tf_chain.sh step 3b, C sweep = tfce_C.sh. The interpreter comes from the model config's
# "python" field (default python3), so no model is named in the run function.
#   usage: newmodel_tf_phase1.sh            (logs, .DONE / .FAIL flags in results_newmodels_20261112/logs)
set -u
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/code_newmodels_20261112
RES=$ROOT/results_newmodels_20261112
LOG=$HOST/results_newmodels_20261112/logs
mkdir -p "$LOG"
[ -d "$HOST/code_newmodels_20261112" ] || { echo "missing fixed code copy $HOST/code_newmodels_20261112" >&2; exit 2; }
mkdir "$LOG/.lock_phase1" 2>/dev/null || { echo "phase 1 already started (lock $LOG/.lock_phase1)" >&2; exit 3; }

py_of() {  # the interpreter named in the model config (periphery); python3 when the config names none
  docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$CODE/configs/models/$1.json"
}

run() {  # run NAME GPU MODEL RESULTS_ROOT script args...   -> LOG/NAME.log, LOG/NAME.DONE or LOG/NAME.FAIL
  local name=$1 gpu=$2 model=$3 res=$4; shift 4
  local py
  py=$(py_of "$model") || { echo "== $(date '+%F %T') FAILED $name: no model config" >> "$LOG/chain.log"; touch "$LOG/$name.FAIL"; return 1; }
  echo "== $(date '+%F %T') start $name (gpu $gpu, $py)" >> "$LOG/chain.log"
  if docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_RESULTS_DIR="$res" \
      -e TD_DATA_DIR=$ROOT/code/data/processed -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES="$gpu" \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE \
      -w $CODE truth_dec "$py" "$@" > "$LOG/$name.log" 2>&1; then
    echo "== $(date '+%F %T') done $name" >> "$LOG/chain.log"; touch "$LOG/$name.DONE"
  else
    echo "== $(date '+%F %T') FAILED $name" >> "$LOG/chain.log"; touch "$LOG/$name.FAIL"; return 1
  fi
}

E1_FLAGS=(scripts/e1_directions.py --exp E1_directions_tqa --direction-prompt tf --das-anchor prompt --das-tags E --das-pairs 96
          --das-loss ce --das-bidir)
E23=(scripts/e23_knob_sweep.py --exp E2b_tqa --tune-pairs 192)

lane9b() {
  local M=qwen3.5-9b
  run e1_$M 0 $M $RES "${E1_FLAGS[@]}" --model $M --das-all-layers || return 1
  run e23_das_$M 0 $M $RES "${E23[@]}" --model $M --all-layers --skip-cp --no-cs \
      --e-estimator das --das-dir $RES/E1/$M --das-variant tf-ce-bidir --tag das_tfce_full || return 1
  run e23_md_$M 0 $M $RES "${E23[@]}" --model $M --all-layers --skip-cp --no-cs --e-estimator meandiff --tag meandiff_full || return 1
  run cmp_$M 0 $M $RES scripts/e23_compare_estimators.py --root $RES --model $M --tag-a das_tfce_full --label-a DAS-CE \
      --tag-b meandiff_full --label-b meandiff --out $RES/E23/$M/compare_das_tfce_vs_meandiff_full.md || return 1
  run drtable_$M 0 $M $RES scripts/dr_tf_table.py --root $RES --models $M --pairs-file configs/splits/dr_100pairs_20261110.json \
      --out $RES/dr_tf_table_$M || return 1       # the Dr's table does not need C: produced before the 3-hour C sweep
  run e23_C_$M 0 $M $RES "${E23[@]}" --model $M --all-layers --skip-e \
      --e-estimator das --das-dir $RES/E1/$M --das-variant tf-ce-bidir --tag das_tfce_C || return 1
  run r2x2_$M 0 $M $RES scripts/report_2x2.py --e-root $RES --e-tag das_tfce_full --c-root $RES --c-tag das_tfce_C --model $M \
      --out $RES/E23/$M/report_2x2_das_tfce.md || return 1
  touch "$LOG/lane9b.DONE"
}

M27=qwen3.8-27b
{ lane9b || touch "$LOG/lane9b.FAIL"; } &     # phase 2's gpu0 lane waits on lane9b.DONE / .FAIL
run e1a_$M27 1 $M27 $RES/chunks/e1_a "${E1_FLAGS[@]}" --model $M27 --das-layers $(seq 0 24) &
run e1b_$M27 2 $M27 $RES/chunks/e1_b "${E1_FLAGS[@]}" --model $M27 --das-layers $(seq 25 63) &
run e23_md_$M27 3 $M27 $RES "${E23[@]}" --model $M27 --all-layers --skip-cp --no-cs --e-estimator meandiff --tag meandiff_full &
wait
echo "== $(date '+%F %T') phase 1 finished" >> "$LOG/chain.log"

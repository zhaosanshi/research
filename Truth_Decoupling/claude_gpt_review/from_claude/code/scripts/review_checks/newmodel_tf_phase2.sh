#!/usr/bin/env bash
# from_claude/41, phases 2-4 for qwen3.8-27b (Claude, 2026-11-12). Host side; started together with newmodel_tf_phase1.sh and
# waits on its .DONE / .FAIL flags. Same run function, interpreter from the model config.
#   phase 2: merge the two E1 chunks (e1_merge_chunks.py), then the DAS-E sweep in two layer chunks (gpu1: 0-31, gpu2: 32-63)
#   phase 3: the C sweep in eight chunks of eight layers, taken from a queue by whichever GPU is free
#            (gpu3 after its mean-difference sweep, gpu0 after the 9B lane, gpu1 / gpu2 after their DAS-E chunk)
#   phase 4: concatenate the chunks (e23_concat_chunks.py), reports (compare, 2x2, the Dr's table for 27B and for both models)
# Any failed prerequisite stops the lanes that depend on it (loud: .FAIL flag and a line in chain.log). A lane that gives up
# writes the .FAIL flag of the first step it would have run, so no other lane waits for it forever (red team 2, 07:28).
# Restart after a failure: kill the host script AND its lanes / docker exec children, then remove the lock, claim_C_c* and
# stale .DONE / .FAIL flags and outputs by hand before launching again.
set -u
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/code_newmodels_20261112
RES=$ROOT/results_newmodels_20261112
LOG=$HOST/results_newmodels_20261112/logs
M=qwen3.8-27b
M9=qwen3.5-9b
NC=8                                   # C chunks of 8 layers
mkdir -p "$LOG"
mkdir "$LOG/.lock_phase2" 2>/dev/null || { echo "phase 2 already started (lock $LOG/.lock_phase2)" >&2; exit 3; }

py_of() {
  docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$CODE/configs/models/$1.json"
}
run() {  # run NAME GPU MODEL RESULTS_ROOT script args...
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
wait_for() {  # wait_for NAME... : 0 when every NAME.DONE exists, 1 as soon as one NAME.FAIL exists
  local n
  while true; do
    for n in "$@"; do [ -e "$LOG/$n.FAIL" ] && { echo "== $(date '+%F %T') stop: prerequisite $n failed" >> "$LOG/chain.log"; return 1; }; done
    local all=1
    for n in "$@"; do [ -e "$LOG/$n.DONE" ] || all=0; done
    [ $all = 1 ] && return 0
    sleep 60
  done
}
E23=(scripts/e23_knob_sweep.py --exp E2b_tqa --tune-pairs 192 --model $M)
DASARGS=(--e-estimator das --das-dir $RES/E1/$M --das-variant tf-ce-bidir)

c_queue() {  # c_queue GPU : take C chunks until none is left; a chunk is claimed atomically by mkdir
  local gpu=$1 k
  for k in $(seq 0 $((NC - 1))); do
    mkdir "$LOG/claim_C_c$k" 2>/dev/null || continue
    run e23_C_${M}_c$k "$gpu" $M $RES "${E23[@]}" --layers $(seq $((8 * k)) $((8 * k + 7))) --skip-e "${DASARGS[@]}" \
        --tag das_tfce_C_c$k || return 1
  done
}

lane1() {
  wait_for e1a_$M e1b_$M || { touch "$LOG/e1merge_$M.FAIL"; return 1; }
  run e1merge_$M 1 $M $RES scripts/review_checks/e1_merge_chunks.py --chunks $RES/chunks/e1_a/E1/$M $RES/chunks/e1_b/E1/$M \
      --out $RES/E1/$M --expect-layers 64 || return 1
  run e23_das_${M}_c0 1 $M $RES "${E23[@]}" --layers $(seq 0 31) --skip-cp --no-cs "${DASARGS[@]}" --tag das_tfce_full_c0 || return 1
  c_queue 1
}
lane2() {
  wait_for e1merge_$M || { touch "$LOG/e23_das_${M}_c1.FAIL"; return 1; }
  run e23_das_${M}_c1 2 $M $RES "${E23[@]}" --layers $(seq 32 63) --skip-cp --no-cs "${DASARGS[@]}" --tag das_tfce_full_c1 || return 1
  c_queue 2
}
lane3() { wait_for e23_md_$M e1merge_$M || return 1; c_queue 3; }
lane0() { wait_for lane9b e1merge_$M || return 1; c_queue 0; }
lane_e() {  # the Dr's table needs only the two E sweeps: do not wait for the eight C chunks (red team 2)
  wait_for e23_das_${M}_c0 e23_das_${M}_c1 e23_md_$M || { touch "$LOG/cat_das_$M.FAIL"; return 1; }
  run cat_das_$M 0 $M $RES scripts/review_checks/e23_concat_chunks.py --root $RES --model $M --tags das_tfce_full_c0 das_tfce_full_c1 \
      --out-tag das_tfce_full --expect-layers 64 || return 1
  run cmp_$M 0 $M $RES scripts/e23_compare_estimators.py --root $RES --model $M --tag-a das_tfce_full --label-a DAS-CE \
      --tag-b meandiff_full --label-b meandiff --out $RES/E23/$M/compare_das_tfce_vs_meandiff_full.md || return 1
  run drtable_$M 0 $M $RES scripts/dr_tf_table.py --root $RES --models $M --pairs-file configs/splits/dr_100pairs_20261110.json \
      --out $RES/dr_tf_table_$M || return 1
  until [ -e "$LOG/drtable_$M9.DONE" ]; do       # the 9B lane writes drtable_<9b>.DONE, or lane9b.FAIL if it stopped earlier
    [ -e "$LOG/lane9b.FAIL" ] || [ -e "$LOG/drtable_$M9.FAIL" ] && { echo "== $(date '+%F %T') no 9B table: both-model table skipped" >> "$LOG/chain.log"; return 1; }
    sleep 60
  done
  run drtable_both 0 $M $RES scripts/dr_tf_table.py --root $RES --models $M9 $M --pairs-file configs/splits/dr_100pairs_20261110.json \
      --out $RES/dr_tf_table_both || return 1
}

lane1 & lane2 & lane3 & lane0 & lane_e &
wait
# every lane has ended, so no new flag can appear: check once instead of waiting (red team 2)
CS=(); for k in $(seq 0 $((NC - 1))); do CS+=(e23_C_${M}_c$k); done
for n in "${CS[@]}" e23_das_${M}_c0; do
  [ -e "$LOG/$n.DONE" ] || { echo "== $(date '+%F %T') stop: $n did not finish; C concat and 2x2 skipped" >> "$LOG/chain.log"; exit 1; }
done
TAGS_C=(); for k in $(seq 0 $((NC - 1))); do TAGS_C+=(das_tfce_C_c$k); done
run cat_C_$M 0 $M $RES scripts/review_checks/e23_concat_chunks.py --root $RES --model $M --tags "${TAGS_C[@]}" \
    --out-tag das_tfce_C --expect-layers 64 || exit 1
[ -e "$LOG/cat_das_$M.DONE" ] || { echo "== $(date '+%F %T') stop: no concatenated DAS-E sweep; 2x2 skipped" >> "$LOG/chain.log"; exit 1; }
run r2x2_$M 0 $M $RES scripts/report_2x2.py --e-root $RES --e-tag das_tfce_full --c-root $RES --c-tag das_tfce_C --model $M \
    --out $RES/E23/$M/report_2x2_das_tfce.md || exit 1
echo "== $(date '+%F %T') phases 2-4 finished" >> "$LOG/chain.log"

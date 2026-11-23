#!/bin/bash
# 2026-11-24 01 h, exploration, queue M section 2 (discussion/work_queue/2026-11-24_queue_m.md; executor G, Claude Opus 5.5):
# true/false judging of Gemma-3-12B and Gemma-3-27B with the current DAS method, the steps of the 11-12 round for
# Qwen3.5-9B / Qwen3.8-27B (claude_gpt_review/from_claude/41; results_newmodels_20261112/newmodel_tf_phase1.sh lane9b and
# code_newmodels_20261112/scripts/review_checks/newmodel_tf_phase2.sh) = Iteration_01/launchers/tf_lane.sh, unchanged
# scripts and flags, only the model is a parameter. Difference to tf_lane.sh: every sweep runs in chunks of 8 layers (as
# the 11-12 27B run: e1 --das-layers, e23 --layers, merged by scripts/review_checks/e1_merge_chunks.py and
# e23_concat_chunks.py), and the card lock is taken per chunk and released after it, so other jobs on the card get turns.
#   E1 DAS chunks -> merge -> DAS-E sweep chunks -> mean-diff sweep chunks -> concat + finite checks -> compare ->
#   the Dr's table (100 pairs and 529 pairs) -> C sweep chunks -> concat + finite -> 2x2 report
# Code base Iteration_01/code (configs/models/gemma3-12b.json, gemma3-27b.json as the free-form runs); container python3.
# Cards 1 or 2 only, through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, retried every minute) and ours plus
# others at most 66560 MiB on the card (checked after the lock is taken, as falseqa_model.sh). One card per model run.
# A step whose .DONE flag exists is skipped (restart after an interruption resumes); any failure stops the run loudly.
# Results: Iteration_01/results/tf/gemma3/<12b|27b> (MODE full) or .../<12b|27b>_smoke_20261124 (MODE smoke); step logs,
# .DONE / .FAIL flags and chain.log in <results>/logs. Status lines "<date> tfg3 <model> <mode>: ..." in
# Iteration_01/logs/newsets_20261119/tfg3_all.log; markers TFG3_<MODEL>_DONE / TFG3_<MODEL>_FAILED / TFG3_<MODEL>_SMOKE_DONE.
# MODE smoke: one layer (12B: 24, 27B: 31): E1 DAS at that layer (96 pairs), then the DAS sweep (E, Cp, Cs) and the
# mean-diff sweep on the first 64 test statements (tune 8 pairs) with multiplier 0 added (E 0 2, C 0 1), and
# diag/tfg3_tables.py smoke (multiplier 0 = original model bit for bit; baseline numbers); peak card memory sampled.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/tfg3_model.sh KEY GPU smoke|full \
#                 > Iteration_01/logs/newsets_20261119/tfg3_<KEY>_<MODE>.out 2>&1 < /dev/null &      KEY = g12 | g27
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=${1:-}; GPU=${2:-}; MODE=${3:-}
[ -n "$MODE" ] || { echo "usage: tfg3_model.sh KEY GPU smoke|full"; exit 2; }
case $MODE in smoke|full) ;; *) echo "MODE must be smoke or full"; exit 2 ;; esac
case $GPU in 1|2) LIM=66560 ;; *) echo "cards 1 or 2 only"; exit 2 ;; esac
case $KEY in
  g12) MODEL=gemma3-12b; SUB=12b; SL=24; NEED=${TFG3_NEED:-30000} ;;
  g27) MODEL=gemma3-27b; SUB=27b; SL=31; NEED=${TFG3_NEED:-62000} ;;
  *) echo "unknown key $KEY"; exit 2 ;;
esac
CODE=$ROOT/Iteration_01/code; HC=$HOST/Iteration_01/code
if [ $MODE = full ]; then RSUB=Iteration_01/results/tf/gemma3/$SUB; else RSUB=Iteration_01/results/tf/gemma3/${SUB}_smoke_20261124; fi
RES=$ROOT/$RSUB; RH=$HOST/$RSUB; LOG=$RH/logs
MK=$L/TFG3_$MODEL
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') tfg3 $MODEL $MODE: $*" >> $L/tfg3_all.log; }
fail() { say "FAILED: $*"; echo "== $(date '+%F %T') FAILED: $*" >> "$LOG/chain.log"; touch ${MK}_FAILED; exit 1; }
mkdir -p "$LOG" || { echo "cannot create $LOG"; exit 2; }
mkdir "$LOG/.lock_tfg3" 2>/dev/null || { echo "run already started (lock $LOG/.lock_tfg3)"; exit 3; }
rm -f ${MK}_FAILED
[ $MODE = full ] && rm -f ${MK}_DONE
[ $MODE = smoke ] && rm -f ${MK}_SMOKE_DONE
[ -f $HC/configs/models/$MODEL.json ] || fail "no model config $MODEL"
NL=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["num_layers"]))' $HC/configs/models/$MODEL.json) || fail "num_layers"
PY=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("python", "python3"))' $HC/configs/models/$MODEL.json) || fail "python"
for f in scripts/review_checks/e1_merge_chunks.py scripts/review_checks/e23_concat_chunks.py; do [ -s $HC/$f ] || fail "missing $f"; done
[ -s $HOST/Iteration_01/diag/tfg3_tables.py ] || fail "missing diag/tfg3_tables.py"

code_sum() {  # the files the five true/false scripts import (listed from sys.modules on 2026-11-24), configs, data, merge tools
  ( cd $HC && sha256sum core/__init__.py core/config.py core/directions.py core/intervene.py core/io_utils.py core/layer_score.py \
      core/metrics.py core/model_io.py core/readout.py scripts/common.py scripts/dr_tf_table.py scripts/e1_directions.py \
      scripts/e22_two_stage.py scripts/e23_compare_estimators.py scripts/e23_knob_sweep.py scripts/report_2x2.py \
      scripts/review_checks/e1_merge_chunks.py scripts/review_checks/e23_concat_chunks.py \
      configs/models/$MODEL.json configs/experiments/E1_directions_tqa.json configs/experiments/E2b_tqa.json \
      configs/splits/dr_100pairs_20261110.json configs/prompts/default_v1.json \
      $HOST/code/data/processed/truthfulqa_tf/*.jsonl $HOST/code/data/processed/truthfulqa_tf/manifest.json ) \
    | sha256sum | cut -c1-16
}
SUM0=$(code_sum) && [ -n "$SUM0" ] || fail "cannot hash code, configs and data"
say "start (pid $$), card $GPU, $NL layers, results $RSUB, code+config+data hash $SUM0, need $NEED MiB"
echo "== $(date '+%F %T') start: model $MODEL, mode $MODE, gpu $GPU, results $RES, hash $SUM0" >> "$LOG/chain.log"

LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$GPU.lock
take() {  # take the card's lock (retry every minute), then wait for room under the memory rule
  exec 9>"$LOCK" || fail "cannot open $LOCK"
  until flock -n 9; do sleep 60; done
  while true; do
    USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1 == g {print $2}')
    [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $GPU"
    [ $((USED + NEED)) -le $LIM ] && break
    say "GPU $GPU holds $USED MiB, $NEED more would exceed $LIM; waiting"; sleep 60
  done
}
give() { flock -u 9; exec 9>&-; }

run() {  # run NAME gpu|cpu RESULTS_ROOT script args...  -> LOG/NAME.log, NAME.DONE / NAME.FAIL; skipped when NAME.DONE exists
  local name=$1 where=$2 res=$3 cv="" rc; shift 3
  [ -e "$LOG/$name.DONE" ] && { echo "== $(date '+%F %T') skip $name (done before)" >> "$LOG/chain.log"; return 0; }
  rm -f "$LOG/$name.FAIL"
  [ "$(code_sum)" = "$SUM0" ] || fail "$name: code, configs or data changed since the start"
  if [ $where = gpu ]; then take; cv=$GPU; fi
  echo "== $(date '+%F %T') start $name ($where${cv:+ $cv}, $PY${USED:+, card $USED MiB before})" >> "$LOG/chain.log"
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_RESULTS_DIR="$res" \
      -e TD_DATA_DIR=$ROOT/code/data/processed -e TD_MODEL_ROOT=$ROOT/models -e CUDA_VISIBLE_DEVICES="$cv" \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE \
      -w $CODE truth_dec "$PY" "$@" > "$LOG/$name.log" 2>&1
  rc=$?
  # after a card step: release the lock and pause 75 s, longer than the other launchers' 60 s retry, so a waiting job
  # gets its turn between our chunks (otherwise our next chunk retakes the lock at once)
  if [ $where = gpu ]; then give; [ $rc = 0 ] && sleep 75; fi
  USED=
  if [ $rc = 0 ]; then
    echo "== $(date '+%F %T') done $name" >> "$LOG/chain.log"; touch "$LOG/$name.DONE"
  else
    touch "$LOG/$name.FAIL"; fail "$name (exit $rc, log $RSUB/logs/$name.log)"
  fi
}

finite() {  # finite NAME NPZ: every float array of a sweep file is finite (as tf_lane.sh)
  local name=$1 npz=$2
  [ -e "$LOG/$name.DONE" ] && return 0
  if docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $CODE truth_dec python3 -B -c "import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True)
bad = [k for k in z.files if z[k].dtype.kind == 'f' and not np.isfinite(z[k]).all()]
print('arrays:', len(z.files), 'non-finite:', bad)
sys.exit(1 if bad else 0)" "$npz" > "$LOG/$name.log" 2>&1; then
    echo "== $(date '+%F %T') done $name" >> "$LOG/chain.log"; touch "$LOG/$name.DONE"
  else
    touch "$LOG/$name.FAIL"; fail "$name: non-finite values in $npz"
  fi
}

E1_FLAGS=(scripts/e1_directions.py --exp E1_directions_tqa --direction-prompt tf --das-anchor prompt --das-tags E --das-pairs 96
          --das-loss ce --das-bidir --model $MODEL)
E23=(scripts/e23_knob_sweep.py --exp E2b_tqa --model $MODEL)
DASARGS=(--e-estimator das --das-dir $RES/E1/$MODEL --das-variant tf-ce-bidir)

if [ $MODE = smoke ]; then
  MEM=$LOG/smoke_memory.txt
  ( while true; do echo "$(date '+%T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1 == g {print $2}')"; sleep 5; done ) > $MEM 2>&1 &
  SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null' EXIT
  SM=(--limit 64 --tune-pairs 8 --layers $SL)
  run e1_smoke gpu $RES "${E1_FLAGS[@]}" --das-layers $SL
  run e23_das_smoke gpu $RES "${E23[@]}" "${SM[@]}" --e-alphas 0 2 --c-alphas 0 1 "${DASARGS[@]}" --tag smoke_das
  run e23_md_smoke gpu $RES "${E23[@]}" "${SM[@]}" --e-alphas 0 2 --skip-cp --no-cs --e-estimator meandiff --tag smoke_md
  kill $SAMPLER 2>/dev/null
  run smoke_check cpu $RES $ROOT/Iteration_01/diag/tfg3_tables.py smoke --res $RES --model $MODEL --layer $SL
  PEAK=$(awk '{print $2}' $MEM | sort -n | tail -1); LOW=$(awk '{print $2}' $MEM | sort -n | head -1)
  say "smoke done: card memory during the smoke $LOW .. $PEAK MiB (ours plus others); check in $RSUB/logs/smoke_check.log"
  touch ${MK}_SMOKE_DONE
  exit 0
fi

# ---- full run: chunks of 8 layers
NC=$(( (NL + 7) / 8 ))
lay() { local k=$1 a b; a=$((8 * k)); b=$((8 * k + 7)); [ $b -ge $NL ] && b=$((NL - 1)); seq $a $b; }
E1C=(); TD=(); TM=(); TC=()
for k in $(seq 0 $((NC - 1))); do
  run e1_c$k gpu $RES/chunks/e1_c$k "${E1_FLAGS[@]}" --das-layers $(lay $k)
  E1C+=($RES/chunks/e1_c$k/E1/$MODEL)
  say "E1 DAS chunk $k of $NC (layers $(lay $k | head -1)-$(lay $k | tail -1)) done"
done
run e1merge cpu $RES scripts/review_checks/e1_merge_chunks.py --chunks "${E1C[@]}" --out $RES/E1/$MODEL --expect-layers $NL
for k in $(seq 0 $((NC - 1))); do
  run e23_das_c$k gpu $RES "${E23[@]}" --tune-pairs 192 --layers $(lay $k) --skip-cp --no-cs "${DASARGS[@]}" --tag das_tfce_full_c$k
  TD+=(das_tfce_full_c$k)
  say "DAS-E sweep chunk $k of $NC done"
done
for k in $(seq 0 $((NC - 1))); do
  run e23_md_c$k gpu $RES "${E23[@]}" --tune-pairs 192 --layers $(lay $k) --skip-cp --no-cs --e-estimator meandiff --tag meandiff_full_c$k
  TM+=(meandiff_full_c$k)
  say "mean-diff sweep chunk $k of $NC done"
done
run cat_das cpu $RES scripts/review_checks/e23_concat_chunks.py --root $RES --model $MODEL --tags "${TD[@]}" --out-tag das_tfce_full --expect-layers $NL
finite finite_das $RES/E23/$MODEL/knob_sweep_das_tfce_full.npz
run cat_md cpu $RES scripts/review_checks/e23_concat_chunks.py --root $RES --model $MODEL --tags "${TM[@]}" --out-tag meandiff_full --expect-layers $NL
finite finite_md $RES/E23/$MODEL/knob_sweep_meandiff_full.npz
run cmp cpu $RES scripts/e23_compare_estimators.py --root $RES --model $MODEL --tag-a das_tfce_full --label-a DAS-CE \
    --tag-b meandiff_full --label-b meandiff --out $RES/E23/$MODEL/compare_das_tfce_vs_meandiff_full.md
run drtable cpu $RES scripts/dr_tf_table.py --root $RES --models $MODEL --pairs-file configs/splits/dr_100pairs_20261110.json \
    --out $RES/dr_tf_table_$MODEL
say "three-method table written (dr_tf_table_$MODEL.md); C sweep next"
for k in $(seq 0 $((NC - 1))); do
  run e23_C_c$k gpu $RES "${E23[@]}" --tune-pairs 192 --layers $(lay $k) --skip-e "${DASARGS[@]}" --tag das_tfce_C_c$k
  TC+=(das_tfce_C_c$k)
  say "C sweep chunk $k of $NC done"
done
run cat_C cpu $RES scripts/review_checks/e23_concat_chunks.py --root $RES --model $MODEL --tags "${TC[@]}" --out-tag das_tfce_C --expect-layers $NL
finite finite_C $RES/E23/$MODEL/knob_sweep_das_tfce_C.npz
run r2x2 cpu $RES scripts/report_2x2.py --e-root $RES --e-tag das_tfce_full --c-root $RES --c-tag das_tfce_C --model $MODEL \
    --out $RES/E23/$MODEL/report_2x2_das_tfce.md
echo "== $(date '+%F %T') finished" >> "$LOG/chain.log"
say "all steps done"
touch ${MK}_DONE

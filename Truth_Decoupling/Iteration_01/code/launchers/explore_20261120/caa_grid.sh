#!/bin/bash
# 2026-11-22 19 h, exploration (Claude Opus 5.5): the formal CAA grid for one model (discussion/work_queue/
# 2026-11-22_queue_f.md, sections 1 and 7). WRITTEN, NOT RUN: it is started only after the acceptance
# (caa_accept.sh, ACCEPT_DONE) and GPT's review. Nothing here calls a judge (the judge cells are only derived).
# Needs baselines/caa/<MODEL>/vectors.pt, run_caa/ and models_run_caa.json from caa_accept.sh (scripts/newsets/
# caa_vector.py fit / mirror). The CAA vector is written through the engine unchanged: kind md (multiplier x the
# vector), position rule gen (last prompt token and every generated token) via scripts/review_checks/e44_positions.py
# gen (answers) and scripts/capability/cap_mcq_sweep.py arms ending in @gen (MMLU).
# MODE=grid (default): for every layer of LAYERS:
#   1. MMLU by option probability on the 14,042 items, one pass with the arms x and x:E<l>.md.p<m>@gen for every
#      multiplier (baselines/caa/<MODEL>/mmlu/L<l>.json); the selection-half change of each multiplier against the
#      file's own x (diag/per_model_table.halves, as the ITI budget) is written to mmlu/L<l>.budget;
#   2. early stop as in iti_grid.sh (multipliers in increasing order): the first multiplier whose selection-half change
#      is below -STOP (default 3) and every larger one get no tune answers (their MMLU is measured anyway: one pass per
#      layer is cheaper than one pass per point with its own no-write reference). The budget itself (default 1 point)
#      is applied when the table is read, not here;
#   3. TruthfulQA tune answers (192 questions) for the kept multipliers, one judge input per layer
#      (run_caa/eval_extra/judge_tune/caa_L<l>_<TAG>.json) and its cells in judge_tune_caa_L<l>_<TAG>_tune/.
# MODE=neg LAYER: the direction check of section 7, multiplier -1 at the chosen LAYER (tune answers, own judge input).
# MODE=zero LAYER: multiplier 0 on the whole tune split (caa_vector.py check with 24 batches at LAYER).
# Holds the lock of every card it uses (Iteration_01/newsets/selfjudge/.gpu<card>.lock, as discussion/validation_dataset/
# 2026-11-22d_probe.patch); a held lock is waited for (retried every minute, never bypassed); exits 3 when a card would
# exceed 65 GB (the caller retries).
# 2026-11-22 20 h (Claude Opus 5.5, GPT review f): GPT's guard (scripts/newsets/caa_run_guard.py grid / budget,
# caa_vector.py verify) applied; config key qwen3.8-27b; per-card locks; status lines in
# Iteration_01/logs/newsets_20261119/caa_all.log.
# usage (host): caa_grid.sh MODEL GPU "LAYER ..." "MULT ..." TAG       e.g. caa_grid.sh llama3.1-8b 1 "8 10 12 14 16 18 20 22 24" "0.5 1 2 4" g1
#               MODE=neg caa_grid.sh MODEL GPU LAYER - TAG ;  MODE=zero caa_grid.sh MODEL GPU LAYER - TAG
# Markers: Iteration_01/baselines/caa/<MODEL>/CAA_<MODE>_<TAG>_DONE / _FAILED; log caa_<MODE>_<TAG>.log there; one-line
# status in Iteration_01/logs/newsets_20261119/caa_all.log.
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
MODEL=${1:?model}; GPU=${2:?gpu}; LAYERS=${3:?layers}; MULTS=${4:?multipliers}; TAG=${5:?tag}
MODE=${MODE:-grid}; STOP=${STOP:-3}
case "$MODEL" in
  llama3.1-8b) NEED=17000 ;; qwen3.5-9b) NEED=20000 ;; gemma3-12b) NEED=26000 ;;
  gemma3-27b|qwen3.8-27b) NEED=58000 ;; *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
case "$MODE" in grid|neg|zero) ;; *) echo "MODE is grid, neg or zero" >&2; exit 2 ;; esac
B=Iteration_01/baselines/caa/$MODEL
[ -f $HOST/$B/acceptance/ACCEPT_DONE ] || { echo "no passed acceptance for $MODEL (caa_accept.sh)" >&2; exit 2; }
for f in vectors.pt models_run_caa.json run_caa/MIRROR.json; do [ -s $HOST/$B/$f ] || { echo "missing $B/$f" >&2; exit 2; }; done
GUARD=$HOST/Iteration_01/code/scripts/newsets/caa_run_guard.py
python3 -B "$GUARD" grid "$MODEL" "$LAYERS" "$MULTS" || exit 2
[[ $GPU =~ ^[0-9](,[0-9])*$ ]] || { echo "GPU is a comma-separated list of card numbers" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock unavailable on GPU host" >&2; exit 2; }
mkdir -p $HOST/Iteration_01/newsets/selfjudge || { echo "cannot prepare the card reservations" >&2; exit 2; }
for g in ${GPU//,/ }; do
  LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$g.lock
  exec {fd}>"$LOCK" || { echo "cannot open GPU $g reservation" >&2; exit 2; }
  until flock -n $fd; do echo "$(date '+%F %T') another worker holds GPU $g ($LOCK); retry in 60 s" >&2; sleep 60; done
done
NC=$(echo "$GPU" | tr ',' '\n' | grep -c .)
for g in ${GPU//,/ }; do
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$g" '$1 == g {print $2}') || { echo "cannot query GPU memory" >&2; exit 2; }
  [[ $USED =~ ^[0-9]+$ ]] || { echo "cannot read GPU $g memory" >&2; exit 2; }
  if [ $((USED + NEED / NC)) -gt 66560 ]; then echo "GPU $g holds $USED MiB; with $((NEED / NC)) MiB more it exceeds 65 GB, not starting" >&2; exit 3; fi
done
read -r CODE PY SRC E_ON < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["code"], m["python"], m["source_run"], m["e_on"])' $HOST/$B/models_run_caa.json "$MODEL") || { echo "model not in the mirror config" >&2; exit 2; }
[ -n "$E_ON" ] || { echo "model not in the mirror config" >&2; exit 2; }
CFG=$ROOT/$B/models_run_caa.json
RUN=$B/run_caa
MK=$HOST/$B/CAA_${MODE}_$TAG
LOG=$HOST/$B/caa_${MODE}_$TAG.log
rm -f ${MK}_DONE ${MK}_FAILED
say() { echo "$(date '+%F %T') caa $MODE $MODEL $TAG: $*" | tee -a $LOG >> $HOST/Iteration_01/logs/newsets_20261119/caa_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
ENV=(-u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models -e TD_DATA_DIR=$ROOT/code/data/processed
     -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPU)
DXR=(docker exec "${ENV[@]}" -w $ROOT truth_dec)
DXC=(docker exec "${ENV[@]}" -e PYTHONPATH=$ROOT/$CODE -w $ROOT/$CODE truth_dec)
COMMON=(--source-run $ROOT/$SRC --run-dir $ROOT/$RUN --e-on $E_ON --split tune --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
ITEMS=$ROOT/data_external/mmlu_20261119/all/test-00000-of-00001.parquet
IDS=$ROOT/Iteration_01/capability/sets/mmlu_14042_20261120.json
tok() { local a=${1//:/--}; a=${a//./-}; echo "${a}-at-gen_tune.json"; }   # e44_eval_extra.arm_token + e44_positions suffix
tune_arms() {  # tune_arms LAYER NAME ARMS... : answers, judge input, cells
  local l=$1 name=$2; shift 2
  local gen="" a W=$RUN/eval_extra/judge_tune_${name}_tune
  for a in "$@"; do gen="$gen $(tok $a)"; done
  "${DXC[@]}" $PY scripts/review_checks/e44_positions.py gen generate "${COMMON[@]}" --arms "$@" >> $LOG 2>&1 || return 1
  "${DXC[@]}" $PY scripts/review_checks/e44_positions.py gen export "${COMMON[@]}" --gen $gen --name $name >> $LOG 2>&1 || return 1
  mkdir -p $HOST/$W/judge2/cache && cp $HOST/$RUN/eval_extra/judge_tune/$name.json $HOST/$W/judge_input.json || return 1
  "${DXC[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $LOG 2>&1
}
"${DXR[@]}" $PY -I $ROOT/Iteration_01/code/scripts/newsets/caa_vector.py verify --models "$CFG" --model "$MODEL" --mirror "$RUN" --vectors "$ROOT/$B/vectors.pt" >> "$LOG" 2>&1 || fail "CAA fit/mirror verification"
say "start on GPU $GPU: layers $LAYERS, multipliers $MULTS, STOP $STOP"
if [ "$MODE" = zero ]; then
  "${DXR[@]}" $PY -I $ROOT/Iteration_01/code/scripts/newsets/caa_vector.py check --models $CFG --model $MODEL --mirror $RUN \
      --vectors $ROOT/$B/vectors.pt --out $ROOT/$B/check_zero_full_L${LAYERS}_$TAG.json --first-batches 24 --layers $LAYERS >> $LOG 2>&1 || fail "zero"
  say "done"; touch ${MK}_DONE; exit 0
fi
if [ "$MODE" = neg ]; then
  tune_arms $LAYERS caa_neg_L${LAYERS}_$TAG x:E$LAYERS.md.m1 || fail "neg"
  say "done"; touch ${MK}_DONE; exit 0
fi
mkdir -p $HOST/$B/mmlu
FAILED=0
for l in $LAYERS; do
  arms="x"; for s in $MULTS; do arms="$arms x:E$l.md.p$s@gen"; done
  cache_mode=reuse
  if [ ! -s $HOST/$B/mmlu/L$l.json ]; then
    cache_mode=fresh
    "${DXR[@]}" $PY -I $ROOT/Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models $CFG --model $MODEL \
        --items $ITEMS --ids $IDS --out $ROOT/$B/mmlu/L$l.json --arms $arms >> $LOG 2>&1 || { say "mmlu L$l failed"; FAILED=1; continue; }
  fi
  python3 -B "$GUARD" budget "$MODEL" "$l" "$MULTS" "$cache_mode" > "$HOST/$B/mmlu/L$l.budget" 2>> "$LOG" || { say "no valid MMLU reading for L$l"; FAILED=1; continue; }
  keep=""
  while read -r s chg; do
    if awk -v c="$chg" -v t="$STOP" 'BEGIN { exit !(c < -t) }'; then
      say "L$l multiplier $s: selection-half MMLU change $chg, below -$STOP: this and larger multipliers get no tune answers"
      break
    fi
    keep="$keep x:E$l.md.p$s"
  done < $HOST/$B/mmlu/L$l.budget
  if [ -n "$keep" ]; then
    tune_arms $l caa_L${l}_$TAG $keep || { say "tune L$l failed"; FAILED=1; continue; }
  fi
  say "L$l done (tune arms:$keep)"
done
if [ "$FAILED" -ne 0 ]; then touch ${MK}_FAILED; exit 1; fi
say "done"
touch ${MK}_DONE

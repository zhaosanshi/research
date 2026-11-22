#!/bin/bash
# 2026-11-22 20 h, exploration (Claude Opus 5.5): one model's CAA baseline from its grid to its test and GSM8K readings
# without anybody watching (plan discussion/work_queue/2026-11-22_queue_f.md sections 1, 7 and 8; the ITI chain
# iti_chain.sh + iti_free.sh in one). WRITTEN, NOT RUN: started only after GPT's review. Nothing is chosen by hand:
#   1. the grid (caa_grid.sh MODE=grid: MMLU per layer, early stop, tune answers, judge inputs and cells);
#   2. the budget table (diag/caa_table.py budget);
#   3. DeepSeek Pro on every layer's tune work dir, in parallel (caa_judge.sh: cache of the usual run's LIKE_TUNE work dir
#      merged first; not started in peak hours or too close to them). Every point the early stop reached is judged once,
#      so the reading within the budget and the reading among all measured points come from the same verdicts;
#   4. caa_table.py tune --write-choice (within the budget) and tune --all --write-choice (the best among the measured
#      points), both with the declared grid; existing choice files are kept;
#   5. the direction checks of section 7: multiplier -1 at each chosen layer (caa_grid.sh MODE=neg, judged) and
#      multiplier 0 on the whole tune split at the budget choice's layer (caa_grid.sh MODE=zero);
#   6. the test answers of each chosen point (scripts/review_checks/e44_positions.py gen generate --split test, engine
#      unchanged), the judge input and cells (export names caa_main / caa_all; when both choices are the same point the
#      second is not exported and the table reads the first), DeepSeek Pro with the usual run's LIKE_TEST cache;
#   7. the test tables (caa_table.py test, test --all) and the direction-check table (caa_table.py neg);
#   8. GSM8K at each chosen point, rule gen (scripts/newsets/ns_positions.py gen gen, newsets_gen.py unchanged, token cap
#      TOK, batch 8, neutral system prompt, on GSM_GPU), and the GSM8K table (diag/caa_gsm8k.py, in the container).
# Card 1 is shared: the chain holds the card-1 lock (Iteration_01/newsets/selfjudge/.gpu1.lock) only around its own model
# steps (6 and 8) and waits until the lock is free before each caa_grid.sh call (that script takes the lock itself);
# every model step refuses to start when a card would exceed 65 GB (caa_grid.sh's NEED values).
# Tables: Iteration_01/diag/out_caa_<MODEL>_{budget,tune,tune_all,neg,test,test_all,gsm8k}_20261122.txt.
# usage (host): caa_chain.sh MODEL GPU "LAYER ..." "MULT ..." TAG LIKE_TUNE LIKE_TEST GSM_GPU [TOK]
#   e.g. caa_chain.sh llama3.1-8b 1 "8 10 12 14 16 18 20 22 24" "0.5 1 2 4" g1 pospromptl16 pospromptl16 1 512
#   (layer lists per model: Iteration_01/code/configs/newsets/caa_grid_20261122.json)
# Markers: Iteration_01/logs/newsets_20261119/CAA_CHAIN_<MODEL>_DONE / _FAILED; log lines in caa_all.log there.
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
D=Iteration_01/diag
MODEL=${1:?model}; GPU=${2:?gpu}; LAYERS=${3:?layers}; MULTS=${4:?multipliers}; TAG=${5:?tag}
LT=${6:?like tune}; LX=${7:?like test}; GG=${8:?gsm8k gpu}; TOK=${9:-512}
case "$MODEL" in
  llama3.1-8b) NEED=17000 ;; qwen3.5-9b) NEED=20000 ;; gemma3-12b) NEED=26000 ;;
  gemma3-27b|qwen3.8-27b) NEED=58000 ;; *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
B=Iteration_01/baselines/caa/$MODEL
RUN=$B/run_caa
MK=$HOST/$L/CAA_CHAIN_${MODEL}
log=$HOST/$L/caa_chain_$MODEL.log
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu1.lock
say() { echo "$(date '+%F %T') caa chain $MODEL: $*" >> $HOST/$L/caa_all.log; }
rm -f ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
[ -f $HOST/$B/acceptance/ACCEPT_DONE ] || fail "no passed acceptance (caa_accept.sh)"
read -r CODE PY SRC E_ON < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["code"], m["python"], m["source_run"], m["e_on"])' $HOST/$B/models_run_caa.json "$MODEL") || fail "model not in the mirror config"
[ -n "$E_ON" ] || fail "model not in the mirror config"
CFG=$ROOT/$B/models_run_caa.json
ENV=(-u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models -e TD_DATA_DIR=$ROOT/code/data/processed
     -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
on1() { [[ ",$1," == *",1,"* ]]; }
wait_free() { on1 "$1" || return 0; until flock -n "$LOCK" true; do sleep 60; done; }
lock_on() { on1 "$1" || return 0; mkdir -p "$(dirname "$LOCK")" && exec 8>"$LOCK" && flock -w 86400 8 || fail "cannot take the card-1 lock"; }
lock_off() { exec 8>&-; }
mem_ok() {  # mem_ok GPUS: every card has room for NEED / number of cards
  local g used nc
  nc=$(echo "$1" | tr ',' '\n' | grep -c .)
  for g in ${1//,/ }; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$g" '$1 == g {print $2}')
    [[ $used =~ ^[0-9]+$ ]] || return 1
    [ $((used + NEED / nc)) -le 66560 ] || { say "GPU $g holds $used MiB; with $((NEED / nc)) MiB more it exceeds 65 GB"; return 1; }
  done
}
grid() {  # grid MODE LAYERS MULTS ; caa_grid.sh exit 3 = card busy or full: wait and retry (at most 4 h)
  local rc i
  for i in $(seq 1 48); do
    wait_free "$GPU"
    MODE=$1 bash $HOST/$L/caa_grid.sh $MODEL $GPU "$2" "$3" $TAG >> $log 2>&1
    rc=$?
    [ "$rc" -eq 3 ] || return $rc
    sleep 300
  done
  return 3
}
TB=(timeout 900 python3 -B $D/caa_table.py)
OUT=$D/out_caa_$MODEL
judge_all() {  # judge_all LIKE WORK ... : in parallel, skipping work dirs already judged
  local like=$1 w pids=() bad=0 pid
  shift
  for w in "$@"; do
    [ -s $HOST/$w/judge2/single_pro.jsonl ] && [ -e $HOST/$B/CAA_JUDGE_$(basename $w)_DONE ] && continue
    bash $HOST/$L/caa_judge.sh $MODEL $w $like >> $log 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
  return $bad
}
say "start: GPU $GPU, layers $LAYERS, multipliers $MULTS, tag $TAG, like $LT / $LX, GSM8K on GPU $GG (cap $TOK)"
# 1-2. grid and budget
grid grid "$LAYERS" "$MULTS" || fail "the grid"
"${TB[@]}" budget $MODEL > $OUT"_budget_20261122.txt" 2>&1 || fail "budget table"
# 3. tune verdicts
works=()
for l in $LAYERS; do
  w=$RUN/eval_extra/judge_tune_caa_L${l}_${TAG}_tune
  [ -s $HOST/$w/judge2/cells.jsonl ] && works+=("$w")
done
[ "${#works[@]}" -gt 0 ] || fail "no tune work dir with cells"
judge_all $LT "${works[@]}" || fail "a tune judging"
say "tune judged (${#works[@]} work dirs)"
# 4. choices
[ -e $HOST/$B/choice.json ] || "${TB[@]}" tune $MODEL --layers "$LAYERS" --mults "$MULTS" --write-choice > $OUT"_tune_20261122.txt" 2>&1 || fail "tune table"
[ -e $HOST/$B/choice_all.json ] || "${TB[@]}" tune $MODEL --all --layers "$LAYERS" --mults "$MULTS" --write-choice > $OUT"_tune_all_20261122.txt" 2>&1 || fail "tune table (all measured points)"
pick() { python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c["layer"], "%g" % c["mult"])' "$1"; }
read -r ML MM <<< "$(pick $HOST/$B/choice.json)"; [ -n "$MM" ] || fail "no budget choice"
read -r AL AM <<< "$(pick $HOST/$B/choice_all.json)"; [ -n "$AM" ] || fail "no choice among the measured points"
say "chosen: within the budget layer $ML x $MM; among the measured points layer $AL x $AM"
# 5. direction checks
nw=()
for l in $(printf "%s\n" $ML $AL | awk '!seen[$0]++'); do
  grid neg $l - || fail "multiplier -1 at layer $l"
  nw+=("$RUN/eval_extra/judge_tune_caa_neg_L${l}_${TAG}_tune")
done
judge_all $LT "${nw[@]}" || fail "judging multiplier -1"
[ -e $HOST/$B/check_zero_full_L${ML}_$TAG.json ] || grid zero $ML - || fail "multiplier 0 on the whole tune split"
# 6. test answers and verdicts
COMMON=(--source-run $ROOT/$SRC --run-dir $ROOT/$RUN --e-on $E_ON --split test --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
tokf() { echo "x--E$1-md-p${2//./-}-at-gen_test.json"; }
for pt in "$ML:$MM" "$AL:$AM"; do
  l=${pt%%:*}; m=${pt#*:}
  [ -e $HOST/$RUN/eval_extra/gen/$(tokf $l $m) ] && continue
  lock_on "$GPU"
  mem_ok "$GPU" || { lock_off; fail "no room on GPU $GPU for the test answers"; }
  docker exec "${ENV[@]}" -e CUDA_VISIBLE_DEVICES=$GPU -e PYTHONPATH=$ROOT/$CODE -w $ROOT/$CODE truth_dec \
    $PY scripts/review_checks/e44_positions.py gen generate "${COMMON[@]}" --arms x:E$l.md.p$m >> $log 2>&1 || { lock_off; fail "test answers at layer $l x $m"; }
  lock_off
done
DXC=(docker exec "${ENV[@]}" -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=$ROOT/$CODE -w $ROOT/$CODE truth_dec)
tw=()
for r in "main:$ML:$MM" "all:$AL:$AM"; do
  name=caa_${r%%:*}; rest=${r#*:}; l=${rest%%:*}; m=${rest#*:}
  if [ "$name" = caa_all ] && [ "$l:$m" = "$ML:$MM" ]; then say "the choice among the measured points is the budget choice: test verdicts shared"; continue; fi
  W=$RUN/eval_extra/judge_test_${name}_test
  if [ ! -s $HOST/$W/judge2/cells.jsonl ]; then
    "${DXC[@]}" $PY scripts/review_checks/e44_positions.py gen export "${COMMON[@]}" --gen $(tokf $l $m) --name $name >> $log 2>&1 || fail "test export $name"
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$RUN/eval_extra/judge_test/$name.json $HOST/$W/judge_input.json || fail "test judge input $name"
    "${DXC[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "test cells $name"
  fi
  tw+=("$W")
done
judge_all $LX "${tw[@]}" || fail "test judging"
# 7. tables
"${TB[@]}" test $MODEL > $OUT"_test_20261122.txt" 2>&1 || fail "test table"
"${TB[@]}" test $MODEL --all > $OUT"_test_all_20261122.txt" 2>&1 || fail "test table (all measured points)"
"${TB[@]}" neg $MODEL > $OUT"_neg_20261122.txt" 2>&1 || fail "direction-check table"
say "test read; GSM8K on GPU $GG"
# 8. GSM8K at both choices
mkdir -p $HOST/$B/gsm8k
pts=$(printf "%s\n" "$ML:$MM" "$AL:$AM" | awk '!seen[$0]++')
for pt in $pts; do
  l=${pt%%:*}; m=${pt#*:}
  [ -e $HOST/$B/gsm8k/neutral--E$l-md-p${m//./-}-at-gen.json ] && continue
  lock_on "$GG"
  mem_ok "$GG" || { lock_off; fail "no room on GPU $GG for GSM8K"; }
  docker exec "${ENV[@]}" -e CUDA_VISIBLE_DEVICES=$GG -w $ROOT truth_dec \
    $PY -I $ROOT/Iteration_01/code/scripts/newsets/ns_positions.py gen gen --models $CFG --model $MODEL --task gsm8k \
    --items $ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet --ids $ROOT/Iteration_01/capability/sets/gsm8k_1319_20261119.json \
    --max-new-tokens $TOK --batch-size 8 --out $ROOT/$B/gsm8k --arms neutral:E$l.md.p$m >> $log 2>&1 || { lock_off; fail "GSM8K at layer $l x $m"; }
  lock_off
done
docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -w $ROOT truth_dec python3 -B Iteration_01/diag/caa_gsm8k.py $MODEL --cap $TOK $pts \
  > $HOST/$OUT"_gsm8k_20261122.txt" 2>&1 || fail "GSM8K table"
say "done"
touch ${MK}_DONE

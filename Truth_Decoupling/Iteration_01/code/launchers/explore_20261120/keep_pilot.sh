#!/bin/bash
# 2026-11-20 night, exploration: train the E direction with multiple-choice items among its training questions, so that
# the same training rows ("on the prompt without the honest sentence, the write must reproduce the output of the prompt
# with it") also cover inputs where the sentence changes little. Nothing in the driver or the engine changes: only the
# source directory (scripts/review_checks/make_keep_source.py: the model's training questions + 160 items of the
# 1,000-item MMLU diagnostic list). New run directory Iteration_01/results/ff_keep/<model>/; the frozen method and its
# runs are untouched.
# Steps: source -> freeze train / tune / test (11 prompts, as every run) -> train E at the model's E layer (same
# settings: 10 epochs, 16 held-out questions) -> single-E arms on TruthfulQA tune and test (normal prompt 1 / 1.5 / 2 / 3,
# must-answer prompt 1 / 2), export, cells, DeepSeek Pro (off-peak budget rule) -> MMLU on the full test split (the
# 13,042 items outside the diagnostic list are the read-out) for the same strengths.
# usage: keep_pilot.sh KEY GPU      KEY llama | q9 | q27 | g12 | g27
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
RUNNER=$HOST/Iteration_01/code/scripts/review_checks/ff_explore_ec.sh
TF=$ROOT/envs/tf519/bin/python
KS="0.25 0.5 1 2"
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
KEY=${1:?key}; export GPUO=${2:?gpu}
setm() {
  case $1 in
    llama) MODEL=llama3.1-8b; NG=llama3.1-8b; CODE=$ROOT/code_e45_20261112; PYM=python3; run=results_e44_factorial_das/llama3.1-8b_side_E2
           crun=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; EL=11; CL=19; SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full; GPU=0 ;;
    q9)    MODEL=qwen3.5-9b; NG=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.5-9b_E2
           crun=$E44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; EL=15; CL=16; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b; GPU=0 ;;
    g12)   MODEL=gemma3-12b; NG=gemma3-12b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/12b/e44/gemma3-12b
           crun=$run; E_ON=E2; C_E_ON=E2; EL=22; CL=25; SRC=$ROOT/$G3/12b/source/gemma3-12b; GPU=1 ;;
    g27)   MODEL=gemma3-27b; NG=gemma3-27b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
           crun=$run; E_ON=E2; C_E_ON=E2; EL=26; CL=31; SRC=$ROOT/$G3/27b/source/gemma3-27b; GPU=1 ;;
    q27)   MODEL=qwen3.8-27b-dm; NG=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
           crun=$run; E_ON=E2; C_E_ON=E2; EL=28; CL=33; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm; GPU=2,3 ;;
    *) echo "unknown key $1" >> $L/keep_all.log; exit 2 ;;
  esac
  [ -z "${GPUO:-}" ] || GPU=$GPUO
  E_RUN=$ROOT/$run; C_RUN=$ROOT/$crun
  DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
}
budget() {  # seconds this lane may run now (0 = peak hour)
  local dow=$(date +%u) hm=$((10#$(date +%H%M))) now=$(date +%s) t
  if [ $dow -ge 6 ]; then echo 7200; return; fi
  if [ $hm -lt 900 ]; then t=$(( $(date -d "09:00" +%s) - now ))
  elif [ $hm -lt 1200 ]; then t=0
  elif [ $hm -lt 1400 ]; then t=$(( $(date -d "14:00" +%s) - now ))
  elif [ $hm -lt 1800 ]; then t=0
  else t=7200; fi
  [ $t -gt 7200 ] && t=7200
  [ $t -lt 180 ] && t=0
  echo $t
}
setm $KEY
OLDRUN=$run
case $KEY in
  llama) OT=judge_tune_ecll19_tune; OX=judge_test_tll19_test ;;
  q9)    OT=judge_tune_ec9b_tune;   OX=judge_test_t9b16_test ;;
  q27)   OT=judge_tune_ec2733_tune; OX=judge_test_t2733_test ;;
  g12)   OT=judge_tune_ec25_tune;   OX=judge_test_t25_test ;;
  g27)   OT=judge_tune_ec31_tune;   OX=judge_test_t31_test ;;
esac
NEW=Iteration_01/results/ff_keep/$NG
NSRC=$ROOT/$NEW/source/$NG; NRUN=$ROOT/$NEW/e44/${NG}_keep; nrun=$NEW/e44/${NG}_keep
log=$L/keep_$KEY.log
say() { echo "keep $KEY $* $(date '+%T')" >> $L/keep_all.log; }
COMMON=(--source-run $NSRC --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
{ echo "=== $(date '+%F %T') $KEY source"
  [ -f $HOST/$NEW/source/$NG/bank.json ] || "${DX[@]}" python3 $ROOT/Iteration_01/code/scripts/review_checks/make_keep_source.py \
    --like $SRC --mmlu $ROOT/data_external/mmlu_20261119/all/test-00000-of-00001.parquet \
    --ids $ROOT/Iteration_01/capability/sets/mmlu_1000_20261119.json --n 160 --seed 20261120 \
    --cap-prompts $ROOT/Iteration_01/code/scripts/capability --out $NSRC; } >> $log 2>&1 || { say "source FAILED"; exit 1; }
say "source ok"
{ echo "=== $(date '+%F %T') $KEY freeze"
  "${DX[@]}" $PYM scripts/e44_factorial_das.py freeze "${COMMON[@]}" --run-dir $NRUN --splits train tune &&
  "${DX[@]}" $PYM scripts/review_checks/e44_split_test.py driver freeze "${COMMON[@]}" --run-dir $NRUN --splits test; } >> $log 2>&1 || { say "freeze FAILED"; exit 1; }
say "freeze ok"
{ echo "=== $(date '+%F %T') $KEY train E L$EL"
  "${DX[@]}" $PYM scripts/e44_factorial_das.py train "${COMMON[@]}" --e-on E2 --run-dir $NRUN --family E --layers $EL --val-n 16; } >> $log 2>&1 || { say "train FAILED"; exit 1; }
say "train ok"
arms="x:E$EL.w.p1 x:E$EL.w.p1.5 x:E$EL.w.p2 x:E$EL.w.p3 conf:E$EL.w.p1 conf:E$EL.w.p2"
for sp in tune test; do
  if [ $sp = tune ]; then XE="scripts/e44_eval_extra.py"; else XE="scripts/review_checks/e44_split_test.py extra"; fi
  EC=(--source-run $NSRC --run-dir $NRUN --e-on E2 --c-run-dir $NRUN --c-e-on E2 --split $sp --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  gen=""; for a in $arms; do f=${a//:/--}; f=${f//./-}; gen="$gen ${f}_$sp.json"; done
  W=$nrun/eval_extra/judge_${sp}_keep_$sp
  { echo "=== $(date '+%F %T') $KEY $sp generate $arms"
    "${DX[@]}" $PYM $XE generate "${EC[@]}" --arms $arms &&
    "${DX[@]}" $PYM $XE export "${EC[@]}" --gen $gen --name keep_$sp &&
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$nrun/eval_extra/judge_$sp/keep_$sp.json $HOST/$W/judge_input.json &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json; } >> $log 2>&1 || { say "tq $sp FAILED"; continue; }
  say "tq $sp ok"
done
mkdir -p Iteration_01/capability/keep
python3 - "$NG" "$CODE" "$PYM" "$NEW" "$EL" "$CL" <<'PYEOF'
import json, sys
ng, code, pym, new, el, cl = sys.argv[1:]
root = "/workspace/Truth_Decoupling/"
entry = {"code": code.replace(root, ""), "python": pym.replace(root, ""), "gpus": 1, "source_run": "%s/source/%s" % (new, ng),
         "run_dir": "%s/e44/%s_keep" % (new, ng), "e_on": "E2", "el": int(el), "cl": int(cl)}
json.dump({"note": "E trained with multiple-choice items among its training questions (keep_pilot.sh); single E only",
           "models": {ng: entry}}, open("Iteration_01/capability/keep/models_%s.json" % ng, "w"), indent=1)
PYEOF
rm -f Iteration_01/capability/keep/$NG.json
docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_VISIBLE_DEVICES=$GPU -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $ROOT truth_dec \
  $PYM -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/capability/keep/models_$NG.json \
  --model $NG --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_14042_20261120.json \
  --out Iteration_01/capability/keep/$NG.json --arms x "x:E#.w.p1" "x:E#.w.p1.5" "x:E#.w.p2" "x:E#.w.p3" > $L/keep_mm_$NG.log 2>&1
say "mmlu exit=$?"
for sp in tune test; do
  W=$nrun/eval_extra/judge_${sp}_keep_$sp
  [ -f $HOST/$W/judge2/cells.jsonl ] || continue
  if [ $sp = tune ]; then oc=$OLDRUN/eval_extra/$OT/judge2/cache; else oc=$OLDRUN/eval_extra/$OX/judge2/cache; fi
  secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
  { echo "=== $(date '+%F %T') $KEY $sp: merge $oc, then pro (at most $secs s)"
    { [ ! -f $HOST/$oc/judge2_cache_deepseek-v4-pro.jsonl ] || "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $ROOT/$oc; } &&
    "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json &&
    "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro &&
    "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro > $HOST/$W/single_pro_stdout.md; } >> $log 2>&1
  say "judge $sp exit=$?"
done
touch $L/KEEP_${KEY}_DONE

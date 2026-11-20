#!/bin/bash
# 2026-11-21 00:3x: E strengths 0.5 and 0.75 for the two models whose MMLU already moves by more than 1.3 points at
# strength 1 (Llama-3.1-8B, Gemma-3-27B), so that the per-model table (diag/per_model_table.py) has a strength within a
# small MMLU budget for them. Same arms as dose_mid_all.sh (single E, E + C at 0.25 / 0.5 / 1 / 2; selfcheck before each
# split), tune then test, work dirs judge_<split>_doselow_<split>; then MMLU on the full test split at the two strengths
# (capability/full_low). DeepSeek Pro off-peak (budget rule of pm2_judge.sh), merging the reply cache of the model's
# E+C work dir of the same split. Llama on GPU 0 now; Gemma-3-27B on GPU 2 after pm3_gemma.sh.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
RUNNER=$HOST/Iteration_01/code/scripts/review_checks/ff_explore_ec.sh
TF=$ROOT/envs/tf519/bin/python
KS="0.25 0.5 1 2"
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
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
    *) echo "unknown key $1" >> $L/low_all.log; exit 2 ;;
  esac
  [ -z "${GPUO:-}" ] || GPU=$GPUO
  E_RUN=$ROOT/$run; C_RUN=$ROOT/$crun
  DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
}
n1arms() { local s k o=""; for s in $1; do o="$o x:E$EL.w.p$s"; for k in $KS; do o="$o x:E$EL.w.p$s+C$CL.w.m$k"; done; done; echo $o; }
tq() {  # tq KEY SPLIT NAME SELFCHECK_ES(- for none) TAG ARMS...
  local key=$1 sp=$2 NAME=$3 es=$4 tag=$5; shift 5
  local arms="$*" XE gen="" a f rc=0 log=$L/low_$key.log
  setm $key
  if [ $sp = tune ]; then XE="scripts/e44_eval_extra.py"; else XE="scripts/review_checks/e44_split_test.py extra"; fi
  if [ "$es" != "-" ]; then
    MODEL=$MODEL CODE_PATH=$CODE TABLE_CODE=$ROOT/Iteration_01/code SOURCE_DIR=$SRC E_RUN=$E_RUN C_RUN=$C_RUN E_ON=$E_ON C_E_ON=$C_E_ON \
      EL=$EL CL=$CL ES="$es" KS="$KS" GPU=$GPU SPLIT=$sp TAG=$tag STAGE=selfcheck bash $RUNNER >> $log 2>&1 ||
      { echo "low tq $key $NAME selfcheck FAILED $(date '+%T')" >> $L/low_all.log; return 11; }
  fi
  local COMMON=(--source-run $SRC --run-dir $E_RUN --e-on $E_ON --c-run-dir $C_RUN --c-e-on $C_E_ON --split $sp
                --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  for a in $arms; do f=${a//:/--}; f=${f//+/-and-}; f=${f//./-}; gen="$gen ${f}_$sp.json"; done
  local W=$run/eval_extra/judge_${sp}_$NAME
  { echo "=== $(date '+%F %T') $key $sp $NAME generate $arms"
    "${DX[@]}" $PYM $XE generate "${COMMON[@]}" --arms $arms &&
    "${DX[@]}" $PYM $XE export "${COMMON[@]}" --gen $gen --name $NAME &&
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$sp/$NAME.json $HOST/$W/judge_input.json &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json; } >> $log 2>&1 || rc=1
  echo "low tq $key $NAME exit=$rc $(date '+%T')" >> $L/low_all.log
  return $rc
}
mm() {  # mm KEY OUTDIR ARMS...
  local key=$1 out=$2; shift 2
  setm $key
  mkdir -p Iteration_01/capability/$out; rm -f Iteration_01/capability/$out/$NG.json
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$GPU -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $ROOT truth_dec \
    $PYM -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model $NG --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_14042_20261120.json \
    --out Iteration_01/capability/$out/$NG.json --arms "$@" > $L/low_mm_${out}_$NG.log 2>&1
  echo "low mmlu $out $key exit=$? $(date '+%T')" >> $L/low_all.log
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
judge() {  # judge KEY WORK OLD_CACHE_WORK
  local key=$1 W=$2 oc=$3/judge2/cache log=$L/low_judge_$1.log secs
  setm $key
  local JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  [ -f $HOST/$W/judge2/cells.jsonl ] || { echo "low judge $key: no cells in $W" >> $L/low_all.log; return 1; }
  secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
  { echo "=== $(date '+%F %T') $key: merge $oc, then pro on $W (at most $secs s)"
    { [ ! -f $HOST/$oc/judge2_cache_deepseek-v4-pro.jsonl ] || "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $ROOT/$oc; } &&
    "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json &&
    "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro &&
    "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro > $HOST/$W/single_pro_stdout.md; } >> $log 2>&1
  echo "low judge $key $(basename $W) exit=$? $(date '+%T')" >> $L/low_all.log
}
one() {  # one KEY OLD_TUNE_WORK OLD_TEST_WORK
  local key=$1 ot=$2 ox=$3 sp A
  setm $key; A=$(n1arms "0.5 0.75")
  for sp in tune test; do tq $key $sp doselow_$sp "0.5 0.75" doselow $A; done
  mm $key full_low x "x:E#.w.p0.5" "x:E#.w.p0.75"
  touch $L/LOW_GPU_${key}_DONE
  setm $key
  judge $key $run/eval_extra/judge_tune_doselow_tune $run/eval_extra/$ot
  judge $key $run/eval_extra/judge_test_doselow_test $run/eval_extra/$ox
}
rm -f $L/low_all.log $L/LOW_DONE $L/LOW_GPU_llama_DONE $L/LOW_GPU_g27_DONE
echo "$(date '+%F %T') start" >> $L/low_all.log
( export GPUO=0; one llama judge_tune_ecll19_tune judge_test_tll19_test ) &
( until [ -e $L/PM3_DONE ]; do sleep 60; done; export GPUO=2; one g27 judge_tune_ec31_tune judge_test_t31_test ) &
wait
touch $L/LOW_DONE

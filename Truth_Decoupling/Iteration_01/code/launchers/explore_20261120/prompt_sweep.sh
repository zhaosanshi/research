#!/bin/bash
# 2026-11-20 night: every prompt of each run as a prompt-only baseline (no write). The comparison "N1 against the
# prompt-only baseline" used one prompt (conf_E2 = must-answer + the honest sentence). A run's frozen answers hold the
# no-write answers of all its prompts (neutral / must-answer / cautious base, with or without the sentences) for the
# tune and the test split, so no model is needed: export them (e44_eval_extra.py export with no generation file and
# --prompts-out), build the judge cells, DeepSeek Pro off-peak (budget rule of pm2_judge.sh; the reply cache of the
# model's must-answer work dir of the same split is merged first). The best prompt is then chosen on tune per model
# (diag/prompt_sweep_table.py) and N1 is compared with it on test.
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
    *) echo "unknown key $1" >> $L/promptsweep_all.log; exit 2 ;;
  esac
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
one() {  # one KEY SPLIT OLD_CACHE_WORK
  local key=$1 sp=$2 oc=$3/judge2/cache XE names secs rc=0 log=$L/promptsweep_$1.log
  setm $key
  if [ $sp = tune ]; then XE="scripts/e44_eval_extra.py"; else XE="scripts/review_checks/e44_split_test.py extra"; fi
  local COMMON=(--source-run $SRC --run-dir $E_RUN --e-on $E_ON --c-run-dir $C_RUN --c-e-on $C_E_ON --split $sp
                --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  names=$(python3 -c "import json,sys; print(' '.join(sorted(json.load(open(sys.argv[1]))['prompt_systems'])))" $HOST/$run/frozen_$sp.json) || return 1
  local NAME=promptsweep_$sp W=$run/eval_extra/judge_${sp}_promptsweep_$sp
  local JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  { echo "=== $(date '+%F %T') $key $sp export prompts: $names"
    "${DX[@]}" $PYM $XE export "${COMMON[@]}" --gen --prompts-out $names --name $NAME &&
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$sp/$NAME.json $HOST/$W/judge_input.json &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json; } >> $log 2>&1 || rc=1
  echo "promptsweep export $key $sp exit=$rc $(date '+%T')" >> $L/promptsweep_all.log
  [ $rc = 0 ] || return 1
  secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
  { echo "=== $(date '+%F %T') $key $sp: merge $oc, then pro (at most $secs s)"
    { [ ! -f $HOST/$oc/judge2_cache_deepseek-v4-pro.jsonl ] || "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $ROOT/$oc; } &&
    "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json &&
    "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro &&
    "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro > $HOST/$W/single_pro_stdout.md; } >> $log 2>&1
  echo "promptsweep judge $key $sp exit=$? $(date '+%T')" >> $L/promptsweep_all.log
}
balance() {
  docker exec -u 1000:1001 truth_dec python3 -c '
import json, urllib.request
k = open("/workspace/Truth_Decoupling/logs/.deepseek_key", encoding="utf-8").read().strip()
r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + k, "Accept": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=30))
print(json.dumps(d.get("balance_infos"), ensure_ascii=False, separators=(",", ":")))' 2>/dev/null
}
rm -f $L/promptsweep_all.log $L/PROMPTSWEEP_DONE
echo "$(date '+%F %T') start; balance $(balance)" >> $L/promptsweep_all.log
LR=results_e44_factorial_das/llama3.1-8b_side_E2
( one llama tune $LR/judge_tune_proD_E_L11; one llama test $LR/judge_test_testD_E_L11 ) &
( one q9 tune $E44/qwen3.5-9b_E2/judge_tune_proD_E_L15; one q9 test $E44/qwen3.5-9b_E2/judge_test_testD_E_L15 ) &
( one q27 tune $E44/qwen3.8-27b-dm/judge_tune_proD_E_L28; one q27 test $E44/qwen3.8-27b-dm/judge_test_testD_E_L28 ) &
( one g12 tune $G3/12b/e44/gemma3-12b/judge_tune_proD_E_L22; one g12 test $G3/12b/e44/gemma3-12b/judge_test_testD_E_L22 ) &
( one g27 tune $G3/27b/e44/gemma3-27b/judge_tune_proD_E_L26; one g27 test $G3/27b/e44/gemma3-27b/judge_test_testD_E_L26 ) &
wait
echo "$(date '+%F %T') balance after: $(balance)" >> $L/promptsweep_all.log
touch $L/PROMPTSWEEP_DONE

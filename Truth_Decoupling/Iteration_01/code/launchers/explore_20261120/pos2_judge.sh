#!/bin/bash
# DeepSeek Pro (non-thinking) on the work dirs of pos2_all.sh (E written from the last prompt token on, test split),
# off-peak only (same budget rule as pm2_judge.sh); each lane first merges the reply cache of the model's existing E+C
# test work dir (identical requests are not paid twice, and the same answer text keeps the verdict the main tables use).
# Only that one cache: the must-answer work dir holds other replies for 23 of the same requests (Llama), and merged
# caches that disagree stop the lane by design. A work dir without verdicts starts from an empty cache.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
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
lane() {  # lane KEY CODE WORK OLD_CACHE_WORK OLD_CACHE_WORK_2
  local key=$1 CODE=$ROOT/$2 W=$3 oc=$4/judge2/cache oc2=$5/judge2/cache log=$L/pos2_judge_$1.log n=0 secs
  local DX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  until [ -f $HOST/$W/judge2/cells.jsonl ] || [ $n -ge 1200 ]; do sleep 60; n=$((n + 1)); done
  if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then echo "pos2 judge $key: no cells" >> $L/pos2_judge.log; return; fi
  secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
  [ -f $HOST/$W/judge2/single_pro.jsonl ] || { rm -rf $HOST/$W/judge2/cache; mkdir -p $HOST/$W/judge2/cache; }
  { echo "=== $(date '+%F %T') $key: merge $oc, then pro (at most $secs s)"
    { [ ! -f $HOST/$oc/judge2_cache_deepseek-v4-pro.jsonl ] || "${DX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $ROOT/$oc; } &&
    "${DX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json &&
    "${DX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro > $HOST/$W/single_pro_stdout.md; } >> $log 2>&1
  echo "pos2 judge $key exit=$? $(date '+%T')" >> $L/pos2_judge.log
}
rm -f $L/POS2_JUDGE_DONE
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3; LR=results_e44_factorial_das/llama3.1-8b_side_E2
Q9=$E44/qwen3.5-9b_E2; Q27=$E44/qwen3.8-27b-dm; G12=$G3/12b/e44/gemma3-12b; G27=$G3/27b/e44/gemma3-27b
lane llama code_e45_20261112 $LR/eval_extra/judge_test_posgen2_test $LR/eval_extra/judge_test_tll19_test $LR/judge_test_testD_E_L11 &
lane q9 code_ff_explore_20261113b $Q9/eval_extra/judge_test_posgen2_test $Q9/eval_extra/judge_test_t9b16_test $Q9/judge_test_testD_E_L15 &
lane q27 code_ff_explore_20261113b $Q27/eval_extra/judge_test_posgen2_test $Q27/eval_extra/judge_test_t2733_test $Q27/judge_test_testD_E_L28 &
lane g12 Iteration_01/code $G12/eval_extra/judge_test_posgen2_test $G12/eval_extra/judge_test_t25_test $G12/judge_test_testD_E_L22 &
lane g27 Iteration_01/code $G27/eval_extra/judge_test_posgen2_test $G27/eval_extra/judge_test_t31_test $G27/judge_test_testD_E_L26 &
wait
touch $L/POS2_JUDGE_DONE

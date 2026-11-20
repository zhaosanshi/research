#!/bin/bash
# DeepSeek Pro (non-thinking) on the work dirs of pm2_all.sh (must-answer + E at the new strengths, test split), off-peak
# only: a lane waits for its cells (at most 20 hours), then for an off-peak moment (peak = Monday-Friday 09-12 and
# 14-18), and runs for at most 2 hours or until the next peak hour, whichever comes first; a rerun resumes from the
# reply caches. Same calls as pm_judge.sh.
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
lane() {  # lane KEY CODE WORK OLD_CACHE_WORK
  local key=$1 CODE=$ROOT/$2 W=$3 oc=$4/judge2/cache log=$L/pm2_judge_$1.log n=0 secs
  local DX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  until [ -f $HOST/$W/judge2/cells.jsonl ] || [ $n -ge 1200 ]; do sleep 60; n=$((n + 1)); done
  if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then echo "pm2 judge $key: no cells" >> $L/pm2_judge.log; return; fi
  secs=$(budget); until [ $secs -gt 0 ]; do sleep 120; secs=$(budget); done
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
  echo "pm2 judge $key exit=$? $(date '+%T')" >> $L/pm2_judge.log
}
rm -f $L/PM2_JUDGE_DONE
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3; LR=results_e44_factorial_das/llama3.1-8b_side_E2
Q9=$E44/qwen3.5-9b_E2; Q27=$E44/qwen3.8-27b-dm; G12=$G3/12b/e44/gemma3-12b; G27=$G3/27b/e44/gemma3-27b
lane llama code_e45_20261112 $LR/eval_extra/judge_test_dmust_test $LR/judge_test_testD_E_L11 &
lane q9 code_ff_explore_20261113b $Q9/eval_extra/judge_test_dmust_test $Q9/judge_test_testD_E_L15 &
lane q27 code_ff_explore_20261113b $Q27/eval_extra/judge_test_dmust_test $Q27/judge_test_testD_E_L28 &
lane g12 Iteration_01/code $G12/eval_extra/judge_test_dmust_test $G12/judge_test_testD_E_L22 &
lane g27 Iteration_01/code $G27/eval_extra/judge_test_dmust_test $G27/judge_test_testD_E_L26 &
wait
touch $L/PM2_JUDGE_DONE

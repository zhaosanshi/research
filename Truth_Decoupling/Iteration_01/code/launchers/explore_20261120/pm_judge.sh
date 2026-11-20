#!/bin/bash
# DeepSeek Pro (non-thinking) on the work dirs of pm_all.sh, off-peak only: waits for START (default 18:00:30), every lane
# stops at END (default 23:55); a rerun resumes from the reply caches. A lane whose cells do not exist yet waits for
# them (until 60 minutes before END). Same calls as dose_mid_judge.sh / extra_judge2.sh. Balance read before and after.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
START=${START:-180030}; END=${END:-23:55}
balance() {
  docker exec -u 1000:1001 truth_dec python3 -c '
import json, urllib.request
k = open("/workspace/Truth_Decoupling/logs/.deepseek_key", encoding="utf-8").read().strip()
r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + k, "Accept": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=30))
print(json.dumps(d.get("balance_infos"), ensure_ascii=False, separators=(",", ":")))' 2>/dev/null
}
lane() {  # lane KEY CODE WORK OLD_CACHE_WORK
  local key=$1 CODE=$ROOT/$2 W=$3 oc=$4/judge2/cache log=$L/pm_judge_$1.log
  local DX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  until [ -f $HOST/$W/judge2/cells.jsonl ] || [ $(( $(date -d "$END" +%s) - $(date +%s) )) -lt 3600 ]; do sleep 60; done
  local secs=$(( $(date -d "$END" +%s) - $(date +%s) ))
  if [ ! -f $HOST/$W/judge2/cells.jsonl ] || [ $secs -lt 120 ]; then echo "pm judge $key: no cells or outside the window" >> $L/pm_judge.log; return; fi
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
  echo "pm judge $key exit=$? $(date '+%T')" >> $L/pm_judge.log
}
rm -f $L/PM_JUDGE_DONE
until [ "$(date +%H%M%S)" -ge "$START" ]; do sleep 20; done
echo "$(date '+%F %T') balance before: $(balance)" >> $L/pm_judge.log
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3; LR=results_e44_factorial_das/llama3.1-8b_side_E2
Q9=$E44/qwen3.5-9b_E2; Q27=$E44/qwen3.8-27b-dm; G12=$G3/12b/e44/gemma3-12b; G27=$G3/27b/e44/gemma3-27b
CL=code_e45_20261112; CQ=code_ff_explore_20261113b; CG=Iteration_01/code
lane A_llama $CL $LR/eval_extra/judge_tune_ideaA6_tune $LR/eval_extra/judge_tune_ecll19_tune &
lane A_q9 $CQ $Q9/eval_extra/judge_tune_ideaA6_tune $Q9/eval_extra/judge_tune_ec9b_tune &
lane A_q27 $CQ $Q27/eval_extra/judge_tune_ideaA6_tune $Q27/eval_extra/judge_tune_ec2733_tune &
lane A_g12 $CG $G12/eval_extra/judge_tune_ideaA6_tune $G12/eval_extra/judge_tune_ec25_tune &
lane A_g27 $CG $G27/eval_extra/judge_tune_ideaA6_tune $G27/eval_extra/judge_tune_ec31_tune &
lane high_q9_tune $CQ $Q9/eval_extra/judge_tune_dosehigh2_tune $Q9/eval_extra/judge_tune_ec9b_tune &
lane high_q9_test $CQ $Q9/eval_extra/judge_test_dosehigh2_test $Q9/eval_extra/judge_test_t9b16_test &
lane high_q27_tune $CQ $Q27/eval_extra/judge_tune_dosehigh_tune $Q27/eval_extra/judge_tune_ec2733_tune &
lane high_q27_test $CQ $Q27/eval_extra/judge_test_dosehigh_test $Q27/eval_extra/judge_test_t2733_test &
for k in mdg md; do
  lane ${k}_llama $CL $LR/eval_extra/judge_test_${k}_test $LR/eval_extra/judge_test_tll19_test &
  lane ${k}_q9 $CQ $Q9/eval_extra/judge_test_${k}_test $Q9/eval_extra/judge_test_t9b16_test &
  lane ${k}_q27 $CQ $Q27/eval_extra/judge_test_${k}_test $Q27/eval_extra/judge_test_t2733_test &
  lane ${k}_g12 $CG $G12/eval_extra/judge_test_${k}_test $G12/eval_extra/judge_test_t25_test &
  lane ${k}_g27 $CG $G27/eval_extra/judge_test_${k}_test $G27/eval_extra/judge_test_t31_test &
done
wait
echo "$(date '+%F %T') balance after: $(balance)" >> $L/pm_judge.log
touch $L/PM_JUDGE_DONE

#!/bin/bash
# DeepSeek Pro on the dosemid work dirs: ten lanes (five models x test / tune), off-peak hours only. Waits for START
# (default 12:00:30) and for the generation chain; every lane is stopped at END (default 13:57); a rerun resumes from
# the reply caches. Each lane first merges the reply cache of the model's existing E+C work dir of the same split
# (identical requests are not paid twice; a conflicting key stops the lane, as in ff_explore_ec.sh), then runs
# judge_two_axis_run.py llm / single exactly as ff_explore_ec.sh does. The balance is read before and after.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
START=${START:-120030}; END=${END:-13:57}
balance() {
  docker exec -u 1000:1001 truth_dec python3 -c '
import json, urllib.request
k = open("/workspace/Truth_Decoupling/logs/.deepseek_key", encoding="utf-8").read().strip()
r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + k, "Accept": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=30))
print(json.dumps(d.get("balance_infos"), ensure_ascii=False, separators=(",", ":")))' 2>/dev/null
}
lane() {  # lane KEY CODE E_RUN SPLIT OLD_WORK_NAME
  local key=$1 CODE=$ROOT/$2 run=$3 sp=$4 old=$5
  local WORK=$ROOT/$run/eval_extra/judge_${sp}_dosemid_$sp hw=$HOST/$run/eval_extra/judge_${sp}_dosemid_$sp
  local oc=$run/eval_extra/$old/judge2/cache log=$L/dosemid_judge_${key}_$sp.log
  local DX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  local secs=$(( $(date -d "$END" +%s) - $(date +%s) ))
  if [ $secs -lt 120 ]; then echo "dosemid judge $key $sp: outside the window" >> $L/dosemid_judge.log; return; fi
  if [ ! -f $hw/judge2/cells.jsonl ] || [ ! -f $HOST/$oc/judge2_cache_deepseek-v4-pro.jsonl ]; then
    echo "dosemid judge $key $sp: no cells or no old cache ($oc)" >> $L/dosemid_judge.log; return; fi
  { echo "=== $(date '+%F %T') $key $sp: merge $oc, then pro (at most $secs s)"
    "${DX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $WORK/judge2/cache $ROOT/$oc &&
    "${DX[@]}" python3 -c "
import json, sys
k = json.load(open(sys.argv[1])).get('keys', {})
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $WORK/judge2/cache/conflicts.json &&
    "${DX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $WORK --judge pro &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $WORK --judge pro > $hw/single_pro_stdout.md; } >> $log 2>&1
  echo "dosemid judge $key $sp exit=$? $(date '+%T')" >> $L/dosemid_judge.log
}
rm -f $L/DOSEMID_JUDGE_DONE
until [ "$(date +%H%M%S)" -ge "$START" ] && [ -e $L/DOSEMID_TQ_DONE ]; do sleep 20; done
echo "$(date '+%F %T') balance before: $(balance)" >> $L/dosemid_judge.log
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
for sp in tune test; do
  t() { [ $sp = tune ] && echo "$1" || echo "$2"; }
  lane llama code_e45_20261112 results_e44_factorial_das/llama3.1-8b_side_E2 $sp $(t judge_tune_ecll19_tune judge_test_tll19_test) &
  lane q9 code_ff_explore_20261113b $E44/qwen3.5-9b_E2 $sp $(t judge_tune_ec9b_tune judge_test_t9b16_test) &
  lane q27 code_ff_explore_20261113b $E44/qwen3.8-27b-dm $sp $(t judge_tune_ec2733_tune judge_test_t2733_test) &
  lane g12 Iteration_01/code $G3/12b/e44/gemma3-12b $sp $(t judge_tune_ec25_tune judge_test_t25_test) &
  lane g27 Iteration_01/code $G3/27b/e44/gemma3-27b $sp $(t judge_tune_ec31_tune judge_test_t31_test) &
done
wait
echo "$(date '+%F %T') balance after: $(balance)" >> $L/dosemid_judge.log
touch $L/DOSEMID_JUDGE_DONE

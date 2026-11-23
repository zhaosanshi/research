#!/bin/bash
# 2026-11-23, queue L section 9 (executor A): DeepSeek Pro (non-thinking, the judge route of n1s_model.sh) on the TE export
# eval_extra/judge_<split>_fillctl_<split> of each model, in order. Before judging, ONE old reply cache is merged
# (judge2_cache_merge.py; the project rule since 11-20 and queue I 00:03: a new work dir merges only one old cache): the
# must-answer work dir of T2 (the model's d dir, diag/fill_only2.py paths), whose answers the export's baseline_conf_E2
# cells repeat word for word; the merged cache must have no conflicting key. (23:39: merging the SE dir's cache as well
# stopped on one key with two replies, F / T, between the SE dir and the d dir of Llama tune.) Off peak only: a model's judging starts
# only when it is not a weekday 09:00-12:00 / 14:00-18:00 Beijing time and at least 30 minutes remain before the next
# peak (so nothing starts after 08:30), and is stopped 5 minutes before the peak. The balance JSON before and after.
# Markers FILLCTL_JUDGE_<MODEL>_<SPLIT>_DONE / _FAILED; log fillctl_judge.log.
# usage (host): fillctl_judge.sh "SPLITS" "KEYS"       e.g. fillctl_judge.sh "tune test" "llama q9 q27 g12 g27"
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119; log=$L/fillctl_judge.log
SPLITS=$1; KEYS=$2
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') fillctl judge: $*" | tee -a $L/fillctl_all.log; }
F2=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B Iteration_01/diag/fill_only2.py)
say "balance before $(bash $L/balance.sh)"
for k in $KEYS; do
  case $k in
    llama) MODEL=llama3.1-8b; CODE=$ROOT/code_e45_20261112 ;;
    q9)    MODEL=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b ;;
    q27)   MODEL=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b ;;
    g12)   MODEL=gemma3-12b; CODE=$ROOT/Iteration_01/code ;;
    g27)   MODEL=gemma3-27b; CODE=$ROOT/Iteration_01/code ;;
    *) say "unknown key $k"; exit 2 ;;
  esac
  read -r RUN D0 D1 EL SSEL CL <<< "$("${F2[@]}" paths $MODEL 2>> $log)"
  [ -n "$CL" ] || { say "$MODEL: paths failed"; exit 1; }
  JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
      -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  for PH in $SPLITS; do
    MK=$L/FILLCTL_JUDGE_${MODEL}_$PH; rm -f ${MK}_DONE ${MK}_FAILED
    fail() { say "$MODEL $PH FAILED: $*"; touch ${MK}_FAILED; exit 1; }
    W=$RUN/eval_extra/judge_${PH}_fillctl_$PH
    [ -s $HOST/$W/judge2/cells.jsonl ] || fail "no cells in $W"
    if [ $PH = tune ]; then DD=$D0; else DD=$D1; fi
    for old in $DD; do
      [ -d $HOST/$old/judge2/cache ] || { say "$MODEL $PH: no cache in $old (skipped)"; continue; }
      "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $ROOT/$old/judge2/cache >> $log 2>&1 || fail "merge $old"
    done
    "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json >> $log 2>&1 || fail "cache conflict"
    secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
    if [ "$secs" -lt 1500 ]; then fail "not started: peak hours or less than 30 minutes before the next peak"; fi
    say "$MODEL $PH: judging $W (at most $secs s)"
    if "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro >> $log 2>&1 &&
       "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro > $HOST/$W/single_pro_stdout.md 2>> $log; then
      say "$MODEL $PH: judged"; touch ${MK}_DONE
    else
      fail "judge"
    fi
  done
done
say "balance after $(bash $L/balance.sh)"

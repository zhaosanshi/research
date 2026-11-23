#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor F, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_m.md section 1):
# DeepSeek Pro (non-thinking, the two-axis judge config of the code copy) on the C-row arms of crow_gen.sh, no GPU.
# Same calls as the Pro block of newmodel_e44.sh / ff_tune_pro_20261117.sh, without AllenAI (the plan judges with Pro only):
#   prep KEY   tune: export (prompt x + the two 0.25 arms, and the two 0.25 random arms where they exist, of every layer of
#              the model that has no Pro verdict on them) into
#              <run>/judge_tune_crow_C, cells, merge ONE existing Pro reply cache of the same tune answers (Llama / 9B /
#              Qwen3.8-27B: their step-0 cache, the runner default; Gemma-3: the tune E+C grid of 11-17), conflict check,
#              then crow_pro_requests.py (counts and expected cost; no API call)
#   pro KEY    tune: Pro on <run>/judge_tune_crow_C (resumable), single, verdict check
#   tprep KEY  test (g12 / g27 only): crow_pro_requests.py on the 523-question C work dir judge_test_testS_C_L<N1 layer>
#              against the Pro cache of the same model's test E+C grid (no API call)
#   tpro KEY   test (g12 / g27 only): launchers/ff_tune_pro_20261117.sh on that work dir, seeded with that cache
# pro / tpro refuse to start in DeepSeek peak hours (weekdays 09:00-12:00, 14:00-18:00 Beijing) and after 08:30, and stop
# themselves at 08:55 / 13:55 (the Pro pass resumes from pro.jsonl and the cache). Balance is read by balance.sh before and after.
# usage (host): crow_judge.sh prep|pro|tprep|tpro KEY      KEY = llama | q9 | q27 | g12 | g27
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ACT=${1:-}; KEY=${2:-}
G3=Iteration_01/results/ff/gemma3
case $KEY in
  llama) MODEL=llama3.1-8b; CODE=code_e45_20261112; RUN=results_e44_factorial_das/llama3.1-8b
         SRC=results_qa_das_v1u/llama3.1-8b/full; SEED=$RUN/judge_tune_step0/judge2/cache
         LAYERS="$(seq 0 18 | tr '\n' ' ')$(seq 20 30 | tr '\n' ' ')" ;;
  q9)    MODEL=qwen3.5-9b; CODE=code_ff_explore_20261113; RUN=results_newmodels_ff_20261112/e44/qwen3.5-9b
         SRC=results_newmodels_ff_20261112/source/qwen3.5-9b; SEED=$RUN/judge_tune_step0/judge2/cache
         LAYERS="$(seq 0 11 | tr '\n' ' ')13 14 15 $(seq 18 31 | tr '\n' ' ')" ;;
  q27)   MODEL=qwen3.8-27b-dm; CODE=code_ff_explore_20261113b; RUN=results_newmodels_ff_20261112/e44/qwen3.8-27b-dm
         SRC=results_newmodels_ff_20261112/source/qwen3.8-27b-dm; SEED=$RUN/judge_tune_step0/judge2/cache
         LAYERS="19 21 23 25 27 29 31" ;;
  g12)   MODEL=gemma3-12b; CODE=Iteration_01/code; RUN=$G3/12b/e44/gemma3-12b; SRC=$G3/12b/source/gemma3-12b
         SEED=$RUN/eval_extra/judge_tune_ec25_tune/judge2/cache; LAYERS="15 17 19 21 23 25 27"
         TW=$RUN/judge_test_testS_C_L25; TSEED=$RUN/eval_extra/judge_test_t25_test/judge2/cache ;;
  g27)   MODEL=gemma3-27b; CODE=Iteration_01/code; RUN=$G3/27b/e44/gemma3-27b; SRC=$G3/27b/source/gemma3-27b
         SEED=$RUN/eval_extra/judge_tune_ec31_tune/judge2/cache; LAYERS="19 21 23 25 27 29 31 33 35 37"
         TW=$RUN/judge_test_testS_C_L31; TSEED=$RUN/eval_extra/judge_test_t31_test/judge2/cache ;;
  *) echo "usage: crow_judge.sh prep|pro|tprep|tpro llama|q9|q27|g12|g27"; exit 2 ;;
esac
WORK=$RUN/judge_tune_crow_C
log=$L/crow_judge_$MODEL.log
say() { echo "$(date '+%F %T') crow judge $ACT $MODEL: $*" | tee -a $L/crow_all.log >> $log; }
fail() { say "FAILED: $*"; touch $L/CROW_${MODEL}_${ACT}_FAILED; exit 1; }
rm -f $L/CROW_${MODEL}_${ACT}_FAILED
DX=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models
    -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$W/$CODE
    -e DEEPSEEK_API_KEY_FILE=$W/logs/.deepseek_key -w $W/$CODE truth_dec)
pyj() { "${DX[@]}" python3 "$@"; }
offpeak_ok() {  # weekdays: refuse 08:30-12:00 (no new batch after 08:30) and 14:00-18:00; weekends always off-peak
  local t wd
  t=$((10#$(date +%H) * 60 + 10#$(date +%M))); wd=$(date +%u)
  [ $wd -le 5 ] || return 0
  [ $t -ge 510 ] && [ $t -lt 720 ] && return 1
  [ $t -ge 840 ] && [ $t -lt 1080 ] && return 1
  return 0
}
stop_secs() {  # seconds until the next of 08:55 / 13:55 (5 min before a peak); a cut Pro pass resumes from pro.jsonl + cache
  local now t
  now=$(date +%s)
  for t in "today 08:55" "today 13:55" "tomorrow 08:55"; do
    t=$(date -d "$t" +%s); [ $t -gt $now ] && { echo $((t - now)); return; }
  done
}
stop_inner() {  # after a timeout: the docker exec client is gone, stop the judge process inside the container too
  docker exec truth_dec pkill -TERM -f "$1" > /dev/null 2>&1
  sleep 5
  docker exec truth_dec pgrep -f "$1" > /dev/null 2>&1 && docker exec truth_dec pkill -KILL -f "$1" > /dev/null 2>&1
}
balance() { say "balance: $(bash $L/balance.sh)"; }
check() {  # check WORK JUDGE -> "rows bad empty declined"
  pyj -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
empty = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'empty_output_no_judge_vote' and r.get('empty_output') is True]
decl = [r for r in rows if r['status'] != 'scored' and r.get('basis') == 'pro_declined']
print(len(rows), sum(r['status'] != 'scored' for r in rows) - len(empty) - len(decl), len(empty), ','.join(r['cell_id'] for r in decl) or '-')" "$W/$1/judge2/single_$2.jsonl"
}
case $ACT in
prep)
  GEN=""
  for l in $LAYERS; do for a in p0-25 m0-25; do
    f=C_L${l}_x-w-${a}_tune.json
    [ -s $RUN/gen/$f ] || fail "missing $RUN/gen/$f (run crow_gen.sh first)"
    GEN="$GEN $f"
    f=C_L${l}_x-rand-${a}_tune.json            # the random arms where they exist already (Gemma-3: the proS layers of 11-17)
    [ -s $RUN/gen/$f ] && GEN="$GEN $f"
  done; done
  [ -e $WORK/judge2/pro.jsonl ] && fail "$WORK has a Pro pass already: not re-exported"
  say "export $(echo $GEN | wc -w) generation files + prompt x"
  pyj scripts/e44_export_judge.py --source-run $W/$SRC --run-dir $W/$RUN --split tune --gen $GEN --prompts x --name crow_C >> $log 2>&1 \
    || fail "export"
  mkdir -p $WORK/judge2/cache
  [ -f $WORK/judge_input.json ] || cp $RUN/judge_tune/crow_C.json $WORK/judge_input.json
  cmp -s $RUN/judge_tune/crow_C.json $WORK/judge_input.json || fail "judge_input differs from the export"
  pyj scripts/review_checks/judge_two_axis_run.py cells $W/$WORK --generation judge_input.json >> $log 2>&1 || fail "cells"
  [ -f $SEED/judge2_cache_deepseek-v4-pro.jsonl ] || fail "no Pro reply cache in $SEED"
  pyj scripts/review_checks/judge2_cache_merge.py $W/$WORK/judge2/cache $W/$SEED >> $log 2>&1 || fail "cache merge"
  pyj -c "
import json, sys
k = json.load(open(sys.argv[1])).get('keys', {})
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('merged caches give %d request keys different replies: stop' % n if n else 0)" $W/$WORK/judge2/cache/conflicts.json >> $log 2>&1 \
    || fail "cache conflicts"
  pyj -B $W/Iteration_01/diag/crow_pro_requests.py $W/$WORK > $L/crow_cost_$MODEL.txt 2>> $log || fail "request count"
  cat $L/crow_cost_$MODEL.txt >> $log
  say "$(tr '\n' ' ' < $L/crow_cost_$MODEL.txt)"
  touch $L/CROW_${MODEL}_PREP_DONE ;;
pro)
  [ -e $L/CROW_${MODEL}_PREP_DONE ] || fail "run prep first"
  offpeak_ok || fail "peak hours or after 08:30: not started"
  balance
  say "pro on $WORK (stops at 08:55 if still running)"
  timeout --signal=TERM $(stop_secs) "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py llm $W/$WORK --judge pro >> $log 2>&1
  rc=$?
  balance
  [ $rc = 124 ] && { stop_inner "judge_two_axis_run.py llm $W/$WORK --judge pro"; fail "stopped before the peak; resumes with the same command off-peak"; }
  [ $rc = 0 ] || fail "the Pro pass failed (rc $rc)"
  pyj scripts/review_checks/judge_two_axis_run.py single $W/$WORK --judge pro > $WORK/single_pro_stdout.md 2>> $log || fail "single (pro)"
  read -r n bad empty decl <<< "$(check $WORK pro)" || fail "cannot read the Pro verdicts"
  [ "$bad" = 0 ] || fail "Pro left $bad cells without a verdict"
  say "Pro complete: $n cells ($empty empty answers, declined: $decl)"
  touch $L/CROW_${MODEL}_PRO_DONE ;;
tprep)
  [ -n "${TW:-}" ] || fail "test stage only for g12 / g27"
  [ -f $TW/judge2/cells.jsonl ] || fail "no cells in $TW"
  pyj -B $W/Iteration_01/diag/crow_pro_requests.py $W/$TW $W/$TSEED > $L/crow_tcost_$MODEL.txt 2>> $log || fail "request count"
  cat $L/crow_tcost_$MODEL.txt >> $log
  say "$(tr '\n' ' ' < $L/crow_tcost_$MODEL.txt)"
  touch $L/CROW_${MODEL}_TPREP_DONE ;;
tpro)
  [ -e $L/CROW_${MODEL}_TPREP_DONE ] || fail "run tprep first"
  offpeak_ok || fail "peak hours or after 08:30: not started"
  balance
  say "pro on $TW via ff_tune_pro_20261117.sh (seed $TSEED; stops at 08:55 if still running)"
  timeout --signal=TERM $(stop_secs) bash Iteration_01/launchers/ff_tune_pro_20261117.sh $W/Iteration_01/results/crow_20261124/lane_$MODEL \
    $W/$TW@$W/$TSEED >> $log 2>&1
  rc=$?
  [ $rc = 124 ] && stop_inner "review_checks/(judge_two_axis_run\\.py (llm|single) $W/$TW --judge pro|judge2_cache_merge\\.py $W/$TW/judge2/cache )"
  balance
  [ $rc = 0 ] || fail "the test Pro lane ended with rc $rc (124 = stopped at 08:55); see $log"
  touch $L/CROW_${MODEL}_TPRO_DONE ;;
*) echo "usage: crow_judge.sh prep|pro|tprep|tpro KEY"; exit 2 ;;
esac

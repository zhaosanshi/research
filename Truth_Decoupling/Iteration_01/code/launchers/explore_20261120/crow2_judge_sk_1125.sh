#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor H, 2026-11-24 08 h; plan discussion/work_queue/2026-11-24_queue_m.md section 4.2):
# DeepSeek Pro (non-thinking, the two-axis judge config of the code copy) on the arms of crow2_gen.sh, no GPU. The calls of
# crow_judge.sh prep / pro (stage one), with:
#   * work dir <run>/judge_tune_crow2_C, export name crow2_C;
#   * layers = every layer of the model that is not judged by Pro in its own judge_tune_proS_C_L<layer> dir (that dir has
#     no judge2/single_pro.jsonl); four arms per layer (x.w.p0.25, x.w.m0.25, x.rand.p0.25, x.rand.m0.25) plus prompt x;
#   * the reply cache is seeded with ONE old cache: the stage-one <run>/judge_tune_crow_C/judge2/cache;
#   * prep stops before Pro when the expected cost of the batch is above 6 CNY (marker CROW2_<model>_COST_STOP).
#   prep KEY   export, cells, cache merge, conflict check, crow_pro_requests.py (counts and expected cost; no API call)
#   pro KEY    Pro on the work dir (resumable), single, verdict check
# pro refuses to start in DeepSeek peak hours (weekdays 09:00-12:00, 14:00-18:00 Beijing) and after 08:30 (exit 3, no
# FAILED marker), and stops itself at 08:55 / 13:55 (exit 4, no FAILED marker; the pass resumes from pro.jsonl and the
# cache with the same command). Other failures exit 1 with CROW2_<model>_<act>_FAILED. Balance by balance.sh before / after.
# CROW2_NAME=<name> (smoke only): another export name and work dir judge_tune_<name>; markers and logs then carry the
# name (CROW2_<name>_<model>_...); CROW2_LAYERS="..." restricts the layers and is refused with the default name.
# usage (host): crow2_judge.sh prep|pro KEY      KEY = llama | q9 | q27 | g12 | g27
#
# crow2_judge_sk_1125.sh (Claude Opus 5.5, executor yin, 2026-11-25 22 h; plan 2026-11-24_queue_m.md, progress entry
# 2026-11-25 22:11 part one): a copy of crow2_judge.sh (427be6cafcfc5587) for the split-kinds results root of gemma3-27b.
# The differences, all of them:
#   (1) one key only, g27sk: code copy Iteration_01/code_splitkinds_20261125 (its export / cells / judge scripts are
#       byte-identical to Iteration_01/code; only core/qa_factorial_das.py and scripts/e44_factorial_das.py differ),
#       run dir Iteration_01/results/ff/gemma3/27b_splitkinds/e44/gemma3-27b, source .../27b_splitkinds/source/gemma3-27b.
#   (2) seed cache (read only): the main root's Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b/judge_tune_crow2_C_from18/
#       judge2/cache (the new root has no stage-one cache).
#   (3) export name default crow2_C_sk (work dir judge_tune_crow2_C_sk, markers CROW2_crow2_C_sk_gemma3-27b_..., log
#       crow2_judge_crow2_C_sk_gemma3-27b.log, cost file crow2_cost_crow2_C_sk_gemma3-27b.txt); CROW2_LAYERS allowed.
#   (4) cost stop at 5 CNY instead of 6 (cost cap of this task).
# usage (host): CROW2_LAYERS="..." crow2_judge_sk_1125.sh prep|pro g27sk
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ACT=${1:-}; KEY=${2:-}
G3=Iteration_01/results/ff/gemma3
case $KEY in
  g27sk) MODEL=gemma3-27b; CODE=Iteration_01/code_splitkinds_20261125; RUN=$G3/27b_splitkinds/e44/gemma3-27b
         SRC=$G3/27b_splitkinds/source/gemma3-27b ;;
  *) echo "usage: crow2_judge_sk_1125.sh prep|pro g27sk"; exit 2 ;;
esac
NAME=${CROW2_NAME:-crow2_C_sk}
case $NAME in ""|*[!A-Za-z0-9_]*) echo "CROW2_NAME must be alphanumeric / underscore"; exit 2 ;; esac
if [ $NAME = crow2_C ]; then
  [ -z "${CROW2_LAYERS:-}" ] || { echo "CROW2_LAYERS only with a smoke CROW2_NAME"; exit 2; }
  MKB=$L/CROW2_${MODEL}; log=$L/crow2_judge_$MODEL.log; COST=$L/crow2_cost_$MODEL.txt
else
  MKB=$L/CROW2_${NAME}_${MODEL}; log=$L/crow2_judge_${NAME}_$MODEL.log; COST=$L/crow2_cost_${NAME}_$MODEL.txt
fi
SEED=$G3/27b/e44/gemma3-27b/judge_tune_crow2_C_from18/judge2/cache
WORK=$RUN/judge_tune_$NAME
ARMS="x.w.p0.25 x.w.m0.25 x.rand.p0.25 x.rand.m0.25"
MAXCOST=5
say() { echo "$(date '+%F %T') crow2 judge $ACT $MODEL${CROW2_NAME:+ ($NAME)}: $*" | tee -a $L/crow_all.log >> $log; }
fail() { say "FAILED: $*"; touch ${MKB}_${ACT}_FAILED; exit 1; }
rm -f ${MKB}_${ACT}_FAILED
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
  NL=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["num_layers"]))' $CODE/configs/models/$MODEL.json) \
    || fail "num_layers"
  LAYERS=""; OWN=""
  for l in ${CROW2_LAYERS:-$(seq 0 $((NL - 1)))}; do
    if [ -f $RUN/judge_tune_proS_C_L$l/judge2/single_pro.jsonl ]; then OWN="$OWN $l"; else LAYERS="$LAYERS $l"; fi
  done
  [ -n "$LAYERS" ] || fail "no layer to judge"
  GEN=""
  for l in $LAYERS; do for a in $ARMS; do
    f=C_L${l}_${a//./-}_tune.json
    [ -s $RUN/gen/$f ] || fail "missing $RUN/gen/$f (run crow2_gen.sh first)"
    GEN="$GEN $f"
  done; done
  [ -e $WORK/judge2/pro.jsonl ] && fail "$WORK has a Pro pass already: not re-exported"
  say "layers [${LAYERS# }] (own proS Pro dirs, left out: [${OWN# }]); export $(echo $GEN | wc -w) generation files + prompt x"
  pyj scripts/e44_export_judge.py --source-run $W/$SRC --run-dir $W/$RUN --split tune --gen $GEN --prompts x --name $NAME >> $log 2>&1 \
    || fail "export"
  mkdir -p $WORK/judge2/cache
  [ -f $WORK/judge_input.json ] || cp $RUN/judge_tune/$NAME.json $WORK/judge_input.json
  cmp -s $RUN/judge_tune/$NAME.json $WORK/judge_input.json || fail "judge_input differs from the export"
  pyj scripts/review_checks/judge_two_axis_run.py cells $W/$WORK --generation judge_input.json >> $log 2>&1 || fail "cells"
  [ -f $SEED/judge2_cache_deepseek-v4-pro.jsonl ] || fail "no Pro reply cache in $SEED"
  pyj scripts/review_checks/judge2_cache_merge.py $W/$WORK/judge2/cache $W/$SEED >> $log 2>&1 || fail "cache merge"
  pyj -c "
import json, sys
k = json.load(open(sys.argv[1])).get('keys', {})
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('merged caches give %d request keys different replies: stop' % n if n else 0)" $W/$WORK/judge2/cache/conflicts.json >> $log 2>&1 \
    || fail "cache conflicts"
  pyj -B $W/Iteration_01/diag/crow_pro_requests.py $W/$WORK > $COST 2>> $log || fail "request count"
  cat $COST >> $log
  say "$(tr '\n' ' ' < $COST)"
  EXP=$(sed -n 's/^expected cost \([0-9.]*\) CNY.*/\1/p' $COST)
  [[ $EXP =~ ^[0-9]+\.[0-9]+$ ]] || fail "cannot read the expected cost from $COST"
  if awk -v e=$EXP -v m=$MAXCOST 'BEGIN {exit !(e > m)}'; then
    touch ${MKB}_COST_STOP; fail "expected cost $EXP CNY is above $MAXCOST CNY: Pro not started (${MKB}_COST_STOP)"
  fi
  touch ${MKB}_PREP_DONE ;;
pro)
  [ -e ${MKB}_PREP_DONE ] || fail "run prep first"
  [ -e ${MKB}_COST_STOP ] && fail "cost stop (${MKB}_COST_STOP)"
  offpeak_ok || { say "peak hours or after 08:30: not started"; exit 3; }
  balance
  say "pro on $WORK (stops at 08:55 / 13:55 if still running)"
  timeout --signal=TERM $(stop_secs) "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py llm $W/$WORK --judge pro >> $log 2>&1
  rc=$?
  balance
  if [ $rc = 124 ]; then
    stop_inner "judge_two_axis_run.py llm $W/$WORK --judge pro"
    say "stopped before the peak; resumes with the same command off-peak"; exit 4
  fi
  [ $rc = 0 ] || fail "the Pro pass failed (rc $rc)"
  pyj scripts/review_checks/judge_two_axis_run.py single $W/$WORK --judge pro > $WORK/single_pro_stdout.md 2>> $log || fail "single (pro)"
  read -r n bad empty decl <<< "$(check $WORK pro)" || fail "cannot read the Pro verdicts"
  [ "$bad" = 0 ] || fail "Pro left $bad cells without a verdict"
  say "Pro complete: $n cells ($empty empty answers, declined: $decl)"
  touch ${MKB}_PRO_DONE ;;
*) echo "usage: crow2_judge.sh prep|pro KEY"; exit 2 ;;
esac

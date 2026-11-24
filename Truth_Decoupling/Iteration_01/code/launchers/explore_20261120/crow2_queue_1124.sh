#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor H, 2026-11-24 08 h; plan discussion/work_queue/2026-11-24_queue_m.md section 4.3,
# steps 2-5): the unattended order of the second stage of the per-layer C row.
#   step 2  wait for TFG3_gemma3-12b_DONE or _FAILED, then crow2_gen.sh g12 (card 1)
#   step 3  crow2_gen.sh llama, then crow2_gen.sh q9 (card 1)
#   step 4  crow2_gen.sh g27 (card 1)
#   step 5  after step 4 and TFG3_gemma3-27b_DONE or _FAILED: crow2_gen.sh q27 (cards 1,2)
# After each model's generation (GEN_DONE marker) a judge lane starts in the background and the card steps go on:
# crow2_judge.sh prep, then crow2_judge.sh pro in the next off-peak window (waits while it is peak time; retried every
# 10 minutes when the guard refuses (exit 3) or the pass stopped itself before a peak (exit 4); one Pro pass at a time,
# serialized by the lock file .crow2_pro.lock), then the reader's table
#   Iteration_01/diag/out_c_row_layers2_<model>_<yyyymmdd>.txt   (c_row_layers2.py tune KEY)
# and the stage-one row check (c_row_layers2.py check against diag/ref_c_row_layers_five_models_20261124.txt) appended to
# Iteration_01/diag/out_c_row_layers2_<model>_<yyyymmdd>_check.txt.
# A model whose generation or judging fails is reported and skipped; the other models go on.
# Status lines "crow2 queue: ..." in crow_all.log; own log crow2_queue.log; per-step outputs crow2_<step>_<key>.out;
# marker CROW2_QUEUE_DONE when every step and lane has ended.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/crow2_queue_1124.sh \
#                 > Iteration_01/logs/newsets_20261119/crow2_queue.out 2>&1 < /dev/null &
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
Q=$L/crow2_queue.log
REFT=Iteration_01/diag/ref_c_row_layers_five_models_20261124.txt
say() { echo "$(date '+%F %T') crow2 queue: $*" | tee -a $L/crow_all.log >> $Q; }
mkdir $L/.crow2_queue_running 2>/dev/null || { echo "queue already running ($L/.crow2_queue_running)"; exit 3; }
trap 'rmdir $L/.crow2_queue_running 2>/dev/null' EXIT
for f in crow2_gen.sh crow2_judge.sh; do [ -s $L/$f ] || { say "missing $L/$f"; exit 2; }; done
[ -s Iteration_01/diag/c_row_layers2.py ] && [ -s $REFT ] || { say "missing c_row_layers2.py or $REFT"; exit 2; }
model_of() { case $1 in llama) echo llama3.1-8b ;; q9) echo qwen3.5-9b ;; q27) echo qwen3.8-27b-dm ;; g12) echo gemma3-12b ;; g27) echo gemma3-27b ;; esac; }
offpeak_ok() {  # the guard of crow2_judge.sh pro: weekdays refuse 08:30-12:00 and 14:00-18:00
  local t wd
  t=$((10#$(date +%H) * 60 + 10#$(date +%M))); wd=$(date +%u)
  [ $wd -le 5 ] || return 0
  [ $t -ge 510 ] && [ $t -lt 720 ] && return 1
  [ $t -ge 840 ] && [ $t -lt 1080 ] && return 1
  return 0
}
waitfor() {  # waitfor A B: until one of the two markers exists (polled every 5 minutes)
  say "waiting for $1 or $2"
  until [ -e $1 ] || [ -e $2 ]; do sleep 300; done
  say "seen $( [ -e $1 ] && echo $1 || echo $2 )"
}
lane() {  # lane KEY: prep, pro off-peak, reader table
  local key=$1 m rc out
  m=$(model_of $key)
  if [ ! -e $L/CROW2_${m}_PREP_DONE ]; then
    bash $L/crow2_judge.sh prep $key >> $L/crow2_prep_$key.out 2>&1 || { say "$m: prep failed (see crow2_judge_$m.log); lane stopped"; return; }
  fi
  [ -e $L/CROW2_${m}_PREP_DONE ] || { say "$m: no PREP_DONE after prep; lane stopped"; return; }
  while [ ! -e $L/CROW2_${m}_PRO_DONE ]; do
    until offpeak_ok; do sleep 600; done
    exec 7>$L/.crow2_pro.lock
    flock 7
    if offpeak_ok; then
      bash $L/crow2_judge.sh pro $key >> $L/crow2_pro_$key.out 2>&1; rc=$?
    else
      rc=3
    fi
    flock -u 7; exec 7>&-
    case $rc in
      0) ;;
      3|4) say "$m: pro not finished (rc $rc), retry off-peak"; sleep 600 ;;
      *) say "$m: pro failed (rc $rc, see crow2_judge_$m.log); lane stopped"; return ;;
    esac
  done
  out=Iteration_01/diag/out_c_row_layers2_${m}_$(date +%Y%m%d)
  docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec \
    python3 -B Iteration_01/diag/c_row_layers2.py tune $key > $out.txt 2>&1 || { say "$m: reader failed (see $out.txt)"; return; }
  docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec \
    python3 -B Iteration_01/diag/c_row_layers2.py check $REFT $key > ${out}_check.txt 2>&1
  rc=$?
  say "$m: table $out.txt; stage-one row check rc $rc: $(sed -n 2p ${out}_check.txt)"
  touch $L/CROW2_${m}_TABLE_DONE
}
LANES=()
gen() {  # gen KEY: the card step, then the judge lane in the background
  local key=$1 m rc
  m=$(model_of $key)
  say "start crow2_gen.sh $key"
  bash $L/crow2_gen.sh $key >> $L/crow2_gen_$key.out 2>&1; rc=$?
  if [ -e $L/CROW2_${m}_GEN_DONE ]; then
    say "$m: generation done; judge lane started"
    lane $key & LANES+=($!)
  else
    say "$m: generation failed (rc $rc, see crow2_gen_$m.log); next model"
  fi
}
say "start (pid $$)"
waitfor $L/TFG3_gemma3-12b_DONE $L/TFG3_gemma3-12b_FAILED
gen g12
gen llama
gen q9
gen g27
waitfor $L/TFG3_gemma3-27b_DONE $L/TFG3_gemma3-27b_FAILED
gen q27
say "card steps ended; waiting for the judge lanes (${LANES[*]})"
for p in "${LANES[@]}"; do wait $p; done
say "all steps and lanes ended"
touch $L/CROW2_QUEUE_DONE

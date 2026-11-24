#!/bin/bash
# Iteration_01 (main session, Claude Fable 5.2, 2026-11-24 19 h; plan discussion/work_queue/2026-11-24_queue_n.md,
# progress entry of 19 h), exploration: GSM8K on the first 304 frozen items for Llama-3.1-8B, single E at the layer /
# strength pairs the depth scan chose on MMLU (16 at 3, 17 at 8, 18 at 8, 20 at 6, 22 at 8, 24 at 12, 26 at 16), core
# rule "prompt" (scripts/newsets/ns_positions.py, as gsm8k_pos.sh), cap 512, batch 8, one generator call (the model is
# loaded once). Output Iteration_01/capability/depthscan_gsm8k/<model>/b8 (the main directory gen_main is not touched).
# Card CARD (default 1) through Iteration_01/newsets/selfjudge/.gpu<CARD>.lock with a BLOCKING flock, so that this short
# job takes the card at the next release, ahead of a queue that retries once a minute; ours plus others at most
# 66560 MiB. Then the table of diag/depth_gsm.py (no judge). Markers DEPTH_GSM_<model>_DONE / _FAILED, lines in
# depth_all.log, output in depth_gsm_<model>.log.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/depth_gsm_1124.sh [CARD] \
#                 > Iteration_01/logs/newsets_20261119/depth_gsm.out 2>&1 < /dev/null &
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ALL=$L/depth_all.log
CARD=${1:-1}
LOCK=Iteration_01/newsets/selfjudge/.gpu$CARD.lock
MODEL=llama3.1-8b
NEED=20000
MAXMIB=66560
ARMS="neutral:E16.w.p3 neutral:E17.w.p8 neutral:E18.w.p8 neutral:E20.w.p6 neutral:E22.w.p8 neutral:E24.w.p12 neutral:E26.w.p16"
IDS=Iteration_01/capability/sets/gsm8k_first304_20261124.json
OUT=Iteration_01/capability/depthscan_gsm8k/$MODEL/b8
CFG=Iteration_01/code/configs/newsets/frozen_models_20261119.json
say() { echo "$(date '+%F %T') depth gsm: $*" >> $ALL; }
fail() { say "$MODEL FAILED: $*"; touch $L/DEPTH_GSM_${MODEL}_FAILED; exit 1; }
mkdir $L/.depth_gsm_running 2>/dev/null || { echo "already running ($L/.depth_gsm_running)"; exit 3; }
trap 'rmdir $L/.depth_gsm_running 2>/dev/null' EXIT
rm -f $L/DEPTH_GSM_${MODEL}_FAILED
[ -s $IDS ] || fail "missing $IDS (run depth_gsm.py ids)"
[ -s Iteration_01/diag/depth_gsm.py ] || fail "missing Iteration_01/diag/depth_gsm.py"
PY=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' $CFG $MODEL) || fail "python of $MODEL in $CFG"
say "start (pid $$); waiting for the lock of card $CARD"
exec 9>"$LOCK" || fail "cannot open $LOCK"
flock 9 || fail "cannot take $LOCK"
say "holds the lock of card $CARD"
while true; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$CARD '$1 == g {print $2}')
  [[ $used =~ ^[0-9]+$ ]] || fail "cannot read the memory of card $CARD"
  [ $((used + NEED)) -le $MAXMIB ] && break
  say "card $CARD holds $used MiB, $NEED more would exceed $MAXMIB; waiting"; sleep 60
done
say "card $CARD holds $used MiB, asking $NEED; arms: $ARMS"
mkdir -p $OUT
docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$CARD -w $W truth_dec \
  $PY -I $W/Iteration_01/code/scripts/newsets/ns_positions.py prompt gen --models $W/$CFG --model $MODEL --task gsm8k \
  --items $W/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet --ids $W/$IDS --max-new-tokens 512 \
  --batch-size 8 --out $W/$OUT --arms $ARMS >> $L/depth_gsm_$MODEL.log 2>&1
rc=$?
flock -u 9; exec 9>&-
say "generation exit=$rc; card $CARD released"
[ $rc = 0 ] || fail "generation (see $L/depth_gsm_$MODEL.log)"
T=Iteration_01/diag/out_depth_gsm_${MODEL}_$(date +%Y%m%d).txt
docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $W truth_dec \
  python3 -B Iteration_01/diag/depth_gsm.py table $MODEL > $T 2>&1 || fail "table (see $T)"
touch $L/DEPTH_GSM_${MODEL}_DONE
say "table $T"

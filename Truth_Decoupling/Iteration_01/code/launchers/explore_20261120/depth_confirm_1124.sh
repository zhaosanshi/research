#!/bin/bash
# Iteration_01 (main session, Claude Fable 5.2, 2026-11-24 20 h; plan discussion/work_queue/2026-11-24_queue_n.md,
# progress entry of 20 h), exploration: the late single-E setting of Llama-3.1-8B picked on the tune split (layer 17,
# strength 8: the highest tune net among the layers after the main-table layer that pass both screens, MMLU 1,000
# items within 1.0 point and GSM8K first 304 items not below -3.0 points) on the TEST split, plus the two capability
# checks on the full lists. Card part only; judging is started by hand after the reading rule is committed.
#   1 test split (plain prompt, core rule "prompt", e44_positions.py as depth_queue3_1124.sh): single E at
#     17 / 8 and, generated but not exported, 20 / 6, 22 / 8, 24 / 12, 26 / 16, 26 / 12; positions checked;
#     export and cells of 17 / 8 only (work dir <E run>/eval_extra/judge_test_depthE_test)
#   2 GSM8K, the 1,319 frozen items, neutral:E17.w.p8 at "prompt", cap 512, batch 8, into gen_main (as gsm8k_pos.sh)
#   3 MMLU, the full list mmlu_14042_20261120.json, arms x and x:E17.w.p8 (cap_mcq_sweep.py, x first), into
#     Iteration_01/capability/depthscan/llama3.1-8b_full_L17.json
# Card CARD (default 1) through its lock with a BLOCKING flock (takes the card at the next release); ours plus others
# at most 66560 MiB. Markers DEPTH_CONFIRM_llama3.1-8b_GEN_DONE / _FAILED; lines in depth_all.log; output in
# depth_confirm_llama3.1-8b.log.
# usage (host): nohup setsid bash Iteration_01/logs/newsets_20261119/depth_confirm_1124.sh [CARD] \
#                 > Iteration_01/logs/newsets_20261119/depth_confirm.out 2>&1 < /dev/null &
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ALL=$L/depth_all.log
CARD=${1:-1}
LOCK=Iteration_01/newsets/selfjudge/.gpu$CARD.lock
NG=llama3.1-8b; NEED=20000; MAXMIB=66560
CODE=$W/code_e45_20261112; PYM=python3
RUN=results_e44_factorial_das/llama3.1-8b_side_E2; CRUN=results_e44_factorial_das/llama3.1-8b
SRC=$W/results_qa_das_v1u/llama3.1-8b/full
PICK=x:E17.w.p8
TARMS="x:E17.w.p8 x:E20.w.p6 x:E22.w.p8 x:E24.w.p12 x:E26.w.p16 x:E26.w.p12"
CFG=Iteration_01/code/configs/newsets/frozen_models_20261119.json
MLOG=$L/depth_confirm_$NG.log
say() { echo "$(date '+%F %T') depth confirm: $*" >> $ALL; }
fail() { say "$NG FAILED: $* (see $MLOG)"; touch $L/DEPTH_CONFIRM_${NG}_FAILED; exit 1; }
st() { local name=$1 rc; shift; echo "=== $(date '+%F %T') $name: $*" >> $MLOG; "$@" >> $MLOG 2>&1; rc=$?; say "$NG $name exit=$rc"; return $rc; }
fname() { local f=${1//:/--}; f=${f//+/-and-}; f=${f//./-}; echo "${f}-at-prompt_test.json"; }
mkdir $L/.depth_confirm_running 2>/dev/null || { echo "already running ($L/.depth_confirm_running)"; exit 3; }
trap 'rmdir $L/.depth_confirm_running 2>/dev/null' EXIT
rm -f $L/DEPTH_CONFIRM_${NG}_FAILED
DX=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models
    -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES=$CARD -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
COMMON=(--source-run $SRC --run-dir $W/$RUN --e-on E2 --c-run-dir $W/$CRUN --c-e-on E1 --split test
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
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
say "card $CARD holds $used MiB, asking $NEED"
# 1 test split
st gen_test "${DX[@]}" $PYM scripts/review_checks/e44_positions.py prompt generate "${COMMON[@]}" --arms $TARMS || fail "test generation"
st pos_test python3 -c '
import json, sys
bad = 0
for p in sys.argv[1:]:
    d = json.load(open(p))
    ok = d.get("positions") == "prompt" and d.get("generation_contract", {}).get("positions") == "prompt" and d.get("completed") and not d.get("partial") and d.get("split") == "test"
    print(p.split("/")[-1], "rows", len(d["rows"]), "ok" if ok else "BAD")
    bad += not ok
sys.exit(1 if bad else 0)' $(for a in $TARMS; do echo $RUN/eval_extra/gen/$(fname $a); done) || fail "test generation files (positions / completeness)"
WK=$RUN/eval_extra/judge_test_depthE_test
if [ ! -f $WK/judge2/cells.jsonl ]; then
  st export_test "${DX[@]}" $PYM scripts/review_checks/e44_positions.py prompt export "${COMMON[@]}" --gen $(fname $PICK) --name depthE_test || fail "test export"
  mkdir -p $WK/judge2/cache && cp $RUN/eval_extra/judge_test/depthE_test.json $WK/judge_input.json || fail "test work dir"
  st cells_test "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W/$WK --generation judge_input.json || fail "test cells"
fi
touch $L/DEPTH_CONFIRM_${NG}_TESTGEN_DONE
# 2 GSM8K, full list
st gsm_full docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$CARD -w $W truth_dec \
  $PYM -I $W/Iteration_01/code/scripts/newsets/ns_positions.py prompt gen --models $W/$CFG --model $NG --task gsm8k \
  --items $W/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet --ids $W/Iteration_01/capability/sets/gsm8k_1319_20261119.json \
  --max-new-tokens 512 --batch-size 8 --out $W/Iteration_01/capability/gen_main/gsm8k/$NG/b8 --arms neutral:E17.w.p8 || fail "GSM8K full list"
# 3 MMLU, full list
OUTM=Iteration_01/capability/depthscan/${NG}_full_L17.json
if [ ! -s $OUTM ]; then
  st mmlu_full docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$CARD -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $W truth_dec \
    $PYM -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models $CFG --model $NG \
    --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_14042_20261120.json \
    --out $OUTM --arms x x:E17.w.p8 || fail "MMLU full list"
fi
flock -u 9; exec 9>&-
say "card $CARD released"
touch $L/DEPTH_CONFIRM_${NG}_GEN_DONE
say "card part done"

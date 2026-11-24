#!/bin/bash
# Iteration_01 (main session, Claude Fable 5.2, 2026-11-24 21 h; plan discussion/work_queue/2026-11-24_queue_n.md,
# progress entry of 21:30), exploration: Qwen3.5-9B, the next late single-E candidates after the first three failed
# the GSM8K screen (17 / 8, 20 / 12, 22 / 16 lost 21, 68 and 69 points). Derived by program from
# depth_late_q9_1124.sh. Candidates = the settings of the layers after the main-table layer that have tune verdicts:
#   20 / 6, 17 / 6, 16 / 3.5, 16 / 5, 18 / 5, 24 / 16, 18 / 6 (in the order of their tune net).
#   1 test split (plain prompt, core rule "prompt"): the four settings without a test generation yet (20 / 6, 17 / 6,
#     16 / 3.5, 18 / 5); nothing is exported here (the setting the rule picks is exported and judged afterwards)
#   2 GSM8K, the first 304 frozen items, cap 1024, the seven candidates
# Card CARD (default 1) through its lock with a BLOCKING flock; ours plus others at most 66560 MiB. Markers
# DEPTH_LATE2_qwen3.5-9b_TESTGEN_DONE, _GEN_DONE / _FAILED; output in depth_late2_qwen3.5-9b.log.
# usage (host): (nohup setsid bash Iteration_01/logs/newsets_20261119/depth_late2_q9_1124.sh [CARD] \
#                 > Iteration_01/logs/newsets_20261119/depth_late2_q9.out 2>&1 < /dev/null &)
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ALL=$L/depth_all.log
CARD=${1:-1}
LOCK=Iteration_01/newsets/selfjudge/.gpu$CARD.lock
NG=qwen3.5-9b; NEED=22000; MAXMIB=66560
CODE=$W/code_ff_explore_20261113b; PYM=$W/envs/tf519/bin/python
RUN=results_newmodels_ff_20261112/e44/qwen3.5-9b_E2; CRUN=results_newmodels_ff_20261112/e44/qwen3.5-9b
SRC=$W/results_newmodels_ff_20261112/source/qwen3.5-9b
TARMS="x:E20.w.p6 x:E17.w.p6 x:E16.w.p3.5 x:E18.w.p5"
GARMS="neutral:E20.w.p6 neutral:E17.w.p6 neutral:E16.w.p3.5 neutral:E16.w.p5 neutral:E18.w.p5 neutral:E24.w.p16 neutral:E18.w.p6"
CFG=Iteration_01/code/configs/newsets/frozen_models_20261119.json
MLOG=$L/depth_late2_$NG.log
say() { echo "$(date '+%F %T') depth late2 q9: $*" >> $ALL; }
fail() { say "$NG FAILED: $* (see $MLOG)"; touch $L/DEPTH_LATE2_${NG}_FAILED; exit 1; }
st() { local name=$1 rc; shift; echo "=== $(date '+%F %T') $name: $*" >> $MLOG; "$@" >> $MLOG 2>&1; rc=$?; say "$NG $name exit=$rc"; return $rc; }
fname() { local f=${1//:/--}; f=${f//+/-and-}; f=${f//./-}; echo "${f}-at-prompt_test.json"; }
mkdir $L/.depth_late2_q9_running 2>/dev/null || { echo "already running ($L/.depth_late2_q9_running)"; exit 3; }
trap 'rmdir $L/.depth_late2_q9_running 2>/dev/null' EXIT
rm -f $L/DEPTH_LATE2_${NG}_FAILED
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
touch $L/DEPTH_LATE2_${NG}_TESTGEN_DONE
# 2 GSM8K, first 304 frozen items, cap 1024 (the cap of this model in the main table)
mkdir -p Iteration_01/capability/depthscan_gsm8k/$NG/b8
st gsm_304 docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$CARD -w $W truth_dec \
  $PYM -I $W/Iteration_01/code/scripts/newsets/ns_positions.py prompt gen --models $W/$CFG --model $NG --task gsm8k \
  --items $W/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet --ids $W/Iteration_01/capability/sets/gsm8k_first304_20261124.json \
  --max-new-tokens 1024 --batch-size 8 --out $W/Iteration_01/capability/depthscan_gsm8k/$NG/b8 --arms $GARMS || fail "GSM8K first 304 items"
flock -u 9; exec 9>&-
say "card $CARD released"
touch $L/DEPTH_LATE2_${NG}_GEN_DONE
say "card part done"

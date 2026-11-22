#!/bin/bash
# 2026-11-22 23:4x, exploration, queue I section 7.3: capability with the honest sentence appended to the system prompt
# (scripts/capability/cap_h22.py; programs unchanged), one model, one card (its lock).
#   MMLU (14,042 items; the table reads the 13,042 not in capability/sets/mmlu_1000_20261119.json): x (first 2 batches,
#        the check against the model's existing full-set file), then x (cap_mcq_sweep.py wants it first), x_E2, x_E2:E<L>.w.p<s_sel> (an MMLU read uses the last
#        prompt position only, so the content rule equals "prompt"); out capability/full_h22/<model>.json and
#        capability/full_h22/<model>_check.json.
#   GSM8K (1,319 items, cap 512, batch 8): neutral on the first 2 batches (check against the existing neutral.json),
#        neutral_E2 (no write), neutral_E2:E<L>.w.p<s_sel> at "prompt"; out capability/gen_main/gsm8k_h22/<model>/b8.
# Markers CAPH22_<MODEL>_DONE / _FAILED.  usage: cap_h22_model.sh KEY GPU
# cap_h22_part.sh (2026-11-23 00:1x) = the same with a third argument mmlu|gsm that runs one half only (markers
# CAPH22_<MODEL>_<PART>_DONE / _FAILED): on GPU 1 the MMLU run of Llama-3.1-8B reached 70 GB (over the 65 GB limit) and
# Qwen3.5-9B's ran out of memory, so the two small models run GSM8K on GPU 1 and MMLU on a large card.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2; PART=$3
[ "$PART" = mmlu ] || [ "$PART" = gsm ] || { echo "usage: cap_h22_part.sh KEY GPU mmlu|gsm"; exit 2; }
TF=$ROOT/envs/tf519/bin/python
case $KEY in
  llama) MODEL=llama3.1-8b; PYM=python3; NEED=20000 ;;
  q9)    MODEL=qwen3.5-9b; PYM=$TF; NEED=22000 ;;
  q27)   MODEL=qwen3.8-27b; PYM=$TF; NEED=62000 ;;
  g12)   MODEL=gemma3-12b; PYM=python3; NEED=30000 ;;
  g27)   MODEL=gemma3-27b; PYM=python3; NEED=62000 ;;
  *) echo "unknown key $KEY"; exit 2 ;;
esac
MK=$L/CAPH22_${MODEL}_$PART; log=$L/caph22_${MODEL}_$PART.log
say() { echo "$(date '+%F %T') caph22 $MODEL: $*" >> $L/n1s_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_DONE
CH=$ROOT/Iteration_01/code/scripts/capability/cap_h22.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
CAP=$ROOT/Iteration_01/capability; HCAP=$HOST/Iteration_01/capability
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)
read -r EL S SH CL <<< "$("${RD[@]}" Iteration_01/diag/n1_on_sentence.py plan $MODEL 2>> $log)"
[ -n "$CL" ] || fail "plan"
SS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["strength_dot"])' $HOST/Iteration_01/newsets/h22/n1s_choice_$MODEL.json) || fail "no s_sel"
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$GPU.lock
exec 9>"$LOCK" || fail "cannot open $LOCK"
until flock -n 9; do sleep 60; done
while true; do
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1 == g {print $2}')
  [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $GPU"
  if [ $GPU = 1 ]; then LIM=66560; else LIM=73000; fi
  [ $((USED + NEED)) -le $LIM ] && break
  say "GPU $GPU holds $USED MiB, $NEED more would exceed $LIM; waiting"; sleep 60
done
say "start on GPU $GPU ($USED MiB in use): E$EL at s_sel $SS"
DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
if [ $PART = mmlu ]; then
MI=(--models $M --model $MODEL --items $ROOT/data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids $CAP/sets/mmlu_14042_20261120.json)
mkdir -p $HCAP/full_h22
[ -s $HCAP/full_h22/${MODEL}_check.json ] || "${DQ[@]}" $PYM -I $CH mmlu run "${MI[@]}" --out $CAP/full_h22/${MODEL}_check.json --arms x --first-batches 2 >> $log 2>&1 || fail "MMLU check run"
"${RD[@]}" Iteration_01/diag/sentence_capability.py mmlucheck $MODEL >> $log 2>&1 || fail "MMLU check: x differs from the existing file"
[ -s $HCAP/full_h22/$MODEL.json ] || "${DQ[@]}" $PYM -I $CH mmlu run "${MI[@]}" --out $CAP/full_h22/$MODEL.json --arms x x_E2 x_E2:E$EL.w.p$SS >> $log 2>&1 || fail "MMLU run"
say "MMLU done"
else
GI=(--models $M --model $MODEL --task gsm8k --items $ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet
    --ids $CAP/sets/gsm8k_1319_20261119.json --max-new-tokens 512 --batch-size 8)
GO=$CAP/gen_main/gsm8k_h22/$MODEL/b8
"${DQ[@]}" $PYM -I $CH gsm none gen "${GI[@]}" --out $GO --arms neutral --first-batches 2 >> $log 2>&1 || fail "GSM8K check run"
"${RD[@]}" Iteration_01/diag/sentence_capability.py gsmcheck $MODEL >> $log 2>&1 || fail "GSM8K check: neutral differs from the existing file"
"${DQ[@]}" $PYM -I $CH gsm none gen "${GI[@]}" --out $GO --arms neutral_E2 >> $log 2>&1 || fail "GSM8K neutral_E2"
"${DQ[@]}" $PYM -I $CH gsm prompt gen "${GI[@]}" --out $GO --arms neutral_E2:E$EL.w.p$SS >> $log 2>&1 || fail "GSM8K neutral_E2 + E"
fi
say "$PART done; card released"
touch ${MK}_DONE

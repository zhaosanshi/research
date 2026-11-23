#!/bin/bash
# 2026-11-23 05:2x, exploration, queue I section 9: the capability cost at E(s_half), s_half = s_sel / 2, with the honest
# sentence in the system prompt (scripts/capability/cap_h22.py; programs unchanged), everything else as 7.3.
#   mmlu: x on the first 2 batches (check against the existing full-set file), then x (cap_mcq_sweep.py wants it first) and
#         x_E2:E<L>.w.p<s_half>; out capability/full_h22s/<model>.json (and _check.json).
#   gsm:  neutral on the first 2 batches (check), neutral_E2:E<L>.w.p<s_half> at "prompt" (capability/gen_main/gsm8k_h22/
#         <model>/b8); for Qwen3.5-9B also the no-sentence reference neutral:E15.w.p2 at "prompt" (same dir).
# Write records: E layer and direction hashes = the main table's, amount = main amount x s_half / main strength.
# Markers CAP9_<MODEL>_<PART>_DONE / _FAILED.  usage: cap9_model.sh KEY GPU mmlu|gsm
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2; PART=$3
[ "$PART" = mmlu ] || [ "$PART" = gsm ] || { echo "usage: cap9_model.sh KEY GPU mmlu|gsm"; exit 2; }
TF=$ROOT/envs/tf519/bin/python
case $KEY in
  llama) MODEL=llama3.1-8b; PYM=python3; NEED=20000 ;;
  q9)    MODEL=qwen3.5-9b; PYM=$TF; NEED=22000 ;;
  q27)   MODEL=qwen3.8-27b; PYM=$TF; NEED=62000 ;;
  g12)   MODEL=gemma3-12b; PYM=python3; NEED=30000 ;;
  g27)   MODEL=gemma3-27b; PYM=python3; NEED=62000 ;;
  *) echo "unknown key $KEY"; exit 2 ;;
esac
MK=$L/CAP9_${MODEL}_$PART; log=$L/cap9_${MODEL}_$PART.log
say() { echo "$(date '+%F %T') cap9 $MODEL: $*" >> $L/n1s_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_DONE
CH=$ROOT/Iteration_01/code/scripts/capability/cap_h22.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
CAP=$ROOT/Iteration_01/capability; HCAP=$HOST/Iteration_01/capability
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)
read -r EL S SH CL <<< "$("${RD[@]}" Iteration_01/diag/n1_on_sentence.py plan $MODEL 2>> $log)"
[ -n "$CL" ] || fail "plan"
SS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["strength_dot"])' $HOST/Iteration_01/newsets/h22/n1s_choice_$MODEL.json) || fail "no s_sel"
SH=$(python3 -c "import sys; h=float(sys.argv[1])/2; print(('%.4f' % h).rstrip('0').rstrip('.'))" $SS)
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
say "start on GPU $GPU ($USED MiB in use): E$EL at s_half $SH"
DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
if [ $PART = mmlu ]; then
MI=(--models $M --model $MODEL --items $ROOT/data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids $CAP/sets/mmlu_14042_20261120.json)
mkdir -p $HCAP/full_h22s
[ -s $HCAP/full_h22s/${MODEL}_check.json ] || "${DQ[@]}" $PYM -I $CH mmlu run "${MI[@]}" --out $CAP/full_h22s/${MODEL}_check.json --arms x --first-batches 2 >> $log 2>&1 || fail "MMLU check run"
"${RD[@]}" Iteration_01/diag/sentence_capability.py mmlucheck $MODEL full_h22s >> $log 2>&1 || fail "MMLU check: x differs from the existing file"
[ -s $HCAP/full_h22s/$MODEL.json ] || "${DQ[@]}" $PYM -I $CH mmlu run "${MI[@]}" --out $CAP/full_h22s/$MODEL.json --arms x x_E2:E$EL.w.p$SH >> $log 2>&1 || fail "MMLU run"
"${RD[@]}" Iteration_01/diag/sentence_capability.py mmluparts $MODEL $SH >> $log 2>&1 || fail "MMLU write record"
else
GI=(--models $M --model $MODEL --task gsm8k --items $ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet
    --ids $CAP/sets/gsm8k_1319_20261119.json --max-new-tokens 512 --batch-size 8)
GO=$CAP/gen_main/gsm8k_h22/$MODEL/b8; GM=$CAP/gen_main/gsm8k/$MODEL/b8
NC=$ROOT/Iteration_01/code/scripts/newsets/n1s_check.py
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
"${DQ[@]}" $PYM -I $CH gsm none gen "${GI[@]}" --out $GO --arms neutral --first-batches 2 >> $log 2>&1 || fail "GSM8K check run"
"${RD[@]}" Iteration_01/diag/sentence_capability.py gsmcheck $MODEL >> $log 2>&1 || fail "GSM8K check: neutral differs from the existing file"
DS=$(echo $S | tr . -); DH=$(echo $SH | tr . -)
if [ $MODEL = qwen3.5-9b ]; then
  "${DQ[@]}" $PYM -I $CH gsm prompt gen "${GI[@]}" --out $GO --arms neutral:E$EL.w.p$S >> $log 2>&1 || fail "GSM8K no-sentence E"
  REF=$GO/neutral--E$EL-w-p$DS-at-prompt.json
else
  REF=$GM/neutral--E$EL-w-p$DS-at-prompt.json
fi
"${DQ[@]}" $PYM -I $CH gsm prompt gen "${GI[@]}" --out $GO --arms neutral_E2:E$EL.w.p$SH >> $log 2>&1 || fail "GSM8K neutral_E2 + E(s_half)"
"${DC[@]}" $NC parts $GO/neutral_E2--E$EL-w-p$DH-at-prompt.json $REF $SH $S E >> $log 2>&1 || fail "GSM8K write record"
fi
say "$PART done; card released"
touch ${MK}_DONE

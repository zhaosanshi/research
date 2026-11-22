#!/bin/bash
# 2026-11-22 10:3x (Opus 5.5), exploration: ITI again with the faithful fit (iti_baseline.py fit --faithful: the
# official probe text, i.e. answers with a full stop; the random question of the sigma bank drawn once per question;
# numpy head order) for Llama-3.1-8B and Gemma-3-12B. Why: the read-only review of 2026-11-22 10 h found that the
# official probes read the full stop at the end of the multiple-choice targets, ours the answer's last word; refitting
# with the full stop changed most chosen heads (Llama: 18 of the top 48 heads shared; Gemma-3-12B: 24 of 48). The
# earlier fits stay as a variant ("probe at the answer's last word").
# Grid, rules and readings are those of the first run (discussion/work_queue/2026-11-22_queue.md section 4): K 16 32 48
# 64, alpha 2.5 .. 30, MMLU budget 1 point, the choice on the tune split, the test split read once; then the reading
# without the budget and GSM8K at both choices (iti_chain.sh, iti_free.sh).
# The grid lanes use the staged copy of the new program (ITI_PROGRAM) while a run of the old version is still
# finishing; the staged copy and the copy installed afterwards have the same content.
# usage: iti_faithful_1122.sh          Markers: those of iti_grid.sh / iti_chain.sh / iti_free.sh, fit name faithful_seed42.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ST=$ROOT/Iteration_01/code/scripts/newsets_stage/iti_baseline.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
A="2.5 5 7.5 10 15 20 25 30"
F=faithful_seed42
say() { echo "$(date '+%F %T') iti faithful: $*" >> $L/iti_all.log; }
DX() { local g=$1; shift; docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$g -w $ROOT truth_dec "$@"; }
# Gemma-3-12B: fit and acceptance check on card 0 (the Llama fit was made with the same staged program at 10:2x)
G=Iteration_01/baselines/iti/gemma3-12b
if [ ! -e $G/fit_$F.pt ]; then
  DX 0 python3 -I $ST fit --models $M --model gemma3-12b --faithful --out $ROOT/$G/fit_$F.pt > $L/iti_fit_gemma3-12b_$F.log 2>&1 || { say "FAILED: Gemma fit"; exit 1; }
fi
DX 0 python3 -I $ST check --models $M --model gemma3-12b --fit $ROOT/$G/fit_$F.pt --k 48 --alpha 5 > $L/iti_check_gemma3-12b_${F}_20261122.txt 2>&1
grep -q "^check PASS" $L/iti_check_gemma3-12b_${F}_20261122.txt || { say "FAILED: Gemma acceptance check"; exit 1; }
B=Iteration_01/baselines/iti/llama3.1-8b
DX 3 python3 -I $ST check --models $M --model llama3.1-8b --fit $ROOT/$B/fit_$F.pt --k 48 --alpha 5 > $L/iti_check_llama3.1-8b_${F}_20261122.txt 2>&1
grep -q "^check PASS" $L/iti_check_llama3.1-8b_${F}_20261122.txt || { say "FAILED: Llama acceptance check"; exit 1; }
say "fits and acceptance checks done; grid lanes start"
lane() { ITI_PROGRAM=$ST setsid nohup bash $L/iti_grid.sh "$@" > /dev/null 2>&1 < /dev/null & }
lane gemma3-12b 0 $F endq "16" "$A" f16
lane gemma3-12b 0 $F endq "32" "$A" f32
lane gemma3-12b 2 $F endq "48" "$A" f48
lane gemma3-12b 2 $F endq "64" "$A" f64
lane llama3.1-8b 3 $F endq "16" "$A" f16
lane llama3.1-8b 3 $F endq "32" "$A" f32
# the main and the second reading wait for their lanes (and for each other) by markers
setsid nohup bash $L/iti_chain.sh gemma3-12b $F endq 0 "16 32 48 64" "$A" pospromptl24 f16 f32 f48 f64 > /dev/null 2>&1 < /dev/null &
setsid nohup bash $L/iti_free.sh gemma3-12b $F endq 0 "16 32 48 64" "$A" pospromptl24 0 2 > /dev/null 2>&1 < /dev/null &
setsid nohup bash $L/iti_chain.sh llama3.1-8b $F endq 3 "16 32 48 64" "$A" pospromptl16 f16 f32 f48 f64 > /dev/null 2>&1 < /dev/null &
setsid nohup bash $L/iti_free.sh llama3.1-8b $F endq 3 "16 32 48 64" "$A" pospromptl16 3 1 > /dev/null 2>&1 < /dev/null &
# the two other Llama lanes start when their cards are free: K 48 on card 3 after the paper-sigma check of the first
# fit has ended, K 64 on card 1 after e_hi_llama.sh
until [ -e $L/ITI_CHAIN_llama3.1-8b_train_seed42_pairs_DONE ] || [ -e $L/ITI_CHAIN_llama3.1-8b_train_seed42_pairs_FAILED ]; do sleep 60; done
# no run of the old program is left: install the new one where the chains and second readings call it
CAN=Iteration_01/code/scripts/newsets/iti_baseline.py
cp -p Iteration_01/code/scripts/newsets_stage/iti_baseline.py $CAN.new && mv $CAN.new $CAN
[ "$(sha256sum < $CAN)" = "$(sha256sum < Iteration_01/code/scripts/newsets_stage/iti_baseline.py)" ] || { say "FAILED: install of the new program"; exit 1; }
say "new iti_baseline.py installed ($(sha256sum < $CAN | cut -c1-16))"
lane llama3.1-8b 3 $F endq "48" "$A" f48
until [ -e $L/E_HI_LLAMA_DONE ] || [ -e $L/E_HI_LLAMA_FAILED ]; do sleep 60; done
lane llama3.1-8b 1 $F endq "64" "$A" f64
say "all lanes started"

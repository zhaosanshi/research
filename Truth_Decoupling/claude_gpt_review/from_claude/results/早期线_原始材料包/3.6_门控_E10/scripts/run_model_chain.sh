#!/usr/bin/env bash
# Model-agnostic chain for the teacher's steps on one model and one or more regimes (step 8: other model families).
# Per regime: B0 baseline -> E1 directions (mean-diff) -> E13 clean gap on ALL layers -> pick layers from E13
#             -> E11 dose/rates/controls -> E10 gating -> E6 transfer (+ oracle positive control)
# Layers are chosen by the SAME rule for every model: E write layers = best clean mean-diff layer L* and its two
# neighbours (L*-1, L*, L*+1); probe layer = best clean probe layer; transfer write layer = L*+3 (Llama: 13 -> 12,13,14 / 14 / 16).
# usage: run_model_chain.sh <gpu> <model> <regime> [<regime> ...]     regimes: tqa | halu | knowledge
#   nohup bash .../run_model_chain.sh 0 qwen3-8b tqa knowledge halu > logs/chain_qwen3-8b.log 2>&1 &
set -u
GPU="$1"; MODEL="$2"; shift 2
BASE=/data/sanshizhao/Truth_Decoupling
cd "$BASE"
mkdir -p logs
for REG in "$@"; do
  case "$REG" in
    tqa)       ROOT=results_tqa;  EXP=E2b_tqa;       E1RUN=E1_directions_tqa;  E1EST=E1_directions_tqa;        DS=truthfulqa_tf ;;
    halu)      ROOT=results_halu; EXP=E2b_halu;      E1RUN=E1_directions_halu; E1EST=E1_directions_halu;       DS=halueval_tf ;;
    knowledge) ROOT=results_v2;   EXP=E2b_h1_rescue; E1RUN=E1_directions;      E1EST=E1_directions_knowledge3; DS="fever counterfact got" ;;
    bbq)       ROOT=results_bbq;  EXP=E2b_bbq;       E1RUN=E1_directions_bbq;  E1EST=E1_directions_bbq;        DS=bbq_tf ;;
    *) echo "unknown regime $REG"; continue ;;
  esac
  mkdir -p "$ROOT"
  DEX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e CUDA_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 \
    -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/$ROOT -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed \
    -w /workspace/Truth_Decoupling/code/experiments truth_dec"
  run() { local name="$1"; shift; echo "[$(date +%H:%M:%S)] $REG/$name start"; $DEX "$@" > "logs/${REG}_${name}_$MODEL.log" 2>&1; echo "[$(date +%H:%M:%S)] $REG/$name rc=$?"; }
  LIM=""; [ "$REG" = knowledge ] && LIM="--limits fever=400 counterfact=400 got=400"
  run b0  python3 scripts/baseline_readout.py --model "$MODEL" --datasets $DS --split test $LIM
  run e1  python3 scripts/e1_directions.py --exp "$E1RUN" --model "$MODEL"
  run e13 python3 scripts/e13_clean_gap.py --exp "$EXP" --e1-exp "$E1EST" --model "$MODEL"
  if ! $DEX test -f "/workspace/Truth_Decoupling/$ROOT/E13/$MODEL/clean_gap.json"; then
    echo "[$(date +%H:%M:%S)] $REG FAILED for $MODEL: E13 clean_gap.json missing (see logs/${REG}_b0/e1/e13_$MODEL.log, likely OOM) -- skipping the rest of this regime"
    continue
  fi
  PICK=$($DEX python3 -c "
import json; j=json.load(open('/workspace/Truth_Decoupling/$ROOT/E13/$MODEL/clean_gap.json'))
L=int(j['best_meandiff_layer']); P=int(j['best_probe_layer']); n=max(int(k) for k in j['per_layer'])+2
print(L-1, L, L+1, P, min(L+3, n-2))")
  read EL1 EL2 EL3 PL WL <<< "$PICK"
  echo "[$(date +%H:%M:%S)] $REG layer pick for $MODEL: E layers $EL1 $EL2 $EL3 | probe $PL | transfer write $WL (up_room md $( $DEX python3 -c "import json; j=json.load(open('/workspace/Truth_Decoupling/$ROOT/E13/$MODEL/clean_gap.json')); print('%+.3f probe %+.3f down %+.3f' % (j['up_room_meandiff'], j['up_room_probe'], j['down_room']))" ))"
  run e11 python3 scripts/e11_dose_rates.py --exp "$EXP" --e1-exp "$E1EST" --model "$MODEL" --e-layers $EL1 $EL2 $EL3 --probe-layer $PL
  run e10 python3 scripts/e10_gating.py --exp "$EXP" --e1-exp "$E1EST" --model "$MODEL" --e-layers $EL1 $EL2 $EL3 --probe-layer $PL --write-layer $WL
  run e6  python3 scripts/e6_transfer.py --exp "$EXP" --e1-exp "$E1EST" --model "$MODEL" --read-layers $PL $EL2 --write-layers $WL $EL3
  echo "[$(date +%H:%M:%S)] $REG DONE $MODEL"
done
echo "[$(date +%H:%M:%S)] MODEL CHAIN DONE $MODEL ($*)"

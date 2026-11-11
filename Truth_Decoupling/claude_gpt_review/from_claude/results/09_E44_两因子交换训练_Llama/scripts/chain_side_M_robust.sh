#!/usr/bin/env bash
# Side branch M, candidate layers 11 and 13 (both met column A under both judges): two more optimisation seeds and two more
# random axes on the tuning questions, then both judges. Reviewed driver and judge runner only. Runs on the host.
D=/data/sanshizhao/Truth_Decoupling/review_e44_claude
W=/workspace/Truth_Decoupling/review_e44_claude
L=$D/logs_side_M_robust; mkdir -p $L
SIDEC=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_M
DX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -w $W"
run() { $DX -e MODE=$1 -e SEED=$2 -e LAYERS="11 13" -e GPU=$3 truth_dec bash $W/e44_side_M_robust.sh > $L/$1$2.log 2>&1 || touch $L/FAILED_$1$2; }
run seed 1051 0 &
run seed 2045 1 &
run axis 1051 2 &
run axis 2045 3 &
wait
ls $L | grep -q "^FAILED" && { touch $L/ROBUST_FAILED; exit 1; }
touch $L/ROBUST_GENERATED
gpu=0
for mode in seed1051 seed2045 axis1051 axis2045; do
  RUNC=${SIDEC}_$mode
  for layer in 11 13; do
    case $mode in
      axis*) ARMS="x.w.p1 x.rand.p1 x.rand.m1"; PROMPTS="x" ;;
      *)     ARMS="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1"; PROMPTS="x x_F" ;;
    esac
    # identical answers keep identical Pro labels: start from the side branch's own cache of this layer
    $DX truth_dec bash -c "mkdir -p $RUNC/judge_tune_proA_E_L$layer/judge2/cache && cp -n $SIDEC/judge_tune_proA_E_L$layer/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl $RUNC/judge_tune_proA_E_L$layer/judge2/cache/"
    $DX -e RUN=$RUNC -e FAMILY=E -e LAYER=$layer -e GPU=$((gpu % 4)) -e TAG=A -e ARMS="$ARMS" -e PROMPTS="$PROMPTS" truth_dec bash $W/e44_round1_pro.sh > $L/pro_${mode}_L$layer.log 2>&1 || touch $L/FAILED_pro_${mode}_L$layer &
    gpu=$((gpu + 1))
  done
done
wait
ls $L | grep -q "^FAILED" && touch $L/ROBUST_FAILED
touch $L/ROBUST_JUDGED

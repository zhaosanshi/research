#!/usr/bin/env bash
# Same plan as chain_side_M_robust.sh, on cards 2 and 3 only: cards 0 and 1 carry another service (vLLM-Omni diffusion
# workers, started 11-11 about 15:14 Beijing) that this chain must not touch. The two random axes were already generated
# (cards 2 / 3, 15:19-15:20); the two seed runs were stopped at 15:25 and start again here from scratch.
D=/data/sanshizhao/Truth_Decoupling/review_e44_claude
W=/workspace/Truth_Decoupling/review_e44_claude
L=$D/logs_side_M_robust; mkdir -p $L
SIDEC=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_M
DX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -w $W"
run() { $DX -e MODE=$1 -e SEED=$2 -e LAYERS="11 13" -e GPU=$3 truth_dec bash $W/e44_side_M_robust.sh > $L/$1$2.log 2>&1 || touch $L/FAILED_$1$2; }
run seed 1051 2 &
run seed 2045 3 &
wait
ls $L | grep -q "^FAILED" && { touch $L/ROBUST_FAILED; exit 1; }
touch $L/ROBUST_GENERATED
judge() {   # mode layer card
  local RUNC=${SIDEC}_$1 ARMS PROMPTS
  case $1 in
    axis*) ARMS="x.w.p1 x.rand.p1 x.rand.m1"; PROMPTS="x" ;;
    *)     ARMS="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1"; PROMPTS="x x_F" ;;
  esac
  # identical answers keep identical Pro labels: start from the side branch's own cache of this layer
  $DX truth_dec bash -c "mkdir -p $RUNC/judge_tune_proA_E_L$2/judge2/cache && cp -n $SIDEC/judge_tune_proA_E_L$2/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl $RUNC/judge_tune_proA_E_L$2/judge2/cache/"
  $DX -e RUN=$RUNC -e FAMILY=E -e LAYER=$2 -e GPU=$3 -e TAG=A -e ARMS="$ARMS" -e PROMPTS="$PROMPTS" truth_dec bash $W/e44_round1_pro.sh > $L/pro_$1_L$2.log 2>&1 || touch $L/FAILED_pro_$1_L$2
}
judge seed1051 11 2 & judge seed1051 13 3 & judge seed2045 11 2 & judge seed2045 13 3 & wait     # at most two judges per card
judge axis1051 11 2 & judge axis1051 13 3 & judge axis2045 11 2 & judge axis2045 13 3 & wait
ls $L | grep -q "^FAILED" && touch $L/ROBUST_FAILED
touch $L/ROBUST_JUDGED

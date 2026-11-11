#!/usr/bin/env bash
# Main must-answer E (layer 10, strength 2): two more optimisation seeds and two more random axes on the tuning questions,
# then both judges. Reviewed driver and judge runner only; all four cards (the other service on cards 0 / 1 was stopped
# 11-11 16:35 at the project owner's request). Runs on the host.
D=/data/sanshizhao/Truth_Decoupling/review_e44_claude
W=/workspace/Truth_Decoupling/review_e44_claude
L=$D/logs_main_E_robust; mkdir -p $L
MAINC=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b
DX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -w $W"
run() { $DX -e MODE=$1 -e SEED=$2 -e GPU=$3 truth_dec bash $W/e44_main_E_robust.sh > $L/$1$2.log 2>&1 || touch $L/FAILED_$1$2; }
run seed 1051 0 &
run seed 2045 1 &
run axis 1051 2 &
run axis 2045 3 &
wait
ls $L | grep -q "^FAILED" && { touch $L/ROBUST_FAILED; exit 1; }
touch $L/ROBUST_GENERATED
judge() {   # mode card
  local RUNC=${MAINC}_mainE_$1 ARMS PROMPTS
  case $1 in
    axis*) ARMS="conf.w.p2 conf.rand.p2 conf.rand.m2"; PROMPTS="conf" ;;
    *)     ARMS="conf.w.p1 conf.w.p2 conf.md.p2 conf.rand.p2 conf.rand.m2 conf_E1.w.m2"; PROMPTS="conf conf_E1" ;;
  esac
  # identical answers keep identical Pro labels: start from the main run's cache of the same layer
  $DX truth_dec bash -c "mkdir -p $RUNC/judge_tune_proA_E_L10/judge2/cache && cp -n $MAINC/judge_tune_proB_E_L10/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl $RUNC/judge_tune_proA_E_L10/judge2/cache/"
  $DX -e RUN=$RUNC -e FAMILY=E -e LAYER=10 -e GPU=$2 -e TAG=A -e ARMS="$ARMS" -e PROMPTS="$PROMPTS" truth_dec bash $W/e44_round1_pro.sh > $L/pro_$1.log 2>&1 || touch $L/FAILED_pro_$1
}
judge seed1051 0 & judge seed2045 1 & judge axis1051 2 & judge axis2045 3 & wait
ls $L | grep -q "^FAILED" && touch $L/ROBUST_FAILED
touch $L/ROBUST_JUDGED

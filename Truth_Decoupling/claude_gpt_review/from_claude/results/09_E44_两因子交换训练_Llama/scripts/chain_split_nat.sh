#!/usr/bin/env bash
# Host-side chain for 33 section 1.7 B (split of the learned write, card 2) and D (read-only natural differences, card 3),
# exactly as written there, with GPT round 34's design notes (four cells incl. no write and full w; natural = description).
D=/data/sanshizhao/Truth_Decoupling/review_e44_claude
W=/workspace/Truth_Decoupling/review_e44_claude
L=$D/logs_extra_EC; mkdir -p $L $D/tables_EC
M=/data/sanshizhao/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b
DX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -w $W"
( $DX -e STAGE=nat -e GPU=3 truth_dec bash $W/e44_extra_EC.sh > $L/nat.log 2>&1 && touch $L/NAT_DONE || touch $L/NAT_FAILED ) &
( $DX -e STAGE=genS -e GPU=2 truth_dec bash $W/e44_extra_EC.sh > $L/genS.log 2>&1 &&
  $DX -e STAGE=judgeS -e GPU=2 truth_dec bash $W/e44_extra_EC.sh > $L/judgeS.log 2>&1 &&
  python3 $D/e44_EC_describe.py $M/eval_extra/judge_tune_split_tune --judge allenai > $D/tables_EC/split_describe_allenai.md &&
  python3 $D/e44_EC_describe.py $M/eval_extra/judge_tune_split_tune --judge pro > $D/tables_EC/split_describe_pro.md &&
  touch $L/SPLIT_JUDGED || touch $L/SPLIT_FAILED ) &
wait

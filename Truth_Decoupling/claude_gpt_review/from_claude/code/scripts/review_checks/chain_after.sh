#!/usr/bin/env bash
# Host side: chain_after.sh MODEL GPU -> wait until the currently running first launcher of MODEL has exited
# (its log receives a "launcher exit" line), then run the rest of the chain with full_v1_v2.sh (phases already done are skipped).
set -u
MODEL=$1; GPU=$2
G16=/data/sanshizhao/Truth_Decoupling/review_e25/checks/gpt16
HOSTLOG=/data/sanshizhao/Truth_Decoupling/results_qa_das_v1/$MODEL/full_v1.log
until grep -q "launcher exit" "$HOSTLOG"; do sleep 30; done
echo "== $(date '+%F %T') first launcher finished; continuing the chain to the end for $MODEL (owner decision 2026-11-05: run everything, look at results)" >> "$HOSTLOG"
bash "$G16/full_v1_v2.sh" "$MODEL" "$GPU" judge-diagnostics

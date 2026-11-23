#!/bin/bash
# 2026-11-23, queue L section 9 (executor A): fillctl_gen.sh for several models and phases in order on one card; stops at
# the first failure (marker FILLCTL_CHAIN_<TAG>_FAILED), else FILLCTL_CHAIN_<TAG>_DONE.
# usage (host): fillctl_chain.sh GPU TAG "PHASES" "KEYS"      e.g. fillctl_chain.sh 1 pilots "pilot" "q9 q27 g12 g27"
cd /data/sanshizhao/Truth_Decoupling || exit 2
L=Iteration_01/logs/newsets_20261119
GPU=$1; TAG=$2; PHS=$3; KEYS=$4
rm -f $L/FILLCTL_CHAIN_${TAG}_DONE $L/FILLCTL_CHAIN_${TAG}_FAILED
for k in $KEYS; do
  for ph in $PHS; do
    bash $L/fillctl_gen.sh $k $GPU $ph || { touch $L/FILLCTL_CHAIN_${TAG}_FAILED; exit 1; }
  done
done
touch $L/FILLCTL_CHAIN_${TAG}_DONE

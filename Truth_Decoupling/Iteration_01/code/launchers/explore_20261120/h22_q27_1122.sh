#!/bin/bash
# 2026-11-22 20:1x, queue H: Qwen3.8-27B (h22_model.sh) after Gemma-3-12B has left GPU 3. On GPU 3 only if it can start
# before 21:30; otherwise after the marker ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE on GPU 2 (lane D of
# h22_1122.sh, which this replaces; that lane was not started).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
ITI=$L/ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE
until [ -e $L/H22_gemma3-12b_GPU_DONE ] || [ -e $L/H22_gemma3-12b_FAILED ]; do sleep 30; done
if [ $((10#$(TZ=Asia/Shanghai date +%H%M))) -lt 2130 ]; then
  bash $L/h22_model.sh q27 3 > $L/h22_q27.out 2>&1
else
  until [ -e $ITI ]; do sleep 60; done
  bash $L/h22_model.sh q27 2 > $L/h22_q27.out 2>&1
fi

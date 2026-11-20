#!/bin/bash
# 2026-11-21 02:4x: starts l16_mdg.sh on GPU 3 once the GPU part of l16_strong.sh there has finished (its confidence
# step logs "l16strong conf llama"); both are Llama processes of about 18 GB.
cd /data/sanshizhao/Truth_Decoupling/Iteration_01/logs/newsets_20261119
(setsid nohup bash l16_mdg.sh 3 'grep -q "l16strong conf llama" $L/l16_all.log' > l16_mdg.out 2>&1 < /dev/null &)

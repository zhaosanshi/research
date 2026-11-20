#!/bin/bash
# 2026-11-21 02:1x: starts pressure2.sh for the three smaller models, both sentence variants.
#   Llama        GPU 0 now (beside the two lanes of oncaut2_all.sh, 18 + 20 GB at most);
#   Gemma-3-12B  GPU 0 after Llama's pressure2 has finished;
#   Qwen3.5-9B   GPU 1 after its second-batch confirmation there has finished (the last step of its chain on GPU 1).
cd /data/sanshizhao/Truth_Decoupling/Iteration_01/logs/newsets_20261119
(setsid nohup bash pressure2.sh llama3.1-8b 0 "deny suggest" > pressure2_llama.out 2>&1 < /dev/null &)
(setsid nohup bash pressure2.sh gemma3-12b 0 "deny suggest" '[ -e $L/PRESSURE2_llama3.1-8b_DONE ]' > pressure2_g12.out 2>&1 < /dev/null &)
(setsid nohup bash pressure2.sh qwen3.5-9b 1 "deny suggest" '[ -e $L/NS4_qwen3.5-9b_DONE ]' > pressure2_q9.out 2>&1 < /dev/null &)

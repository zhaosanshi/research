#!/bin/bash
# 2026-11-21 01:3x: starts the three ns4_test2.sh waiters (second-batch confirmation passes of the cautious-C
# combination) for the three smaller models, each on the card where its own earlier lane has finished by then.
#   Llama        GPU 0: after its low-strength MMLU there is done and its tune / test passes exist;
#   Gemma-3-12B  GPU 3: after its last new-dataset step there (NS3 marker);
#   Qwen3.5-9B   GPU 1: after its last new-dataset step there AND after the rerun of the mean-difference pass that
#                       waits for the same marker (two Qwen3.5-9B processes do not fit beside the other user's job).
cd /data/sanshizhao/Truth_Decoupling/Iteration_01/logs/newsets_20261119
(setsid nohup bash ns4_test2.sh llama3.1-8b 0 '[ -e $L/LOW_GPU_llama_DONE ] && [ -e $L/NS2_llama3.1-8b_DONE ]' > ns4_llama.out 2>&1 < /dev/null &)
(setsid nohup bash ns4_test2.sh gemma3-12b 3 '[ -e $L/NS3_gemma3-12b_DONE ]' > ns4_g12.out 2>&1 < /dev/null &)
(setsid nohup bash ns4_test2.sh qwen3.5-9b 1 '[ -e $L/NS3_qwen3.5-9b_DONE ] && grep -q "mdsweep tq q9 mdbest_test exit=0\|mdsweep q9 rerun finished" $L/mdsweep_all.log' > ns4_q9.out 2>&1 < /dev/null &)

#!/usr/bin/env bash
# big_chain.sh <gpu(s)> <model> <download-dir-name>: wait for the ModelScope download, run the four-regime chain, then E14 MC (TQA + BBQ)
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
GPU="$1"; M="$2"; DL="$3"
until [ -f logs/download-$DL.done ]; do sleep 60; done
stamp "chain $M start on gpu $GPU"
bash code/experiments/scripts/run_model_chain.sh $GPU $M tqa bbq halu knowledge > logs/chain_$M.log 2>&1; stamp "chain $M rc=$?"
if [ -f results_tqa/E13/$M/clean_gap.json ]; then
  read EL1 EL2 EL3 PL WL <<< "$(bash logs/pick_layers.sh results_tqa $M)"
  bash code/experiments/scripts/run_e14_model.sh $GPU $M $EL1 $EL2 $EL3 $PL $WL > logs/e14_$M.log 2>&1; stamp "e14 tqa $M rc=$?"
fi
if [ -f results_bbq/E13/$M/clean_gap.json ]; then
  read EL1 EL2 EL3 PL WL <<< "$(bash logs/pick_layers.sh results_bbq $M)"
  bash code/experiments/scripts/run_e14_model.sh $GPU $M $EL1 $EL2 $EL3 $PL $WL bbq_mc bbq_tf E2b_bbq E1_directions_bbq results_bbq > logs/e14bbq_$M.log 2>&1; stamp "e14 bbq $M rc=$?"
fi
stamp "big $M DONE"
EOF
cat > q_big_gpu0.sh <<'"'"'EOF'"'"'
#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
bash logs/big_chain.sh 0 gemma3-4b gemma-3-4b-it
bash logs/das_xmodel.sh 0 qwen3-8b
bash logs/big_chain.sh 0 gemma3-12b gemma-3-12b-it
stamp "QUEUE BIG GPU0 DONE"; touch logs/q_big_gpu0.DONE
EOF
cat > q_big_gpu1.sh <<'"'"'EOF'"'"'
#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
bash logs/das_xmodel.sh 1 mistral-7b
bash logs/big_chain.sh 1 qwen3-14b Qwen3-14B
stamp "QUEUE BIG GPU1 DONE"; touch logs/q_big_gpu1.DONE
EOF
cat > q_big_gpu2.sh <<'"'"'EOF'"'"'
#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
bash logs/das_xmodel.sh 2 qwen3-4b
bash logs/big_chain.sh 2 gemma3-27b gemma-3-27b-it
stamp "QUEUE BIG GPU2 DONE"; touch logs/q_big_gpu2.DONE
EOF
cat > q_big_gpu3.sh <<'"'"'EOF'"'"'
#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
bash logs/das_xmodel.sh 3 qwen3-1.7b
bash logs/big_chain.sh 3 mistral-small-24b Mistral-Small-24B-Instruct-2501
stamp "QUEUE BIG GPU3 DONE"; touch logs/q_big_gpu3.DONE
EOF
cat > q_big_32b.sh <<'"'"'EOF'"'"'
#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
until [ -f logs/q_big_gpu2.DONE ] && [ -f logs/q_big_gpu3.DONE ]; do sleep 120; done
bash logs/big_chain.sh 2,3 qwen3-32b Qwen3-32B
stamp "QUEUE BIG 32B DONE"; touch logs/q_big_32b.DONE
EOF
chmod +x pick_layers.sh das_xmodel.sh big_chain.sh q_big_gpu*.sh q_big_32b.sh; bash -n pick_layers.sh && bash -n das_xmodel.sh && bash -n big_chain.sh && for f in q_big_gpu0.sh q_big_gpu1.sh q_big_gpu2.sh q_big_gpu3.sh q_big_32b.sh; do bash -n $f || echo "SYNTAX $f"; done; echo "scripts ok"; ls -la | grep -E "q_big|das_xmodel|big_chain|pick_layers"'
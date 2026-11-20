#!/bin/bash
# 2026-11-21 03:1x: Gemma-3-12B with the E write at LAYER 24 instead of 22 (alt_chain.sh). The MMLU layer map on the
# 1,000-item list (capability/layermap/gemma3-12b.json): at strength 2 layer 22 loses 2.5 points, layer 24 loses nothing;
# on the tune split (AllenAI, strength 1) single E scores +49 at layer 24 and +54 at layer 22.
# Generation on GPU 0 once the layer-map run there has ended; the MMLU run on GPU 3 once the MMLU run of
# oncaut_mm_g12.sh there has ended (two Gemma-3-12B MMLU runs do not fit on one card beside the other jobs).
cd /data/sanshizhao/Truth_Decoupling/Iteration_01/logs/newsets_20261119
until grep -q "layer map gemma3-12b exit=" ns2_all.log; do sleep 20; done
(setsid nohup bash alt_chain.sh g12 24 "1.5 2 2.5 3" l24 0 3 judge_tune_ec25_tune judge_test_t25_test '[ -e $L/ONCAUT_MM_g12_DONE ]' > alt_chain_g12_l24.out 2>&1 < /dev/null &)

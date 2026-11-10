#!/usr/bin/env bash
# Exploratory strength sweeps (reviewed generation entry point; AllenAI only). C on card 3 now; E on card 0 after the extra arms finish.
D=/data/sanshizhao/Truth_Decoupling/review_e44_claude
W=/workspace/Truth_Decoupling/review_e44_claude
DX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -w $W"
CA="x.w.p0.25 x.w.p0.5 x.w.p2 x.w.m0.25 x.w.m0.5 x.w.m2"
EA="x.w.p0.25 x.w.p0.5 x.w.p0.75 x.w.p1.5 x.w.p2 conf.w.p0.5 conf.w.p2"
( $DX -e FAMILY=C -e LAYERS="19 31" -e ARMS="$CA" -e GPU=3 truth_dec bash $W/e44_round1_extra_gen.sh > $D/logs_round1/dose_gen_C.log 2>&1 \
  && $DX -e FAMILY=C -e NAME=dose_C -e LAYERS="19 31" -e ARMS="x.w.p1 x.w.m1 $CA" -e PROMPTS="x" -e GPU=3 truth_dec bash $W/e44_judge_allenai_subset.sh > $D/logs_round1/dose_judge_C.log 2>&1 ) &
( until grep -qE "extra generation done|Traceback|Error" $D/logs_round1/extra_gen_E.log; do sleep 10; done
  $DX -e FAMILY=E -e LAYERS="8 10" -e ARMS="$EA" -e GPU=0 truth_dec bash $W/e44_round1_extra_gen.sh > $D/logs_round1/dose_gen_E.log 2>&1 \
  && $DX -e FAMILY=E -e NAME=dose_E -e LAYERS="8 10" -e ARMS="x.w.p1 x.w.m1 conf.w.p1 $EA" -e PROMPTS="x conf" -e GPU=0 truth_dec bash $W/e44_judge_allenai_subset.sh > $D/logs_round1/dose_judge_E.log 2>&1 ) &
wait
touch $D/logs_round1/DOSE_DONE

#!/usr/bin/env bash
# Exploratory follow-up of the strength sweeps: controls at the same strengths, then both judges (TAG=B).
#   E layers 8 and 10: random controls / complete-sentence background / negative sign at strengths 0.5 and 0.75; controls on the must-answer prompt at strength 2
#   C layers 19 and 31: the sweep arms under the second judge as well
D=/data/sanshizhao/Truth_Decoupling/review_e44_claude
W=/workspace/Truth_Decoupling/review_e44_claude
DX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -w $W"
EN="x.rand.p0.5 x.rand.m0.5 x_F.w.p0.5 x.w.m0.5 x.md.p0.5 x.rand.p0.75 x.rand.m0.75 x_F.w.p0.75 x.w.m0.75 conf.rand.p2 conf.rand.m2 conf.md.p2 conf_E1.w.m1 conf_E1.w.m2"
EALL="x.w.p0.25 x.w.p0.5 x.w.p0.75 x.w.p1.5 x.w.p2 conf.w.p0.5 conf.w.p2 $EN"
CALL="x.w.p0.25 x.w.p0.5 x.w.p2 x.w.m0.25 x.w.m0.5 x.w.m2"
( $DX -e FAMILY=E -e LAYERS="8 10" -e ARMS="$EN" -e GPU=1 truth_dec bash $W/e44_round1_extra_gen.sh > $D/logs_round1/dose2_gen_E.log 2>&1 || { touch $D/logs_round1/FAILED_dose2_gen_E; exit 1; }
  for layer in 8 10; do
    $DX -e FAMILY=E -e LAYER=$layer -e GPU=1 -e TAG=B -e ARMS="$EALL" -e PROMPTS="x x_F conf conf_E1" truth_dec bash $W/e44_round1_pro.sh > $D/logs_round1/proB_E_L$layer.log 2>&1 &
  done; wait ) &
( for layer in 19 31; do
    $DX -e FAMILY=C -e LAYER=$layer -e GPU=2 -e TAG=B -e ARMS="$CALL" -e PROMPTS="x" truth_dec bash $W/e44_round1_pro.sh > $D/logs_round1/proB_C_L$layer.log 2>&1 &
  done; wait ) &
wait
touch $D/logs_round1/DOSE2_DONE

#!/usr/bin/env bash
# Side branch M (E direction trained between the popular-belief sentence M and the truthful sentence E1):
# the frozen screen (e44_round1_tables.py screen, unchanged) shortlisted layers 11 13 17 12 (+ 10, diagnostic only).
# Step 1  generate the two arms the frozen E criteria still need (second sign of the random control; the same write
#         on the complete-sentence prompt), plus a small exploratory set at layers 11 12 13.
# Step 2  both judges, one process per layer: TAG=A = the frozen-criteria arms; TAG=B = the exploratory arms.
# Same reviewed code tree and judge runner as the first round. Runs on the host; work happens inside the container.
D=/data/sanshizhao/Truth_Decoupling/review_e44_claude
W=/workspace/Truth_Decoupling/review_e44_claude
L=$D/logs_side_M; mkdir -p $L
RUNC=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_M
DX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -w $W"
FROZEN="x.rand.m1 x_F.w.p1"
EXPL="x.w.p0.5 x.w.p2 x.rand.p2 x.rand.m2 x_F.w.p2 conf.w.p1 conf.w.p2 conf.rand.p2 conf.rand.m2"
gen() { $DX -e BRANCH=M -e LAYERS="$1" -e ARMS="$2" -e GPU=$3 truth_dec bash $W/e44_side_extra_gen.sh > $L/gen_$4.log 2>&1 || touch $L/FAILED_gen_$4; }
gen "11" "$FROZEN $EXPL" 0 L11 &
gen "12" "$FROZEN $EXPL" 1 L12 &
gen "13" "$FROZEN $EXPL" 2 L13 &
gen "17 10" "$FROZEN" 3 L17_L10 &
wait
ls $L | grep -q "^FAILED_gen" && { touch $L/SIDE_M_FAILED; exit 1; }
touch $L/SIDE_M_GENERATED
gpu=0
for layer in 11 13 17 12 10; do
  $DX -e RUN=$RUNC -e FAMILY=E -e LAYER=$layer -e GPU=$((gpu % 4)) -e TAG=A truth_dec bash $W/e44_round1_pro.sh > $L/proA_E_L$layer.log 2>&1 || touch $L/FAILED_proA_L$layer &
  gpu=$((gpu + 1))
done
wait
for layer in 11 12 13; do
  $DX -e RUN=$RUNC -e FAMILY=E -e LAYER=$layer -e GPU=$((gpu % 4)) -e TAG=B -e ARMS="$EXPL" -e PROMPTS="x x_F conf" truth_dec bash $W/e44_round1_pro.sh > $L/proB_E_L$layer.log 2>&1 || touch $L/FAILED_proB_L$layer &
  gpu=$((gpu + 1))
done
wait
ls $L | grep -q "^FAILED" && touch $L/SIDE_M_FAILED
touch $L/SIDE_M_JUDGED

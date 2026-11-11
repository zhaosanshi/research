#!/usr/bin/env bash
# Host-side chain for from_claude/37 section 3 (complete-sentence background): selfcheck for base x_F, then E+C and its
# single writes on x_F, the must-answer E on conf_F, one export, both judges, description tables. Card 2 only
# (cards 0 / 1 carry another service). Uses e44_extra_selfcheck.sh and e44_extra_EC.sh.
D=/data/sanshizhao/Truth_Decoupling/review_e44_claude
W=/workspace/Truth_Decoupling/review_e44_claude
L=$D/logs_extra_EC; mkdir -p $L
M=/data/sanshizhao/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b
DX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -w $W"
REC=selfcheck_E10-f62a40ba2c_C31-ba8ca4e7b1.json
cp -n "$M/eval_extra/$REC" "$D/logs_extra_selfcheck/selfcheck_x_full_${REC#selfcheck_}" || { touch $L/F_FAILED; exit 1; }
$DX -e K=0 -e GPU=2 -e ARM=x_F:E10.w.p2+C31.w.m0.25 truth_dec bash $W/e44_extra_selfcheck.sh > $D/logs_extra_selfcheck/selfcheck_xF_full.log 2>&1 || { touch $L/F_FAILED; exit 1; }
cp "$M/eval_extra/$REC" "$D/logs_extra_selfcheck/selfcheck_xF_full_${REC#selfcheck_}"
for s in genF genCF judgeF; do
  $DX -e STAGE=$s -e GPU=2 truth_dec bash $W/e44_extra_EC.sh > $L/$s.log 2>&1 || { touch $L/F_FAILED; exit 1; }
done
mkdir -p $D/tables_EC
python3 $D/e44_EC_describe.py $M/eval_extra/judge_tune_ecF_tune --judge allenai > $D/tables_EC/ECF_describe_allenai.md
python3 $D/e44_EC_describe.py $M/eval_extra/judge_tune_ecF_tune --judge pro > $D/tables_EC/ECF_describe_pro.md
touch $L/F_JUDGED

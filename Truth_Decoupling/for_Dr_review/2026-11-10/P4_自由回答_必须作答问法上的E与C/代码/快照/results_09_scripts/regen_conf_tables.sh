#!/bin/bash
# Regenerate the exploratory must-answer tables with the round-33 fix (untested condition F is never counted as met).
set -euo pipefail
R=/data/sanshizhao/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b
P=/data/sanshizhao/Truth_Decoupling/review_e44_claude/e44_round1_tables.py
O=/data/sanshizhao/Truth_Decoupling/review_e44_claude/tables_r33
mkdir -p "$O"
f1="$O/round1_cells_E_two_judges_conf_base_exploratory_L8_9_10.md"; : > "$f1"
for L in 8 9 10; do for J in allenai pro; do
  python3 "$P" cells "$R" E "$R/judge_tune_proA_E_L$L" --judge $J --layers $L --base conf >> "$f1"; echo >> "$f1"
  echo "Other arms of the same layer and judge (arm | base R / W / H / X | arm R / W / H / X | dR | dAns | dX | keep: n, dR, W>R / R>W | out / in / toH | G_same (n)):" >> "$f1"; echo >> "$f1"
  python3 "$P" cross "$R" E "$R/judge_tune_proA_E_L$L" --judge $J --layers $L | grep -E "^\| (caut|conf)" >> "$f1"; echo >> "$f1"
done; done
f2="$O/round1_cells_E_two_judges_conf_base_strength2_exploratory_L10_L8.md"; : > "$f2"
for L in 10 8; do for J in allenai pro; do
  python3 "$P" cells "$R" E "$R/judge_tune_proB_E_L$L" --judge $J --layers $L --base conf --main conf.w.p2 --strength 2 >> "$f2"; echo >> "$f2"
done; done
wc -l "$f1" "$f2"; grep -h "^KEY" "$f1" "$f2" | cut -c1-240

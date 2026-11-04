# E12 subset-level prediction of the steering gain (gating.npz, condition disagree_a5; categories with n >= 40)

| subset | n | output AUROC | clean probe AUROC | gap | realised dAUROC | base acc | d acc (all) | corrected | damaged | disagree share |
|---|---|---|---|---|---|---|---|---|---|---|
| Confusion: People | 40 | 0.235 | 0.958 | +0.723 | -0.047 | 0.350 | +0.050 | 3 | 1 | 0.57 |
| Superstitions | 40 | 0.655 | 0.983 | +0.328 | +0.220 | 0.650 | +0.250 | 10 | 0 | 0.28 |
| Law | 102 | 0.600 | 0.867 | +0.267 | +0.146 | 0.569 | +0.137 | 16 | 2 | 0.27 |
| Stereotypes | 40 | 0.765 | 0.970 | +0.205 | +0.108 | 0.650 | +0.225 | 10 | 1 | 0.35 |
| History | 48 | 0.745 | 0.939 | +0.194 | -0.035 | 0.667 | +0.062 | 4 | 1 | 0.29 |
| Paranormal | 46 | 0.796 | 0.987 | +0.191 | +0.155 | 0.696 | +0.217 | 11 | 1 | 0.30 |
| Economics | 54 | 0.643 | 0.787 | +0.144 | -0.041 | 0.630 | +0.056 | 5 | 3 | 0.35 |
| Fiction | 50 | 0.862 | 0.995 | +0.133 | +0.016 | 0.680 | +0.200 | 10 | 0 | 0.28 |
| Language | 42 | 0.776 | 0.896 | +0.120 | +0.041 | 0.619 | +0.143 | 8 | 2 | 0.36 |
| Health | 92 | 0.837 | 0.940 | +0.103 | +0.053 | 0.717 | +0.141 | 13 | 0 | 0.26 |
| Misconceptions | 174 | 0.792 | 0.880 | +0.088 | -0.010 | 0.690 | +0.086 | 20 | 5 | 0.24 |
| Sociology | 100 | 0.818 | 0.863 | +0.045 | -0.113 | 0.680 | +0.050 | 8 | 4 | 0.33 |
| Conspiracies | 44 | 0.967 | 0.992 | +0.025 | -0.118 | 0.818 | +0.091 | 7 | 3 | 0.30 |
| Adversarial | 774 | 0.717 | 0.913 | +0.196 | +0.046 | 0.620 | +0.149 | 134 | 21 | 0.33 |
| Non-Adversarial | 668 | 0.802 | 0.922 | +0.120 | -0.017 | 0.717 | +0.091 | 77 | 18 | 0.27 |

## Spearman rank correlation across the 13 categories

| predictor | vs dAUROC | vs d acc | vs corrected - damaged |
|---|---|---|---|
| output AUROC alone (lower = more room) | +0.19 | -0.16 | -0.20 |
| probe AUROC alone | +0.22 | +0.61 | +0.05 |
| gap = probe - output | +0.49 | +0.26 | -0.08 |
| disagreement share | -0.35 | -0.24 | -0.80 |

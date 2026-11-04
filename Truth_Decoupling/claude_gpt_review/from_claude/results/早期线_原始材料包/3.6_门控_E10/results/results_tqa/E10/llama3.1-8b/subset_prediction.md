# E12 subset-level prediction of the steering gain (gating.npz, condition global_a3; categories with n >= 40)

| subset | n | output AUROC | clean probe AUROC | gap | realised dAUROC | base acc | d acc (all) | corrected | damaged | disagree share |
|---|---|---|---|---|---|---|---|---|---|---|
| Confusion: People | 40 | 0.235 | 0.958 | +0.723 | -0.002 | 0.350 | +0.000 | 1 | 1 | 0.57 |
| Superstitions | 40 | 0.655 | 0.983 | +0.328 | +0.265 | 0.650 | +0.225 | 10 | 1 | 0.28 |
| Law | 102 | 0.600 | 0.867 | +0.267 | +0.198 | 0.569 | +0.127 | 16 | 3 | 0.27 |
| Stereotypes | 40 | 0.765 | 0.970 | +0.205 | +0.210 | 0.650 | +0.225 | 11 | 2 | 0.35 |
| History | 48 | 0.745 | 0.939 | +0.194 | +0.111 | 0.667 | +0.042 | 7 | 5 | 0.29 |
| Paranormal | 46 | 0.796 | 0.987 | +0.191 | +0.159 | 0.696 | +0.196 | 9 | 1 | 0.30 |
| Economics | 54 | 0.643 | 0.787 | +0.144 | -0.045 | 0.630 | -0.093 | 3 | 9 | 0.35 |
| Fiction | 50 | 0.862 | 0.995 | +0.133 | +0.091 | 0.680 | +0.180 | 10 | 1 | 0.28 |
| Language | 42 | 0.776 | 0.896 | +0.120 | +0.116 | 0.619 | +0.190 | 13 | 5 | 0.36 |
| Health | 92 | 0.837 | 0.940 | +0.103 | +0.038 | 0.717 | +0.087 | 12 | 4 | 0.26 |
| Misconceptions | 174 | 0.792 | 0.880 | +0.088 | +0.057 | 0.690 | +0.080 | 21 | 7 | 0.24 |
| Sociology | 100 | 0.818 | 0.863 | +0.045 | +0.004 | 0.680 | +0.010 | 8 | 8 | 0.33 |
| Conspiracies | 44 | 0.967 | 0.992 | +0.025 | +0.000 | 0.818 | +0.023 | 7 | 6 | 0.30 |
| Adversarial | 774 | 0.717 | 0.913 | +0.196 | +0.109 | 0.620 | +0.125 | 137 | 46 | 0.33 |
| Non-Adversarial | 668 | 0.802 | 0.922 | +0.120 | +0.058 | 0.717 | +0.055 | 78 | 44 | 0.27 |

## Spearman rank correlation across the 13 categories

| predictor | vs dAUROC | vs d acc | vs corrected - damaged |
|---|---|---|---|
| output AUROC alone (lower = more room) | +0.15 | -0.09 | -0.01 |
| probe AUROC alone | +0.26 | +0.48 | +0.19 |
| gap = probe - output | +0.45 | +0.29 | +0.17 |
| disagreement share | -0.28 | -0.25 | -0.64 |

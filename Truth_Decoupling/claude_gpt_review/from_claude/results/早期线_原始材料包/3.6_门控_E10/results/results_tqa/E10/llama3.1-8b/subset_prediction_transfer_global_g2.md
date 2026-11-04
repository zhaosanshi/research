# E12 subset-level prediction of the steering gain (gating.npz, condition transfer_global_g2; categories with n >= 40)

| subset | n | output AUROC | clean probe AUROC | gap | realised dAUROC | base acc | d acc (all) | corrected | damaged | disagree share |
|---|---|---|---|---|---|---|---|---|---|---|
| Confusion: People | 40 | 0.235 | 0.958 | +0.723 | +0.465 | 0.350 | +0.300 | 13 | 1 | 0.57 |
| Superstitions | 40 | 0.655 | 0.983 | +0.328 | +0.250 | 0.650 | +0.175 | 7 | 0 | 0.28 |
| Law | 102 | 0.600 | 0.867 | +0.267 | +0.185 | 0.569 | +0.118 | 18 | 6 | 0.27 |
| Stereotypes | 40 | 0.765 | 0.970 | +0.205 | +0.162 | 0.650 | +0.125 | 8 | 3 | 0.35 |
| History | 48 | 0.745 | 0.939 | +0.194 | +0.168 | 0.667 | +0.083 | 8 | 4 | 0.29 |
| Paranormal | 46 | 0.796 | 0.987 | +0.191 | +0.125 | 0.696 | +0.087 | 6 | 2 | 0.30 |
| Economics | 54 | 0.643 | 0.787 | +0.144 | +0.118 | 0.630 | -0.019 | 3 | 4 | 0.35 |
| Fiction | 50 | 0.862 | 0.995 | +0.133 | +0.117 | 0.680 | +0.160 | 9 | 1 | 0.28 |
| Language | 42 | 0.776 | 0.896 | +0.120 | +0.050 | 0.619 | +0.167 | 8 | 1 | 0.36 |
| Health | 92 | 0.837 | 0.940 | +0.103 | +0.098 | 0.717 | +0.130 | 13 | 1 | 0.26 |
| Misconceptions | 174 | 0.792 | 0.880 | +0.088 | +0.060 | 0.690 | +0.063 | 16 | 5 | 0.24 |
| Sociology | 100 | 0.818 | 0.863 | +0.045 | +0.058 | 0.680 | +0.010 | 9 | 9 | 0.33 |
| Conspiracies | 44 | 0.967 | 0.992 | +0.025 | +0.027 | 0.818 | +0.045 | 5 | 3 | 0.30 |
| Adversarial | 774 | 0.717 | 0.913 | +0.196 | +0.153 | 0.620 | +0.145 | 137 | 28 | 0.33 |
| Non-Adversarial | 668 | 0.802 | 0.922 | +0.120 | +0.105 | 0.717 | +0.070 | 78 | 33 | 0.27 |

## Spearman rank correlation across the 13 categories

| predictor | vs dAUROC | vs d acc | vs corrected - damaged |
|---|---|---|---|
| output AUROC alone (lower = more room) | +0.77 | +0.25 | +0.20 |
| probe AUROC alone | +0.07 | +0.45 | +0.11 |
| gap = probe - output | +0.96 | +0.57 | +0.34 |
| disagreement share | +0.03 | +0.07 | -0.47 |

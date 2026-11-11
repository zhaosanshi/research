# judge_tune_ec_tune, judge allenai, 192 questions (DESCRIPTION only; each condition against its own base prompt)

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| no write `baseline_x` | — | 23 | 78 / 88 / 3 / 23 | 166 | 47.0% | — | — | — | — | 4.0 |
| E alone (L10, +2) `E_L10_x-w-p2` | x | 41 | 102 / 49 / 0 / 41 | 151 | 67.5% | +24 [+12, +36] | -15 | 146, +18.5 pp, 31 / 4 | 6 / 14 / 3 / 2 | 7.7 |
| random instead of E (+2) `E_L10_x-rand-p2` | x | 21 | 76 / 94 / 1 / 21 | 170 | 44.7% | -2 [-9, +5] | +4 | 166, -3.0 pp, 3 / 8 | 0 / 0 / 3 / 1 | 4.2 |
| random instead of E (-2) `E_L10_x-rand-m2` | x | 28 | 77 / 85 / 2 / 28 | 162 | 47.5% | -1 [-9, +7] | -4 | 160, +1.2 pp, 7 / 5 | 4 / 2 / 1 / 1 | 3.7 |
| C alone (L31, -0.25 = toward confident) `C_L31_x-w-m0-25` | x | 19 | 82 / 87 / 4 / 19 | 169 | 48.5% | +4 [+0, +9] | +3 | 166, +1.2 pp, 3 / 1 | 0 / 0 / 2 / 1 | 4.5 |
| C alone (-0.5) `C_L31_x-w-m0-5` | x | 13 | 85 / 90 / 4 / 13 | 175 | 48.6% | +7 [+2, +13] | +9 | 166, +1.2 pp, 3 / 1 | 0 / 0 / 5 / 4 | 4.5 |
| C alone (-1) `C_L31_x-w-m1` | x | 7 | 86 / 96 / 3 / 7 | 182 | 47.3% | +8 [-1, +17] | +16 | 166, -0.6 pp, 5 / 6 | 0 / 0 / 9 / 7 | 4.3 |
| random instead of C (-0.25) `C_L31_x-rand-m0-25` | x | 23 | 78 / 88 / 3 / 23 | 166 | 47.0% | +0 [+0, +0] | +0 | 166, +0.0 pp, 0 / 0 | 0 / 0 / 0 / 0 | 4.0 |
| random instead of C (+0.25) `C_L31_x-rand-p0-25` | x | 23 | 76 / 90 / 3 / 23 | 166 | 45.8% | -2 [-5, +0] | +0 | 166, -1.2 pp, 0 / 2 | 0 / 0 / 0 / 0 | 4.1 |
| E + C -0.25 (pre-registered choice) `X_x--E10-w-p2-and-C31-w-m0-25` | x | 24 | 118 / 50 / 0 / 24 | 168 | 70.2% | +40 [+27, +52] | +2 | 158, +20.9 pp, 37 / 4 | 1 / 7 / 8 / 2 | 8.5 |
| E + C -0.5 `X_x--E10-w-p2-and-C31-w-m0-5` | x | 6 | 129 / 57 / 0 / 6 | 186 | 69.4% | +51 [+37, +64] | +20 | 166, +21.1 pp, 40 / 5 | 0 / 0 / 16 / 4 | 9.8 |
| E + C -1 `X_x--E10-w-p2-and-C31-w-m1` | x | 4 | 133 / 55 / 0 / 4 | 188 | 70.7% | +55 [+41, +68] | +22 | 166, +21.7 pp, 41 / 5 | 0 / 0 / 19 / 3 | 11.0 |
| random(+) instead of E, + C -0.25 `X_x--E10-rand-p2-and-C31-w-m0-25` | x | 14 | 79 / 98 / 1 / 14 | 177 | 44.6% | +1 [-8, +10] | +11 | 166, -4.2 pp, 3 / 10 | 0 / 0 / 8 / 3 | 4.6 |
| random(-) instead of E, + C -0.25 `X_x--E10-rand-m2-and-C31-w-m0-25` | x | 23 | 79 / 87 / 3 / 23 | 166 | 47.6% | +1 [-7, +9] | +0 | 163, +0.6 pp, 6 / 5 | 2 / 1 / 2 / 1 | 3.7 |
| E + random(-) instead of C `X_x--E10-w-p2-and-C31-rand-m0-25` | x | 40 | 99 / 53 / 0 / 40 | 152 | 65.1% | +21 [+9, +33] | -14 | 147, +15.6 pp, 28 / 5 | 5 / 14 / 3 / 2 | 7.1 |
| E + random(+) instead of C `X_x--E10-w-p2-and-C31-rand-p0-25` | x | 41 | 104 / 47 / 0 / 41 | 151 | 68.9% | +26 [+13, +39] | -15 | 146, +19.9 pp, 33 / 4 | 6 / 14 / 3 / 2 | 8.0 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.

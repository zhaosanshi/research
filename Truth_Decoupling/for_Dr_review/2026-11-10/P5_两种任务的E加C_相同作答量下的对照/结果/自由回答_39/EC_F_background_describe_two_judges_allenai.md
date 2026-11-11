# judge_tune_ecF_tune, judge allenai, 192 questions (DESCRIPTION only; each condition against its own base prompt)

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| no write, complete-sentence prompt x_F `baseline_x_F` | — | 30 | 102 / 59 / 1 / 30 | 161 | 63.4% | — | — | — | — | 9.3 |
| E alone on x_F `E_L10_x_F-w-p2` | x_F | 41 | 116 / 34 / 1 / 41 | 150 | 77.3% | +14 [-1, +29] | -11 | 141, +11.3 pp, 27 / 11 | 11 / 9 / 9 / 0 | 14.2 |
| C alone on x_F (-0.25) `C_L31_x_F-w-m0-25` | x_F | 20 | 110 / 60 / 2 / 20 | 170 | 64.7% | +8 [+2, +15] | +9 | 160, +1.9 pp, 4 / 1 | 1 / 0 / 6 / 4 | 9.5 |
| E + C -0.25 on x_F `X_x_F--E10-w-p2-and-C31-w-m0-25` | x_F | 24 | 130 / 37 / 1 / 24 | 167 | 77.8% | +28 [+13, +43] | +6 | 150, +11.3 pp, 30 / 13 | 6 / 5 / 17 / 0 | 15.7 |
| no write, must-answer + complete sentence (conf_F) `baseline_conf_F` | — | 1 | 112 / 76 / 3 / 1 | 188 | 59.6% | — | — | — | — | 7.2 |
| no write, conf_F + truth sentence (conf_E1_F) `baseline_conf_E1_F` | — | 0 | 133 / 57 / 2 / 0 | 190 | 70.0% | — | — | — | — | 9.6 |
| E alone on conf_F (+1) `E_L10_conf_F-w-p1` | conf_F | 0 | 131 / 59 / 2 / 0 | 190 | 68.9% | +19 [+8, +31] | +2 | 188, +9.0 pp, 26 / 9 | 0 / 0 / 2 / 0 | 9.1 |
| E alone on conf_F (+2) `E_L10_conf_F-w-p2` | conf_F | 0 | 146 / 44 / 2 / 0 | 190 | 76.8% | +34 [+21, +48] | +2 | 187, +17.1 pp, 43 / 11 | 0 / 1 / 2 / 1 | 12.4 |
| random instead of E on conf_F (+2) `E_L10_conf_F-rand-p2` | conf_F | 1 | 110 / 78 / 3 / 1 | 188 | 58.5% | -2 [-11, +7] | +0 | 188, -1.1 pp, 11 / 13 | 0 / 0 / 0 / 0 | 8.3 |
| random instead of E on conf_F (-2) `E_L10_conf_F-rand-m2` | conf_F | 1 | 111 / 78 / 2 / 1 | 189 | 58.7% | -1 [-11, +9] | +1 | 187, -1.1 pp, 10 / 12 | 1 / 0 / 2 / 0 | 6.4 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.

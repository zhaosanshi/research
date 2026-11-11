# judge_tune_ecF_tune, judge pro, 192 questions (DESCRIPTION only; each condition against its own base prompt)

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| no write, complete-sentence prompt x_F `baseline_x_F` | — | 30 | 80 / 80 / 2 / 30 | 160 | 50.0% | — | — | — | — | 9.3 |
| E alone on x_F `E_L10_x_F-w-p2` | x_F | 41 | 95 / 54 / 2 / 41 | 149 | 63.8% | +15 [+1, +29] | -11 | 139, +7.2 pp, 23 / 13 | 5 / 16 / 10 / 0 | 14.2 |
| C alone on x_F (-0.25) `C_L31_x_F-w-m0-25` | x_F | 20 | 86 / 82 / 3 / 21 | 168 | 51.2% | +6 [+1, +12] | +8 | 160, +1.9 pp, 4 / 1 | 0 / 0 / 3 / 5 | 9.5 |
| E + C -0.25 on x_F `X_x_F--E10-w-p2-and-C31-w-m0-25` | x_F | 24 | 104 / 58 / 4 / 26 | 162 | 64.2% | +24 [+10, +39] | +2 | 150, +10.0 pp, 29 / 14 | 3 / 7 / 12 / 0 | 15.7 |
| no write, must-answer + complete sentence (conf_F) `baseline_conf_F` | — | 1 | 87 / 99 / 6 / 0 | 186 | 46.8% | — | — | — | — | 7.2 |
| no write, conf_F + truth sentence (conf_E1_F) `baseline_conf_E1_F` | — | 0 | 99 / 84 / 8 / 1 | 183 | 54.1% | — | — | — | — | 9.6 |
| E alone on conf_F (+1) `E_L10_conf_F-w-p1` | conf_F | 0 | 103 / 81 / 6 / 2 | 184 | 56.0% | +16 [+5, +27] | -2 | 183, +8.7 pp, 24 / 8 | 1 / 2 / 1 / 0 | 9.1 |
| E alone on conf_F (+2) `E_L10_conf_F-w-p2` | conf_F | 0 | 113 / 70 / 6 / 3 | 183 | 61.7% | +26 [+13, +40] | -3 | 180, +13.9 pp, 36 / 11 | 2 / 4 / 3 / 0 | 12.4 |
| random instead of E on conf_F (+2) `E_L10_conf_F-rand-p2` | conf_F | 1 | 87 / 98 / 7 / 0 | 185 | 47.0% | +0 [-9, +9] | -1 | 184, +1.1 pp, 9 / 7 | 2 / 0 / 0 / 1 | 8.3 |
| random instead of E on conf_F (-2) `E_L10_conf_F-rand-m2` | conf_F | 1 | 84 / 100 / 8 / 0 | 184 | 45.7% | -3 [-10, +4] | -2 | 184, -1.6 pp, 5 / 8 | 0 / 2 / 0 / 0 | 6.4 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.

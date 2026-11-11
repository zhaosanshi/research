# judge_tune_ec_tune, judge pro, 192 questions (DESCRIPTION only; each condition against its own base prompt)

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| no write `baseline_x` | — | 23 | 64 / 98 / 6 / 24 | 162 | 39.5% | — | — | — | — | 4.0 |
| E alone (L10, +2) `E_L10_x-w-p2` | x | 41 | 85 / 67 / 1 / 39 | 152 | 55.9% | +21 [+9, +34] | -10 | 144, +10.4 pp, 24 / 9 | 2 / 16 / 8 / 0 | 7.7 |
| random instead of E (+2) `E_L10_x-rand-p2` | x | 21 | 62 / 102 / 6 / 22 | 164 | 37.8% | -2 [-9, +5] | +2 | 161, -1.9 pp, 3 / 6 | 1 / 0 / 2 / 1 | 4.2 |
| random instead of E (-2) `E_L10_x-rand-m2` | x | 28 | 65 / 92 / 7 / 28 | 157 | 41.4% | +1 [-6, +8] | -5 | 156, +1.3 pp, 6 / 4 | 2 / 4 / 1 / 0 | 3.7 |
| C alone (L31, -0.25 = toward confident) `C_L31_x-w-m0-25` | x | 19 | 66 / 98 / 7 / 21 | 164 | 40.2% | +2 [-2, +6] | +2 | 162, +0.6 pp, 2 / 1 | 0 / 0 / 1 / 1 | 4.5 |
| C alone (-0.5) `C_L31_x-w-m0-5` | x | 13 | 67 / 101 / 8 / 16 | 168 | 39.9% | +3 [-1, +8] | +6 | 162, +0.6 pp, 2 / 1 | 0 / 0 / 2 / 4 | 4.5 |
| C alone (-1) `C_L31_x-w-m1` | x | 7 | 73 / 104 / 8 / 7 | 177 | 41.2% | +9 [+1, +17] | +15 | 162, +1.9 pp, 7 / 4 | 0 / 0 / 6 / 9 | 4.3 |
| random instead of C (-0.25) `C_L31_x-rand-m0-25` | x | 23 | 65 / 98 / 5 / 24 | 163 | 39.9% | +1 [+0, +3] | +1 | 162, +0.0 pp, 0 / 0 | 0 / 0 / 1 / 0 | 4.0 |
| random instead of C (+0.25) `C_L31_x-rand-p0-25` | x | 23 | 64 / 99 / 5 / 24 | 163 | 39.3% | +0 [-3, +3] | +1 | 162, -0.6 pp, 0 / 1 | 0 / 0 / 1 / 0 | 4.1 |
| E + C -0.25 (pre-registered choice) `X_x--E10-w-p2-and-C31-w-m0-25` | x | 24 | 93 / 68 / 4 / 27 | 161 | 57.8% | +29 [+16, +43] | -1 | 152, +15.1 pp, 32 / 9 | 3 / 7 / 9 / 0 | 8.5 |
| E + C -0.5 `X_x--E10-w-p2-and-C31-w-m0-5` | x | 6 | 103 / 75 / 5 / 9 | 178 | 57.9% | +39 [+25, +53] | +16 | 159, +15.7 pp, 35 / 10 | 2 / 1 / 16 / 3 | 9.8 |
| E + C -1 `X_x--E10-w-p2-and-C31-w-m1` | x | 4 | 103 / 76 / 7 / 6 | 179 | 57.5% | +39 [+24, +54] | +17 | 159, +15.7 pp, 36 / 11 | 2 / 1 / 16 / 4 | 11.0 |
| random(+) instead of E, + C -0.25 `X_x--E10-rand-p2-and-C31-w-m0-25` | x | 14 | 64 / 103 / 8 / 17 | 167 | 38.3% | +0 [-8, +8] | +5 | 161, -1.2 pp, 5 / 7 | 1 / 0 / 3 / 3 | 4.6 |
| random(-) instead of E, + C -0.25 `X_x--E10-rand-m2-and-C31-w-m0-25` | x | 23 | 66 / 94 / 8 / 24 | 160 | 41.2% | +2 [-5, +9] | -2 | 159, +1.9 pp, 6 / 3 | 2 / 1 / 1 / 0 | 3.7 |
| E + random(-) instead of C `X_x--E10-w-p2-and-C31-rand-m0-25` | x | 40 | 82 / 70 / 1 / 39 | 152 | 53.9% | +18 [+6, +31] | -10 | 144, +8.3 pp, 23 / 11 | 2 / 16 / 8 / 0 | 7.1 |
| E + random(+) instead of C `X_x--E10-w-p2-and-C31-rand-p0-25` | x | 41 | 86 / 65 / 1 / 40 | 151 | 57.0% | +22 [+10, +35] | -11 | 143, +11.9 pp, 25 / 8 | 3 / 16 / 8 / 0 | 8.0 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.

# judge_tune_ecll31_tune, judge pro, 192 questions (DESCRIPTION only; each condition against its own base prompt)

Choice of k fixed before judging (rule abstentions; baseline x = 23): E +1 -> C -0.5, E +2 -> C -0.5

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_x` | — | 23 | 64 / 98 / 6 / 24 | 162 | 39.5% | — | — | — | — | 4.0 |
| `E_L11_x-rand-m1` | x | 26 | 65 / 95 / 5 / 27 | 160 | 40.6% | +1 [-5, +7] | -2 | 159, +0.0 pp, 4 / 4 | 0 / 3 / 1 / 0 | 4.1 |
| `E_L11_x-rand-m2` | x | 29 | 62 / 94 / 6 / 30 | 156 | 39.7% | -2 [-9, +5] | -6 | 155, -1.3 pp, 4 / 6 | 1 / 6 / 1 / 0 | 4.3 |
| `E_L11_x-rand-p1` | x | 22 | 66 / 99 / 4 / 23 | 165 | 40.0% | +2 [-2, +6] | +3 | 162, +0.0 pp, 1 / 1 | 0 / 0 / 2 / 1 | 4.0 |
| `E_L11_x-rand-p2` | x | 21 | 64 / 101 / 6 / 21 | 165 | 38.8% | +0 [-7, +7] | +3 | 160, -0.6 pp, 4 / 5 | 1 / 1 / 2 / 3 | 4.0 |
| `E_L11_x-w-p1` | x | 53 | 71 / 65 / 3 / 53 | 136 | 52.2% | +7 [-4, +18] | -26 | 130, +7.7 pp, 15 / 5 | 8 / 24 / 5 / 1 | 4.6 |
| `E_L11_x-w-p2` | x | 77 | 69 / 39 / 5 / 79 | 108 | 63.9% | +5 [-10, +20] | -54 | 101, +15.8 pp, 26 / 10 | 16 / 45 / 5 / 2 | 5.1 |
| `C_L31_x-rand-m0-5` | x | 23 | 67 / 96 / 5 / 24 | 163 | 41.1% | +3 [+0, +7] | +1 | 162, +1.2 pp, 2 / 0 | 0 / 0 / 1 / 0 | 4.1 |
| `C_L31_x-rand-p0-5` | x | 23 | 66 / 97 / 5 / 24 | 163 | 40.5% | +2 [-2, +6] | +1 | 162, +0.6 pp, 2 / 1 | 0 / 0 / 1 / 0 | 4.1 |
| `C_L31_x-w-m0-25` | x | 19 | 66 / 98 / 7 / 21 | 164 | 40.2% | +2 [-2, +6] | +2 | 162, +0.6 pp, 2 / 1 | 0 / 0 / 1 / 1 | 4.5 |
| `C_L31_x-w-m0-5` | x | 13 | 67 / 101 / 8 / 16 | 168 | 39.9% | +3 [-1, +8] | +6 | 162, +0.6 pp, 2 / 1 | 0 / 0 / 2 / 4 | 4.5 |
| `C_L31_x-w-m1` | x | 7 | 73 / 104 / 8 / 7 | 177 | 41.2% | +9 [+1, +17] | +15 | 162, +1.9 pp, 7 / 4 | 0 / 0 / 6 / 9 | 4.3 |
| `C_L31_x-w-m2` | x | 2 | 73 / 106 / 10 / 3 | 179 | 40.8% | +9 [-1, +19] | +17 | 162, +0.0 pp, 8 / 8 | 0 / 0 / 9 / 8 | 5.0 |
| `X_x--E11-rand-m1-and-C31-w-m0-5` | x | 13 | 67 / 100 / 8 / 17 | 167 | 40.1% | +3 [-3, +10] | +5 | 162, +0.6 pp, 5 / 4 | 0 / 0 / 2 / 3 | 4.7 |
| `X_x--E11-rand-m2-and-C31-w-m0-5` | x | 16 | 66 / 98 / 9 / 19 | 164 | 40.2% | +2 [-5, +9] | +2 | 160, +0.0 pp, 5 / 5 | 1 / 1 / 3 / 1 | 4.7 |
| `X_x--E11-rand-p1-and-C31-w-m0-5` | x | 13 | 67 / 102 / 7 / 16 | 169 | 39.6% | +3 [-3, +9] | +7 | 162, +0.0 pp, 3 / 3 | 0 / 0 / 3 / 4 | 4.2 |
| `X_x--E11-rand-p2-and-C31-w-m0-5` | x | 12 | 65 / 105 / 7 / 15 | 170 | 38.2% | +1 [-7, +9] | +8 | 161, -1.2 pp, 5 / 7 | 1 / 0 / 4 / 5 | 4.3 |
| `X_x--E11-w-p1-and-C31-rand-m0-5` | x | 52 | 69 / 68 / 3 / 52 | 137 | 50.4% | +5 [-6, +17] | -25 | 131, +6.1 pp, 15 / 7 | 8 / 23 / 5 / 1 | 4.6 |
| `X_x--E11-w-p1-and-C31-rand-p0-5` | x | 53 | 70 / 65 / 3 / 54 | 135 | 51.9% | +6 [-5, +17] | -27 | 129, +7.0 pp, 14 / 5 | 8 / 25 / 5 / 1 | 4.7 |
| `X_x--E11-w-p1-and-C31-w-m0-25` | x | 36 | 76 / 76 / 3 / 37 | 152 | 50.0% | +12 [+1, +23] | -10 | 145, +6.2 pp, 17 / 8 | 3 / 14 / 6 / 1 | 4.9 |
| `X_x--E11-w-p1-and-C31-w-m0-5` (chosen k) | x | 22 | 80 / 83 / 5 / 24 | 163 | 49.1% | +16 [+4, +28] | +1 | 153, +5.2 pp, 17 / 9 | 2 / 7 / 10 / 0 | 5.4 |
| `X_x--E11-w-p1-and-C31-w-m1` | x | 5 | 93 / 82 / 6 / 11 | 175 | 53.1% | +29 [+17, +41] | +13 | 160, +10.6 pp, 24 / 7 | 1 / 1 / 13 / 2 | 6.1 |
| `X_x--E11-w-p1-and-C31-w-m2` | x | 2 | 91 / 90 / 5 / 6 | 181 | 50.3% | +27 [+14, +40] | +19 | 161, +7.5 pp, 21 / 9 | 1 / 0 / 16 / 4 | 6.7 |
| `X_x--E11-w-p2-and-C31-rand-m0-5` | x | 76 | 69 / 40 / 5 / 78 | 109 | 63.3% | +5 [-10, +20] | -53 | 102, +15.7 pp, 26 / 10 | 16 / 44 / 5 / 2 | 5.1 |
| `X_x--E11-w-p2-and-C31-rand-p0-5` | x | 77 | 69 / 37 / 6 / 80 | 106 | 65.1% | +5 [-10, +20] | -56 | 99, +17.2 pp, 28 / 11 | 17 / 46 / 5 / 2 | 4.9 |
| `X_x--E11-w-p2-and-C31-w-m0-25` | x | 39 | 89 / 55 / 7 / 41 | 144 | 61.8% | +25 [+8, +42] | -18 | 129, +18.6 pp, 39 / 15 | 12 / 21 / 13 / 2 | 6.2 |
| `X_x--E11-w-p2-and-C31-w-m0-5` (chosen k) | x | 19 | 99 / 63 / 8 / 22 | 162 | 61.1% | +35 [+18, +51] | +0 | 143, +18.2 pp, 41 / 15 | 7 / 12 / 16 / 3 | 7.7 |
| `X_x--E11-w-p2-and-C31-w-m1` | x | 2 | 106 / 70 / 10 / 6 | 176 | 60.2% | +42 [+25, +59] | +14 | 155, +16.8 pp, 45 / 19 | 2 / 5 / 18 / 3 | 8.8 |
| `X_x--E11-w-p2-and-C31-w-m2` | x | 3 | 115 / 66 / 7 / 4 | 181 | 63.5% | +51 [+35, +67] | +19 | 158, +21.5 pp, 50 / 16 | 1 / 3 / 18 / 5 | 10.5 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.

## The questions E alone turns into "I don't know" (rule abstention; answered without any write)

- E +1: 31 questions. Without the write they were R / W / H / X = 7 / 23 / 0 / 1 (right 23% of R + W); the questions answered both times were 57 / 75 / 6 / 0 (right 43%).
  With E +1 and C -0.5 (toward confident), 24 of them are no longer rule abstentions: R / W / H / X = 6 / 17 / 0 / 1.
- E +2: 59 questions. Without the write they were R / W / H / X = 16 / 42 / 1 / 0 (right 28% of R + W); the questions answered both times were 48 / 56 / 5 / 1 (right 46%).
  With E +2 and C -0.5 (toward confident), 44 of them are no longer rule abstentions: R / W / H / X = 21 / 22 / 1 / 0.

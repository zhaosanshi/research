# judge_tune_ecll19_tune, judge pro, 192 questions (DESCRIPTION only; each condition against its own base prompt)

Choice of k fixed before judging (rule abstentions; baseline x = 23): E +1 -> C -0.5, E +2 -> C -0.5

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_x` | — | 23 | 64 / 98 / 6 / 24 | 162 | 39.5% | — | — | — | — | 4.0 |
| `E_L11_x-rand-m1` | x | 26 | 65 / 95 / 5 / 27 | 160 | 40.6% | +1 [-5, +7] | -2 | 159, +0.0 pp, 4 / 4 | 0 / 3 / 1 / 0 | 4.1 |
| `E_L11_x-rand-m2` | x | 29 | 64 / 92 / 6 / 30 | 156 | 41.0% | +0 [-6, +6] | -6 | 155, +0.0 pp, 4 / 4 | 1 / 6 / 1 / 0 | 4.3 |
| `E_L11_x-rand-p1` | x | 22 | 66 / 99 / 4 / 23 | 165 | 40.0% | +2 [-2, +6] | +3 | 162, +0.0 pp, 1 / 1 | 0 / 0 / 2 / 1 | 4.0 |
| `E_L11_x-rand-p2` | x | 21 | 63 / 102 / 6 / 21 | 165 | 38.2% | -1 [-8, +6] | +3 | 160, -1.2 pp, 3 / 5 | 1 / 1 / 2 / 3 | 4.0 |
| `E_L11_x-w-p1` | x | 53 | 71 / 65 / 3 / 53 | 136 | 52.2% | +7 [-4, +18] | -26 | 130, +7.7 pp, 15 / 5 | 8 / 24 / 5 / 1 | 4.6 |
| `E_L11_x-w-p2` | x | 77 | 69 / 39 / 5 / 79 | 108 | 63.9% | +5 [-10, +20] | -54 | 101, +15.8 pp, 26 / 10 | 16 / 45 / 5 / 2 | 5.1 |
| `C_L19_x-rand-m0-5` | x | 23 | 68 / 95 / 5 / 24 | 163 | 41.7% | +4 [+1, +8] | +1 | 162, +1.9 pp, 3 / 0 | 0 / 0 / 1 / 0 | 4.0 |
| `C_L19_x-rand-p0-5` | x | 24 | 65 / 97 / 5 / 25 | 162 | 40.1% | +1 [-3, +6] | +0 | 161, +0.0 pp, 2 / 2 | 0 / 1 / 1 / 0 | 4.0 |
| `C_L19_x-w-m0-25` | x | 20 | 66 / 97 / 8 / 21 | 163 | 40.5% | +2 [+0, +5] | +1 | 162, +1.2 pp, 2 / 0 | 0 / 0 / 0 / 1 | 4.1 |
| `C_L19_x-w-m0-5` | x | 15 | 66 / 100 / 9 / 17 | 166 | 39.8% | +2 [-3, +8] | +4 | 162, +0.6 pp, 4 / 3 | 0 / 0 / 1 / 3 | 4.1 |
| `C_L19_x-w-m1` | x | 8 | 67 / 106 / 9 / 10 | 173 | 38.7% | +3 [-5, +11] | +11 | 162, -0.6 pp, 6 / 7 | 0 / 0 / 4 / 7 | 4.3 |
| `C_L19_x-w-m2` | x | 2 | 69 / 114 / 6 / 3 | 183 | 37.7% | +5 [-5, +15] | +21 | 162, -3.1 pp, 5 / 10 | 0 / 0 / 10 / 11 | 3.9 |
| `X_x--E11-rand-m1-and-C19-w-m0-5` | x | 16 | 65 / 100 / 9 / 18 | 165 | 39.4% | +1 [-5, +8] | +3 | 162, +0.0 pp, 5 / 5 | 0 / 0 / 1 / 2 | 4.3 |
| `X_x--E11-rand-m2-and-C19-w-m0-5` | x | 19 | 68 / 95 / 7 / 22 | 163 | 41.7% | +4 [-2, +10] | +1 | 160, +1.2 pp, 5 / 3 | 0 / 2 / 2 / 1 | 4.3 |
| `X_x--E11-rand-p1-and-C19-w-m0-5` | x | 15 | 65 / 103 / 7 / 17 | 168 | 38.7% | +1 [-5, +7] | +6 | 162, -0.6 pp, 3 / 4 | 0 / 0 / 2 / 4 | 4.0 |
| `X_x--E11-rand-p2-and-C19-w-m0-5` | x | 15 | 61 / 107 / 7 / 17 | 168 | 36.3% | -3 [-10, +4] | +6 | 161, -3.1 pp, 3 / 8 | 0 / 1 / 2 / 5 | 4.1 |
| `X_x--E11-w-p1-and-C19-rand-m0-5` | x | 51 | 67 / 68 / 3 / 54 | 135 | 49.6% | +3 [-9, +15] | -27 | 130, +5.4 pp, 16 / 9 | 8 / 24 / 4 / 1 | 4.7 |
| `X_x--E11-w-p1-and-C19-rand-p0-5` | x | 56 | 69 / 64 / 3 / 56 | 133 | 51.9% | +5 [-6, +16] | -29 | 128, +7.8 pp, 15 / 5 | 9 / 25 / 4 / 1 | 4.6 |
| `X_x--E11-w-p1-and-C19-w-m0-25` | x | 41 | 77 / 70 / 3 / 42 | 147 | 52.4% | +13 [+2, +24] | -15 | 140, +8.6 pp, 16 / 4 | 5 / 17 / 6 / 1 | 4.7 |
| `X_x--E11-w-p1-and-C19-w-m0-5` (chosen k) | x | 33 | 80 / 74 / 4 / 34 | 154 | 51.9% | +16 [+5, +27] | -8 | 147, +8.8 pp, 18 / 5 | 3 / 12 / 6 / 1 | 4.7 |
| `X_x--E11-w-p1-and-C19-w-m1` | x | 9 | 89 / 85 / 5 / 13 | 174 | 51.1% | +25 [+13, +37] | +12 | 158, +8.2 pp, 20 / 7 | 2 / 2 / 14 / 2 | 5.2 |
| `X_x--E11-w-p1-and-C19-w-m2` | x | 1 | 92 / 87 / 9 / 4 | 179 | 51.4% | +28 [+15, +41] | +17 | 159, +7.5 pp, 20 / 8 | 2 / 1 / 18 / 2 | 4.8 |
| `X_x--E11-w-p2-and-C19-rand-m0-5` | x | 72 | 68 / 43 / 6 / 75 | 111 | 61.3% | +4 [-11, +19] | -51 | 104, +14.4 pp, 26 / 11 | 16 / 42 / 5 / 2 | 5.1 |
| `X_x--E11-w-p2-and-C19-rand-p0-5` | x | 79 | 66 / 40 / 5 / 81 | 106 | 62.3% | +2 [-13, +16] | -56 | 99, +13.1 pp, 24 / 11 | 16 / 47 / 5 / 2 | 4.9 |
| `X_x--E11-w-p2-and-C19-w-m0-25` | x | 53 | 84 / 47 / 7 / 54 | 131 | 64.1% | +20 [+4, +36] | -31 | 117, +18.8 pp, 34 / 12 | 14 / 31 / 12 / 2 | 5.8 |
| `X_x--E11-w-p2-and-C19-w-m0-5` (chosen k) | x | 32 | 92 / 58 / 8 / 34 | 150 | 61.3% | +28 [+11, +45] | -12 | 133, +18.0 pp, 38 / 14 | 11 / 18 / 15 / 2 | 6.3 |
| `X_x--E11-w-p2-and-C19-w-m1` | x | 5 | 103 / 71 / 9 / 9 | 174 | 59.2% | +39 [+23, +56] | +12 | 154, +17.5 pp, 44 / 17 | 3 / 5 / 15 / 5 | 6.7 |
| `X_x--E11-w-p2-and-C19-w-m2` | x | 1 | 111 / 70 / 8 / 3 | 181 | 61.3% | +47 [+31, +63] | +19 | 159, +19.5 pp, 48 / 17 | 1 / 2 / 17 / 5 | 6.5 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.

## The questions E alone turns into "I don't know" (rule abstention; answered without any write)

- E +1: 31 questions. Without the write they were R / W / H / X = 7 / 23 / 0 / 1 (right 23% of R + W); the questions answered both times were 57 / 75 / 6 / 0 (right 43%).
  With E +1 and C -0.5 (toward confident), 18 of them are no longer rule abstentions: R / W / H / X = 6 / 11 / 0 / 1.
- E +2: 59 questions. Without the write they were R / W / H / X = 16 / 42 / 1 / 0 (right 28% of R + W); the questions answered both times were 48 / 56 / 5 / 1 (right 46%).
  With E +2 and C -0.5 (toward confident), 37 of them are no longer rule abstentions: R / W / H / X = 14 / 20 / 3 / 0.

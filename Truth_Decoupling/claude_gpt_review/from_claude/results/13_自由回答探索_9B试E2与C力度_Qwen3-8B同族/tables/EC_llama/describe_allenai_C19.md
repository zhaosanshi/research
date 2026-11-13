# judge_tune_ecll19_tune, judge allenai, 192 questions (DESCRIPTION only; each condition against its own base prompt)

Choice of k fixed before judging (rule abstentions; baseline x = 23): E +1 -> C -0.5, E +2 -> C -0.5

| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_x` | — | 23 | 78 / 88 / 3 / 23 | 166 | 47.0% | — | — | — | — | 4.0 |
| `E_L11_x-rand-m1` | x | 26 | 81 / 84 / 1 / 26 | 165 | 49.1% | +3 [-3, +9] | -1 | 163, +2.5 pp, 5 / 1 | 2 / 1 / 1 / 1 | 4.1 |
| `E_L11_x-rand-m2` | x | 29 | 88 / 74 / 1 / 29 | 162 | 54.3% | +10 [+2, +18] | -4 | 160, +6.2 pp, 12 / 2 | 2 / 4 / 2 / 0 | 4.3 |
| `E_L11_x-rand-p1` | x | 22 | 80 / 88 / 2 / 22 | 168 | 47.6% | +2 [-2, +6] | +2 | 166, +0.6 pp, 2 / 1 | 0 / 0 / 1 / 1 | 4.0 |
| `E_L11_x-rand-p2` | x | 21 | 78 / 91 / 2 / 21 | 169 | 46.2% | +0 [-9, +9] | +3 | 165, -1.2 pp, 7 / 9 | 1 / 0 / 3 / 1 | 4.0 |
| `E_L11_x-w-p1` | x | 53 | 86 / 53 / 0 / 53 | 139 | 61.9% | +8 [-4, +20] | -27 | 135, +12.6 pp, 22 / 5 | 11 / 20 / 2 / 2 | 4.6 |
| `E_L11_x-w-p2` | x | 77 | 82 / 31 / 2 / 77 | 113 | 72.6% | +4 [-12, +20] | -53 | 106, +17.9 pp, 30 / 11 | 20 / 40 / 5 / 2 | 5.1 |
| `C_L19_x-rand-m0-5` | x | 23 | 80 / 86 / 3 / 23 | 166 | 48.2% | +2 [-2, +6] | +0 | 166, +1.2 pp, 3 / 1 | 0 / 0 / 0 / 0 | 4.0 |
| `C_L19_x-rand-p0-5` | x | 24 | 74 / 92 / 2 / 24 | 166 | 44.6% | -4 [-8, -1] | +0 | 165, -2.4 pp, 0 / 4 | 0 / 1 / 0 / 1 | 4.0 |
| `C_L19_x-w-m0-25` | x | 20 | 82 / 87 / 3 / 20 | 169 | 48.5% | +4 [+0, +9] | +3 | 166, +1.8 pp, 4 / 1 | 0 / 0 / 1 / 2 | 4.1 |
| `C_L19_x-w-m0-5` | x | 15 | 84 / 91 / 2 / 15 | 175 | 48.0% | +6 [-1, +13] | +9 | 166, +0.6 pp, 5 / 4 | 0 / 0 / 5 / 4 | 4.1 |
| `C_L19_x-w-m1` | x | 8 | 86 / 96 / 2 / 8 | 182 | 47.3% | +8 [-2, +18] | +16 | 166, -1.2 pp, 7 / 9 | 0 / 0 / 10 / 6 | 4.3 |
| `C_L19_x-w-m2` | x | 2 | 84 / 105 / 1 / 2 | 189 | 44.4% | +6 [-6, +17] | +23 | 165, -4.2 pp, 7 / 14 | 0 / 1 / 13 / 11 | 3.9 |
| `X_x--E11-rand-m1-and-C19-w-m0-5` | x | 16 | 88 / 86 / 2 / 16 | 174 | 50.6% | +10 [+3, +18] | +8 | 166, +2.4 pp, 7 / 3 | 0 / 0 / 6 / 2 | 4.3 |
| `X_x--E11-rand-m2-and-C19-w-m0-5` | x | 19 | 90 / 82 / 1 / 19 | 172 | 52.3% | +12 [+3, +21] | +6 | 164, +4.3 pp, 10 / 3 | 2 / 0 / 7 / 1 | 4.3 |
| `X_x--E11-rand-p1-and-C19-w-m0-5` | x | 15 | 78 / 95 / 4 / 15 | 173 | 45.1% | +0 [-7, +7] | +7 | 165, -2.4 pp, 2 / 6 | 1 / 0 / 5 / 3 | 4.0 |
| `X_x--E11-rand-p2-and-C19-w-m0-5` | x | 15 | 76 / 98 / 3 / 15 | 174 | 43.7% | -2 [-11, +7] | +8 | 166, -3.6 pp, 5 / 11 | 0 / 0 / 4 / 4 | 4.1 |
| `X_x--E11-w-p1-and-C19-rand-m0-5` | x | 51 | 89 / 52 / 0 / 51 | 141 | 63.1% | +11 [-1, +24] | -25 | 137, +12.4 pp, 23 / 6 | 9 / 20 / 3 / 1 | 4.7 |
| `X_x--E11-w-p1-and-C19-rand-p0-5` | x | 56 | 83 / 53 / 0 / 56 | 136 | 61.0% | +5 [-8, +18] | -30 | 132, +12.1 pp, 23 / 7 | 13 / 21 / 2 / 2 | 4.6 |
| `X_x--E11-w-p1-and-C19-w-m0-25` | x | 41 | 88 / 63 / 0 / 41 | 151 | 58.3% | +10 [-2, +22] | -15 | 146, +8.2 pp, 22 / 10 | 5 / 15 / 3 / 2 | 4.7 |
| `X_x--E11-w-p1-and-C19-w-m0-5` (chosen k) | x | 33 | 90 / 69 / 0 / 33 | 159 | 56.6% | +12 [+0, +24] | -7 | 152, +7.2 pp, 22 / 11 | 3 / 11 / 4 / 3 | 4.7 |
| `X_x--E11-w-p1-and-C19-w-m1` | x | 9 | 110 / 73 / 0 / 9 | 183 | 60.1% | +32 [+18, +46] | +17 | 164, +10.4 pp, 28 / 11 | 1 / 1 / 16 / 3 | 5.2 |
| `X_x--E11-w-p1-and-C19-w-m2` | x | 1 | 111 / 80 / 0 / 1 | 191 | 58.1% | +33 [+18, +47] | +25 | 166, +6.0 pp, 23 / 13 | 0 / 0 / 23 / 2 | 4.8 |
| `X_x--E11-w-p2-and-C19-rand-m0-5` | x | 72 | 84 / 34 / 2 / 72 | 118 | 71.2% | +6 [-10, +22] | -48 | 111, +18.0 pp, 31 / 11 | 19 / 36 / 5 / 2 | 5.1 |
| `X_x--E11-w-p2-and-C19-rand-p0-5` | x | 79 | 78 / 33 / 2 / 79 | 111 | 70.3% | +0 [-16, +15] | -55 | 104, +14.4 pp, 27 / 12 | 20 / 42 / 5 / 2 | 4.9 |
| `X_x--E11-w-p2-and-C19-w-m0-25` | x | 53 | 94 / 39 / 6 / 53 | 133 | 70.7% | +16 [+0, +32] | -33 | 122, +17.2 pp, 34 / 13 | 14 / 30 / 9 / 2 | 5.8 |
| `X_x--E11-w-p2-and-C19-w-m0-5` (chosen k) | x | 32 | 114 / 42 / 4 / 32 | 156 | 73.1% | +36 [+19, +53] | -10 | 141, +24.1 pp, 47 / 13 | 11 / 14 / 13 / 2 | 6.3 |
| `X_x--E11-w-p2-and-C19-w-m1` | x | 5 | 134 / 49 / 4 / 5 | 183 | 73.2% | +56 [+39, +73] | +17 | 158, +24.1 pp, 52 / 14 | 3 / 5 / 21 / 4 | 6.7 |
| `X_x--E11-w-p2-and-C19-w-m2` | x | 1 | 138 / 50 / 3 / 1 | 188 | 73.4% | +60 [+44, +76] | +22 | 163, +23.9 pp, 51 / 12 | 0 / 3 / 21 / 4 | 6.5 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.

## The questions E alone turns into "I don't know" (rule abstention; answered without any write)

- E +1: 31 questions. Without the write they were R / W / H / X = 11 / 20 / 0 / 0 (right 35% of R + W); the questions answered both times were 67 / 68 / 3 / 0 (right 50%).
  With E +1 and C -0.5 (toward confident), 18 of them are no longer rule abstentions: R / W / H / X = 8 / 10 / 0 / 0.
- E +2: 59 questions. Without the write they were R / W / H / X = 20 / 38 / 1 / 0 (right 34% of R + W); the questions answered both times were 58 / 50 / 2 / 0 (right 54%).
  With E +2 and C -0.5 (toward confident), 37 of them are no longer rule abstentions: R / W / H / X = 28 / 7 / 2 / 0.

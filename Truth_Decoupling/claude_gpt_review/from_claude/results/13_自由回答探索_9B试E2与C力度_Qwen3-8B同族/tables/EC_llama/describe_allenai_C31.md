# judge_tune_ecll31_tune, judge allenai, 192 questions (DESCRIPTION only; each condition against its own base prompt)

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
| `C_L31_x-rand-m0-5` | x | 23 | 80 / 87 / 2 / 23 | 167 | 47.9% | +2 [+0, +5] | +1 | 166, +0.6 pp, 1 / 0 | 0 / 0 / 1 / 0 | 4.1 |
| `C_L31_x-rand-p0-5` | x | 23 | 78 / 88 / 3 / 23 | 166 | 47.0% | +0 [-4, +4] | +0 | 166, +0.0 pp, 2 / 2 | 0 / 0 / 0 / 0 | 4.1 |
| `C_L31_x-w-m0-25` | x | 19 | 82 / 87 / 4 / 19 | 169 | 48.5% | +4 [+0, +9] | +3 | 166, +1.2 pp, 3 / 1 | 0 / 0 / 2 / 1 | 4.5 |
| `C_L31_x-w-m0-5` | x | 13 | 85 / 90 / 4 / 13 | 175 | 48.6% | +7 [+2, +13] | +9 | 166, +1.2 pp, 3 / 1 | 0 / 0 / 5 / 4 | 4.5 |
| `C_L31_x-w-m1` | x | 7 | 86 / 96 / 3 / 7 | 182 | 47.3% | +8 [-1, +17] | +16 | 166, -0.6 pp, 5 / 6 | 0 / 0 / 9 / 7 | 4.3 |
| `C_L31_x-w-m2` | x | 2 | 90 / 98 / 2 / 2 | 188 | 47.9% | +12 [+0, +23] | +22 | 166, -1.8 pp, 8 / 11 | 0 / 0 / 15 / 7 | 5.0 |
| `X_x--E11-rand-m1-and-C31-w-m0-5` | x | 13 | 94 / 84 / 1 / 13 | 178 | 52.8% | +16 [+8, +24] | +12 | 166, +4.2 pp, 8 / 1 | 0 / 0 / 9 / 3 | 4.7 |
| `X_x--E11-rand-m2-and-C31-w-m0-5` | x | 16 | 92 / 82 / 2 / 16 | 174 | 52.9% | +14 [+5, +23] | +8 | 165, +4.2 pp, 10 / 3 | 1 / 0 / 8 / 1 | 4.7 |
| `X_x--E11-rand-p1-and-C31-w-m0-5` | x | 13 | 86 / 90 / 3 / 13 | 176 | 48.9% | +8 [+0, +16] | +10 | 166, +0.6 pp, 5 / 4 | 0 / 0 / 7 / 3 | 4.2 |
| `X_x--E11-rand-p2-and-C31-w-m0-5` | x | 12 | 84 / 93 / 3 / 12 | 177 | 47.5% | +6 [-3, +15] | +11 | 166, -0.6 pp, 6 / 7 | 0 / 0 / 7 / 4 | 4.3 |
| `X_x--E11-w-p1-and-C31-rand-m0-5` | x | 52 | 84 / 56 / 0 / 52 | 140 | 60.0% | +6 [-6, +18] | -26 | 136, +9.6 pp, 21 / 8 | 10 / 20 / 3 / 1 | 4.6 |
| `X_x--E11-w-p1-and-C31-rand-p0-5` | x | 53 | 87 / 52 / 0 / 53 | 139 | 62.6% | +9 [-3, +21] | -27 | 135, +12.6 pp, 23 / 6 | 10 / 21 / 2 / 2 | 4.7 |
| `X_x--E11-w-p1-and-C31-w-m0-25` | x | 36 | 92 / 64 / 0 / 36 | 156 | 59.0% | +14 [+2, +26] | -10 | 151, +9.3 pp, 24 / 10 | 3 / 12 / 3 / 2 | 4.9 |
| `X_x--E11-w-p1-and-C31-w-m0-5` (chosen k) | x | 22 | 101 / 69 / 0 / 22 | 170 | 59.4% | +23 [+10, +36] | +4 | 159, +9.4 pp, 26 / 11 | 2 / 5 / 10 / 1 | 5.4 |
| `X_x--E11-w-p1-and-C31-w-m1` | x | 5 | 115 / 69 / 3 / 5 | 184 | 62.5% | +37 [+22, +51] | +18 | 165, +12.1 pp, 31 / 11 | 1 / 0 / 18 / 1 | 6.1 |
| `X_x--E11-w-p1-and-C31-w-m2` | x | 2 | 116 / 72 / 2 / 2 | 188 | 61.7% | +38 [+23, +52] | +22 | 166, +10.2 pp, 30 / 13 | 0 / 0 / 21 / 1 | 6.7 |
| `X_x--E11-w-p2-and-C31-rand-m0-5` | x | 76 | 83 / 31 / 2 / 76 | 114 | 72.8% | +5 [-11, +21] | -52 | 107, +18.7 pp, 30 / 10 | 20 / 39 / 5 / 2 | 5.1 |
| `X_x--E11-w-p2-and-C31-rand-p0-5` | x | 77 | 80 / 33 / 2 / 77 | 113 | 70.8% | +2 [-13, +17] | -53 | 106, +15.1 pp, 27 / 11 | 19 / 41 / 5 / 2 | 4.9 |
| `X_x--E11-w-p2-and-C31-w-m0-25` | x | 39 | 108 / 40 / 5 / 39 | 148 | 73.0% | +30 [+13, +47] | -18 | 134, +23.1 pp, 43 / 12 | 13 / 19 / 12 / 2 | 6.2 |
| `X_x--E11-w-p2-and-C31-w-m0-5` (chosen k) | x | 19 | 126 / 44 / 3 / 19 | 170 | 74.1% | +48 [+30, +65] | +4 | 148, +24.3 pp, 51 / 15 | 8 / 10 / 20 / 2 | 7.7 |
| `X_x--E11-w-p2-and-C31-w-m1` | x | 2 | 142 / 45 / 3 / 2 | 187 | 75.9% | +64 [+47, +81] | +21 | 161, +26.1 pp, 57 / 15 | 2 / 3 / 24 / 2 | 8.8 |
| `X_x--E11-w-p2-and-C31-w-m2` | x | 3 | 152 / 35 / 2 / 3 | 187 | 81.3% | +74 [+58, +90] | +21 | 162, +32.7 pp, 63 / 10 | 1 / 3 / 22 / 3 | 10.5 |

R / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.

## The questions E alone turns into "I don't know" (rule abstention; answered without any write)

- E +1: 31 questions. Without the write they were R / W / H / X = 11 / 20 / 0 / 0 (right 35% of R + W); the questions answered both times were 67 / 68 / 3 / 0 (right 50%).
  With E +1 and C -0.5 (toward confident), 24 of them are no longer rule abstentions: R / W / H / X = 11 / 13 / 0 / 0.
- E +2: 59 questions. Without the write they were R / W / H / X = 20 / 38 / 1 / 0 (right 34% of R + W); the questions answered both times were 58 / 50 / 2 / 0 (right 54%).
  With E +2 and C -0.5 (toward confident), 44 of them are no longer rule abstentions: R / W / H / X = 31 / 12 / 1 / 0.

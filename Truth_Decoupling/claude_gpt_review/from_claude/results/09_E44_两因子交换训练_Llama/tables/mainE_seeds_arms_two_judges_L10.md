# seed 1051
# judge_tune_proA_E_L10, judge allenai, 192 questions (each arm against its own base prompt, no write)

| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|
| prompt conf | — | 88 / 101 / 1 / 2 | 189 | 46.6% | — | — | — | — | 3.4 |
| prompt conf_E1 | — | 108 / 83 / 1 / 0 | 191 | 56.5% | — | — | — | — | 4.4 |
| L10 conf.md.p2 | +2 | 82 / 107 / 1 / 2 | 189 | 43.4% | -6 [-14, +2] | +0 | 188, -2.7 pp, 6 / 11 | 1 / 0 / 0 / 1 | 3.3 |
| L10 conf.rand.m2 | -2 | 89 / 100 / 2 / 1 | 189 | 47.1% | +1 [-5, +7] | +0 | 188, +0.5 pp, 5 / 4 | 1 / 0 / 1 / 0 | 3.2 |
| L10 conf.rand.p2 | +2 | 87 / 102 / 1 / 2 | 189 | 46.0% | -1 [-8, +7] | +0 | 188, +0.0 pp, 7 / 7 | 1 / 0 / 0 / 1 | 3.7 |
| L10 conf.w.p1 | +1 | 102 / 88 / 1 / 1 | 190 | 53.7% | +14 [+5, +23] | +1 | 188, +7.4 pp, 18 / 4 | 1 / 0 / 1 / 1 | 4.1 |
| L10 conf.w.p2 | +2 | 127 / 65 / 0 / 0 | 192 | 66.1% | +39 [+25, +52] | +3 | 189, +19.0 pp, 44 / 8 | 0 / 0 / 3 / 0 | 5.2 |
| L10 conf_E1.w.m2 | -2 | 80 / 111 / 0 / 1 | 191 | 41.9% | -28 [-39, -17] | +0 | 190, -14.2 pp, 5 / 32 | 1 / 0 / 0 / 1 | 3.2 |

# seed 1051
# judge_tune_proA_E_L10, judge pro, 192 questions (each arm against its own base prompt, no write)

| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|
| prompt conf | — | 70 / 112 / 9 / 1 | 182 | 38.5% | — | — | — | — | 3.4 |
| prompt conf_E1 | — | 91 / 92 / 7 / 2 | 183 | 49.7% | — | — | — | — | 4.4 |
| L10 conf.md.p2 | +2 | 69 / 111 / 9 / 3 | 180 | 38.3% | -1 [-8, +5] | -2 | 180, +0.0 pp, 5 / 5 | 1 / 1 / 0 / 0 | 3.3 |
| L10 conf.rand.m2 | -2 | 69 / 112 / 10 / 1 | 181 | 38.1% | -1 [-8, +5] | -1 | 180, -0.6 pp, 4 / 5 | 1 / 1 / 1 / 0 | 3.2 |
| L10 conf.rand.p2 | +2 | 69 / 113 / 9 / 1 | 182 | 37.9% | -1 [-8, +7] | +0 | 181, -0.6 pp, 6 / 7 | 1 / 0 / 1 / 0 | 3.7 |
| L10 conf.w.p1 | +1 | 83 / 97 / 8 / 4 | 180 | 46.1% | +13 [+3, +23] | -2 | 177, +7.3 pp, 17 / 4 | 3 / 2 / 3 / 0 | 4.1 |
| L10 conf.w.p2 | +2 | 105 / 77 / 6 / 4 | 182 | 57.7% | +35 [+21, +49] | +0 | 177, +18.1 pp, 41 / 9 | 2 / 3 / 5 / 0 | 5.2 |
| L10 conf_E1.w.m2 | -2 | 60 / 124 / 7 / 1 | 184 | 32.6% | -31 [-44, -18] | +1 | 179, -15.6 pp, 8 / 36 | 4 / 0 / 1 / 4 | 3.2 |

# seed 2045
# judge_tune_proA_E_L10, judge allenai, 192 questions (each arm against its own base prompt, no write)

| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|
| prompt conf | — | 88 / 101 / 1 / 2 | 189 | 46.6% | — | — | — | — | 3.4 |
| prompt conf_E1 | — | 108 / 83 / 1 / 0 | 191 | 56.5% | — | — | — | — | 4.4 |
| L10 conf.md.p2 | +2 | 82 / 107 / 1 / 2 | 189 | 43.4% | -6 [-14, +2] | +0 | 188, -2.7 pp, 6 / 11 | 1 / 0 / 0 / 1 | 3.3 |
| L10 conf.rand.m2 | -2 | 89 / 101 / 1 / 1 | 190 | 46.8% | +1 [-5, +7] | +1 | 189, +0.0 pp, 5 / 5 | 0 / 0 / 1 / 0 | 3.2 |
| L10 conf.rand.p2 | +2 | 86 / 103 / 1 / 2 | 189 | 45.5% | -2 [-9, +5] | +0 | 188, -0.5 pp, 6 / 7 | 1 / 0 / 0 / 1 | 3.7 |
| L10 conf.w.p1 | +1 | 100 / 90 / 1 / 1 | 190 | 52.6% | +12 [+3, +22] | +1 | 188, +6.4 pp, 17 / 5 | 1 / 0 / 1 / 1 | 4.1 |
| L10 conf.w.p2 | +2 | 127 / 65 / 0 / 0 | 192 | 66.1% | +39 [+26, +52] | +3 | 189, +19.0 pp, 43 / 7 | 0 / 0 / 3 / 0 | 5.1 |
| L10 conf_E1.w.m2 | -2 | 80 / 111 / 0 / 1 | 191 | 41.9% | -28 [-40, -16] | +0 | 190, -14.7 pp, 5 / 33 | 1 / 0 / 1 / 0 | 3.2 |

# seed 2045
# judge_tune_proA_E_L10, judge pro, 192 questions (each arm against its own base prompt, no write)

| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|
| prompt conf | — | 70 / 112 / 9 / 1 | 182 | 38.5% | — | — | — | — | 3.4 |
| prompt conf_E1 | — | 91 / 92 / 7 / 2 | 183 | 49.7% | — | — | — | — | 4.4 |
| L10 conf.md.p2 | +2 | 69 / 111 / 9 / 3 | 180 | 38.3% | -1 [-8, +5] | -2 | 180, +0.0 pp, 5 / 5 | 1 / 1 / 0 / 0 | 3.3 |
| L10 conf.rand.m2 | -2 | 71 / 111 / 9 / 1 | 182 | 39.0% | +1 [-5, +7] | +0 | 181, +0.0 pp, 4 / 4 | 0 / 1 / 1 / 0 | 3.2 |
| L10 conf.rand.p2 | +2 | 68 / 114 / 9 / 1 | 182 | 37.4% | -2 [-9, +5] | +0 | 181, -1.1 pp, 5 / 7 | 1 / 0 / 1 / 0 | 3.7 |
| L10 conf.w.p1 | +1 | 83 / 97 / 8 / 4 | 180 | 46.1% | +13 [+3, +23] | -2 | 177, +7.3 pp, 18 / 5 | 3 / 2 / 3 / 0 | 4.1 |
| L10 conf.w.p2 | +2 | 106 / 76 / 6 / 4 | 182 | 58.2% | +36 [+23, +50] | +0 | 177, +18.6 pp, 41 / 8 | 2 / 3 / 5 / 0 | 5.1 |
| L10 conf_E1.w.m2 | -2 | 61 / 123 / 7 / 1 | 184 | 33.2% | -30 [-43, -17] | +1 | 179, -14.5 pp, 9 / 35 | 4 / 0 / 0 / 5 | 3.2 |


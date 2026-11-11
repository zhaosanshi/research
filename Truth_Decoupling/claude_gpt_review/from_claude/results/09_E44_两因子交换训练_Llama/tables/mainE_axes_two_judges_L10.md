# axis 1051
# judge_tune_proA_E_L10, judge allenai, 192 questions (each arm against its own base prompt, no write)

| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|
| prompt conf | — | 88 / 101 / 1 / 2 | 189 | 46.6% | — | — | — | — | 3.4 |
| L10 conf.rand.m2 | -2 | 85 / 103 / 2 / 2 | 188 | 45.2% | -3 [-10, +3] | -1 | 188, -1.1 pp, 4 / 6 | 1 / 0 / 0 / 0 | 3.6 |
| L10 conf.rand.p2 | +2 | 85 / 106 / 1 / 0 | 191 | 44.5% | -3 [-10, +3] | +2 | 188, -1.6 pp, 3 / 6 | 1 / 0 / 1 / 2 | 3.2 |
| L10 conf.w.p2 | +2 | 127 / 65 / 0 / 0 | 192 | 66.1% | +39 [+25, +52] | +3 | 189, +19.0 pp, 44 / 8 | 0 / 0 / 3 / 0 | 5.3 |

# axis 1051
# judge_tune_proA_E_L10, judge pro, 192 questions (each arm against its own base prompt, no write)

| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|
| prompt conf | — | 70 / 112 / 9 / 1 | 182 | 38.5% | — | — | — | — | 3.4 |
| L10 conf.rand.m2 | -2 | 71 / 110 / 10 / 1 | 181 | 39.2% | +1 [-4, +6] | -1 | 181, +1.1 pp, 4 / 2 | 1 / 0 / 0 / 0 | 3.6 |
| L10 conf.rand.p2 | +2 | 64 / 118 / 9 / 1 | 182 | 35.2% | -6 [-13, +1] | +0 | 179, -3.4 pp, 2 / 8 | 2 / 1 / 2 / 1 | 3.2 |
| L10 conf.w.p2 | +2 | 106 / 76 / 6 / 4 | 182 | 58.2% | +36 [+23, +49] | +0 | 177, +18.6 pp, 40 / 7 | 2 / 3 / 5 / 0 | 5.3 |

# axis 2045
# judge_tune_proA_E_L10, judge allenai, 192 questions (each arm against its own base prompt, no write)

| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|
| prompt conf | — | 88 / 101 / 1 / 2 | 189 | 46.6% | — | — | — | — | 3.4 |
| L10 conf.rand.m2 | -2 | 85 / 106 / 0 / 1 | 191 | 44.5% | -3 [-8, +1] | +2 | 189, -1.6 pp, 1 / 4 | 0 / 0 / 0 / 2 | 3.3 |
| L10 conf.rand.p2 | +2 | 85 / 104 / 1 / 2 | 189 | 45.0% | -3 [-10, +4] | +0 | 188, -1.1 pp, 5 / 7 | 1 / 0 / 0 / 1 | 3.5 |
| L10 conf.w.p2 | +2 | 127 / 65 / 0 / 0 | 192 | 66.1% | +39 [+25, +52] | +3 | 189, +19.0 pp, 44 / 8 | 0 / 0 / 3 / 0 | 5.3 |

# axis 2045
# judge_tune_proA_E_L10, judge pro, 192 questions (each arm against its own base prompt, no write)

| condition | strength | R / W / H / X | answered | right among answered | dR [95%] | dAns | keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |
|---|---|---|---|---|---|---|---|---|---|
| prompt conf | — | 70 / 112 / 9 / 1 | 182 | 38.5% | — | — | — | — | 3.4 |
| L10 conf.rand.m2 | -2 | 69 / 113 / 8 / 2 | 182 | 37.9% | -1 [-6, +3] | +0 | 181, -0.6 pp, 2 / 3 | 0 / 1 / 0 / 1 | 3.3 |
| L10 conf.rand.p2 | +2 | 69 / 110 / 12 / 1 | 179 | 38.5% | -1 [-8, +6] | -3 | 178, -0.6 pp, 5 / 6 | 1 / 3 / 1 / 0 | 3.5 |
| L10 conf.w.p2 | +2 | 106 / 76 / 6 / 4 | 182 | 58.2% | +36 [+23, +49] | +0 | 177, +18.6 pp, 40 / 7 | 2 / 3 / 5 / 0 | 5.3 |


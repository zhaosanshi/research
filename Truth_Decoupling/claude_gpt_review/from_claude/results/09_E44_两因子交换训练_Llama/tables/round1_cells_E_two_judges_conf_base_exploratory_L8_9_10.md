# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 98 / 92 / 1 / 1 | 3.9 | +10 [+2, +19] | +1 [-2, +5] | -1 [-3, +0] | 188, +10 [+2, +18], 14 / 4 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +6 (165) |
| conf.md.p1 | 89 / 101 / 0 / 2 | 3.5 | +1 [-3, +5] | +1 [+0, +3] | +0 [+0, +0] | 189, +1 [-3, +5], 3 / 2 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +2 (184) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 3.4 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 189, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (185) |
| conf.rand.m1 | 85 / 105 / 0 / 2 | 3.4 | -3 [-8, +1] | +1 [+0, +3] | +0 [+0, +0] | 189, -3 [-8, +1], 1 / 4 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -2 (187) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +10 [+1, +19]; vs conf.rand.m1 +13 [+4, +22] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-1.0, +2.6]; moved out / in 1 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +6 on 165 questions; mean words 3.4 -> 3.9
KEY family=E judge=allenai layer=8 A=incomplete B=incomplete dR=+10 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+10 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met


Other arms of the same layer and judge (arm | base R / W / H / X | arm R / W / H / X | dR | dAns | dX | keep: n, dR, W>R / R>W | out / in / toH | G_same (n)):

| caut.wq.p1 | 82 / 82 / 1 / 27 | 84 / 73 / 3 / 32 | +2 [-5, +9] | -7 [-12, -2] | +5 [+1, +10] | 157, +3, 7 / 4 | 7 / 0 / 2 | +4 (138) |
| caut_E1.wq.m1 | 90 / 68 / 2 / 32 | 82 / 82 / 1 / 27 | -8 [-15, -2] | +6 [+2, +11] | -5 [-10, -1] | 158, -8, 2 / 10 | 0 / 6 / 0 | -8 (155) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 89 / 100 / 1 / 2 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 189, +1, 1 / 0 | 0 / 0 / 0 | +1 (188) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 89 / 101 / 0 / 2 | +1 [-3, +5] | +1 [+0, +3] | +0 [+0, +0] | 189, +1, 3 / 2 | 0 / 1 / 0 | +2 (184) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 85 / 105 / 0 / 2 | -3 [-8, +1] | +1 [+0, +3] | +0 [+0, +0] | 189, -3, 1 / 4 | 0 / 1 / 0 | -2 (187) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 88 / 101 / 1 / 2 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 189, +0, 2 / 2 | 0 / 0 / 0 | -1 (185) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 98 / 92 / 1 / 1 | +10 [+2, +19] | +1 [-2, +5] | -1 [-3, +0] | 188, +10, 14 / 4 | 1 / 2 / 1 | +6 (165) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 98 / 92 / 1 / 1 | +10 [+1, +19] | +1 [-2, +5] | -1 [-3, +0] | 188, +10, 15 / 5 | 1 / 2 / 1 | +6 (163) |
| conf_E1.wq.m1 | 108 / 83 / 1 / 0 | 90 / 101 / 0 / 1 | -18 [-27, -9] | +0 [-3, +3] | +1 [+0, +3] | 190, -17, 2 / 19 | 1 / 1 / 0 | -16 (184) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 80 / 99 / 11 / 2 | 3.9 | +10 [+0, +20] | -3 [-9, +3] | +1 [-2, +5] | 176, +10 [+1, +19], 15 / 5 | 6 / 3 / 5 | 3 / 3 / 3 / 0 | +6 (158) |
| conf.md.p1 | 71 / 111 / 9 / 1 | 3.5 | +1 [-5, +7] | +0 [-3, +3] | +0 [+0, +0] | 181, +0 [-5, +6], 4 / 4 | 1 / 1 / 1 | 0 / 1 / 1 / 0 | +0 (176) |
| conf.rand.p1 | 70 / 111 / 9 / 2 | 3.4 | +0 [-4, +4] | -1 [-5, +2] | +1 [+0, +3] | 180, -1 [-5, +2], 1 / 2 | 2 / 1 / 1 | 0 / 2 / 1 / 0 | -1 (176) |
| conf.rand.m1 | 70 / 113 / 8 / 1 | 3.4 | +0 [-4, +4] | +1 [+0, +3] | +0 [+0, +0] | 182, -1 [-4, +2], 1 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | -1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +10 [+0, +20]; vs conf.rand.m1 +10 [+0, +20] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.7, +1.6]; moved out / in 6 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 158 questions; mean words 3.4 -> 3.9
KEY family=E judge=pro layer=8 A=incomplete B=not_met dR=+10 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+10 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met


Other arms of the same layer and judge (arm | base R / W / H / X | arm R / W / H / X | dR | dAns | dX | keep: n, dR, W>R / R>W | out / in / toH | G_same (n)):

| caut.wq.p1 | 64 / 94 / 6 / 28 | 65 / 87 / 7 / 33 | +1 [-7, +9] | -6 [-12, +0] | +5 [+1, +10] | 150, +1, 7 / 6 | 8 / 2 / 3 | +4 (132) |
| caut_E1.wq.m1 | 76 / 80 / 4 / 32 | 64 / 95 / 6 / 27 | -12 [-21, -4] | +3 [-3, +10] | -5 [-10, -1] | 152, -11, 3 / 14 | 4 / 7 / 4 | -11 (149) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 72 / 109 / 9 / 2 | +2 [+0, +5] | -1 [-3, +0] | +1 [+0, +3] | 181, +2, 2 / 0 | 1 / 0 / 0 | +2 (180) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 71 / 111 / 9 / 1 | +1 [-5, +7] | +0 [-3, +3] | +0 [+0, +0] | 181, +0, 4 / 4 | 1 / 1 / 1 | +0 (176) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 70 / 113 / 8 / 1 | +0 [-4, +4] | +1 [+0, +3] | +0 [+0, +0] | 182, -1, 1 / 2 | 0 / 1 / 0 | -1 (180) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 70 / 111 / 9 / 2 | +0 [-4, +4] | -1 [-5, +2] | +1 [+0, +3] | 180, -1, 1 / 2 | 2 / 1 / 1 | -1 (176) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 80 / 99 / 11 / 2 | +10 [+0, +20] | -3 [-9, +3] | +1 [-2, +5] | 176, +10, 15 / 5 | 6 / 3 / 5 | +6 (158) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 81 / 98 / 11 / 2 | +11 [+2, +21] | -3 [-8, +2] | +1 [-2, +5] | 177, +11, 16 / 5 | 5 / 2 / 4 | +6 (157) |
| conf_E1.wq.m1 | 91 / 92 / 7 / 2 | 80 / 104 / 6 / 2 | -11 [-21, -1] | +1 [-3, +5] | +0 [-3, +3] | 181, -11, 6 / 17 | 2 / 3 / 1 | -11 (175) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 90 / 1 / 1 | 4.1 | +12 [+4, +21] | +1 [-2, +5] | -1 [-3, +0] | 188, +12 [+4, +20], 15 / 3 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +8 (163) |
| conf.md.p1 | 89 / 101 / 0 / 2 | 3.6 | +1 [-3, +5] | +1 [+0, +3] | +0 [+0, +0] | 189, +1 [-3, +5], 3 / 2 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +2 (181) |
| conf.rand.p1 | 89 / 100 / 1 / 2 | 3.4 | +1 [-4, +6] | +0 [-3, +3] | +0 [+0, +0] | 188, +2 [-3, +7], 4 / 2 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +2 (186) |
| conf.rand.m1 | 82 / 107 / 1 / 2 | 3.5 | -6 [-12, -1] | +0 [+0, +0] | +0 [+0, +0] | 189, -6 [-12, -1], 1 / 7 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -5 (185) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +12 (6.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +11 [+3, +19]; vs conf.rand.m1 +18 [+9, +27] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-1.0, +2.6]; moved out / in 1 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +8 on 163 questions; mean words 3.4 -> 4.1
KEY family=E judge=allenai layer=9 A=incomplete B=incomplete dR=+12 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+11 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met


Other arms of the same layer and judge (arm | base R / W / H / X | arm R / W / H / X | dR | dAns | dX | keep: n, dR, W>R / R>W | out / in / toH | G_same (n)):

| caut.wq.p1 | 82 / 82 / 1 / 27 | 85 / 68 / 4 / 35 | +3 [-6, +12] | -11 [-18, -5] | +8 [+3, +14] | 153, +7, 11 / 4 | 11 / 0 / 3 | +7 (131) |
| caut_E1.wq.m1 | 90 / 68 / 2 / 32 | 80 / 86 / 1 / 25 | -10 [-18, -3] | +8 [+3, +14] | -7 [-13, -2] | 158, -11, 2 / 13 | 0 / 8 / 0 | -10 (154) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 88 / 101 / 1 / 2 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 189, +0, 1 / 1 | 0 / 0 / 0 | +0 (189) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 89 / 101 / 0 / 2 | +1 [-3, +5] | +1 [+0, +3] | +0 [+0, +0] | 189, +1, 3 / 2 | 0 / 1 / 0 | +2 (181) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 82 / 107 / 1 / 2 | -6 [-12, -1] | +0 [+0, +0] | +0 [+0, +0] | 189, -6, 1 / 7 | 0 / 0 / 0 | -5 (185) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 89 / 100 / 1 / 2 | +1 [-4, +6] | +0 [-3, +3] | +0 [+0, +0] | 188, +2, 4 / 2 | 1 / 1 / 1 | +2 (186) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 100 / 90 / 1 / 1 | +12 [+4, +21] | +1 [-2, +5] | -1 [-3, +0] | 188, +12, 15 / 3 | 1 / 2 / 1 | +8 (163) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 100 / 90 / 1 / 1 | +12 [+3, +21] | +1 [-2, +5] | -1 [-3, +0] | 188, +12, 16 / 4 | 1 / 2 / 1 | +8 (164) |
| conf_E1.wq.m1 | 108 / 83 / 1 / 0 | 86 / 106 / 0 / 0 | -22 [-32, -12] | +1 [+0, +3] | +0 [+0, +0] | 191, -22, 4 / 26 | 0 / 1 / 0 | -21 (188) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 84 / 95 / 11 / 2 | 4.1 | +14 [+5, +24] | -3 [-9, +3] | +1 [-2, +5] | 176, +13 [+5, +22], 17 / 4 | 6 / 3 / 5 | 2 / 4 / 3 / 0 | +8 (155) |
| conf.md.p1 | 67 / 114 / 10 / 1 | 3.6 | -3 [-9, +3] | -1 [-3, +0] | +0 [+0, +0] | 181, -3 [-9, +3], 3 / 6 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -4 (174) |
| conf.rand.p1 | 68 / 113 / 9 / 2 | 3.4 | -2 [-7, +3] | -1 [-5, +2] | +1 [+0, +3] | 180, -1 [-5, +3], 2 / 3 | 2 / 1 / 1 | 1 / 1 / 0 / 1 | -1 (178) |
| conf.rand.m1 | 73 / 110 / 8 / 1 | 3.5 | +3 [-2, +8] | +1 [+0, +3] | +0 [+0, +0] | 182, +2 [-3, +7], 4 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +14 (7.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +16 [+7, +25]; vs conf.rand.m1 +11 [+1, +21] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.7, +1.6]; moved out / in 6 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +8 on 155 questions; mean words 3.4 -> 4.1
KEY family=E judge=pro layer=9 A=incomplete B=not_met dR=+14 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+11 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met


Other arms of the same layer and judge (arm | base R / W / H / X | arm R / W / H / X | dR | dAns | dX | keep: n, dR, W>R / R>W | out / in / toH | G_same (n)):

| caut.wq.p1 | 64 / 94 / 6 / 28 | 63 / 88 / 6 / 35 | -1 [-9, +7] | -7 [-14, -1] | +7 [+2, +12] | 149, +1, 6 / 5 | 9 / 2 / 2 | +4 (126) |
| caut_E1.wq.m1 | 76 / 80 / 4 / 32 | 61 / 100 / 6 / 25 | -15 [-24, -6] | +5 [-2, +12] | -7 [-13, -2] | 152, -14, 3 / 17 | 4 / 9 / 4 | -14 (148) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 71 / 110 / 9 / 2 | +1 [-2, +5] | -1 [-3, +0] | +1 [+0, +3] | 181, +1, 2 / 1 | 1 / 0 / 0 | +1 (181) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 67 / 114 / 10 / 1 | -3 [-9, +3] | -1 [-3, +0] | +0 [+0, +0] | 181, -3, 3 / 6 | 1 / 0 / 1 | -4 (174) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 73 / 110 / 8 / 1 | +3 [-2, +8] | +1 [+0, +3] | +0 [+0, +0] | 182, +2, 4 / 2 | 0 / 1 / 0 | +2 (178) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 68 / 113 / 9 / 2 | -2 [-7, +3] | -1 [-5, +2] | +1 [+0, +3] | 180, -1, 2 / 3 | 2 / 1 / 1 | -1 (178) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 84 / 95 / 11 / 2 | +14 [+5, +24] | -3 [-9, +3] | +1 [-2, +5] | 176, +13, 17 / 4 | 6 / 3 / 5 | +8 (155) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 82 / 96 / 11 / 3 | +12 [+3, +22] | -4 [-10, +1] | +2 [-2, +6] | 176, +12, 17 / 5 | 6 / 2 / 5 | +6 (156) |
| conf_E1.wq.m1 | 91 / 92 / 7 / 2 | 74 / 111 / 6 / 1 | -17 [-27, -7] | +2 [-2, +6] | -1 [-3, +0] | 182, -17, 5 / 22 | 1 / 3 / 1 | -17 (179) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 102 / 88 / 1 / 1 | 4.1 | +14 [+5, +23] | +1 [-2, +5] | -1 [-3, +0] | 188, +14 [+5, +23], 18 / 4 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +5 (160) |
| conf.md.p1 | 86 / 103 / 1 / 2 | 3.4 | -2 [-9, +5] | +0 [-3, +3] | +0 [+0, +0] | 188, -1 [-7, +5], 5 / 6 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (183) |
| conf.rand.p1 | 84 / 106 / 0 / 2 | 3.5 | -4 [-10, +1] | +1 [+0, +3] | +0 [+0, +0] | 189, -4 [-10, +1], 2 / 6 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -2 (182) |
| conf.rand.m1 | 90 / 98 / 2 / 2 | 3.3 | +2 [-3, +7] | -1 [-3, +0] | +0 [+0, +0] | 188, +3 [-1, +8], 4 / 1 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +2 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +14 (7.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +18 [+9, +28]; vs conf.rand.m1 +12 [+3, +21] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-1.0, +2.6]; moved out / in 1 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +5 on 160 questions; mean words 3.4 -> 4.1
KEY family=E judge=allenai layer=10 A=incomplete B=incomplete dR=+14 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+12 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met


Other arms of the same layer and judge (arm | base R / W / H / X | arm R / W / H / X | dR | dAns | dX | keep: n, dR, W>R / R>W | out / in / toH | G_same (n)):

| caut.wq.p1 | 82 / 82 / 1 / 27 | 88 / 68 / 2 / 34 | +6 [-3, +15] | -8 [-14, -2] | +7 [+2, +13] | 155, +8, 12 / 4 | 9 / 1 / 1 | +4 (121) |
| caut_E1.wq.m1 | 90 / 68 / 2 / 32 | 77 / 87 / 2 / 26 | -13 [-22, -5] | +6 [+2, +11] | -6 [-11, -2] | 158, -14, 2 / 16 | 0 / 6 / 0 | -13 (154) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 85 / 104 / 1 / 2 | -3 [-7, +0] | +0 [+0, +0] | +0 [+0, +0] | 189, -3, 0 / 3 | 0 / 0 / 0 | -2 (187) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 86 / 103 / 1 / 2 | -2 [-9, +5] | +0 [-3, +3] | +0 [+0, +0] | 188, -1, 5 / 6 | 1 / 1 / 1 | +0 (183) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 90 / 98 / 2 / 2 | +2 [-3, +7] | -1 [-3, +0] | +0 [+0, +0] | 188, +3, 4 / 1 | 1 / 0 / 1 | +2 (184) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 84 / 106 / 0 / 2 | -4 [-10, +1] | +1 [+0, +3] | +0 [+0, +0] | 189, -4, 2 / 6 | 0 / 1 / 0 | -2 (182) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 102 / 88 / 1 / 1 | +14 [+5, +23] | +1 [-2, +5] | -1 [-3, +0] | 188, +14, 18 / 4 | 1 / 2 / 1 | +5 (160) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 104 / 86 / 1 / 1 | +16 [+7, +25] | +1 [-2, +5] | -1 [-3, +0] | 188, +16, 19 / 3 | 1 / 2 / 1 | +7 (157) |
| conf_E1.wq.m1 | 108 / 83 / 1 / 0 | 86 / 104 / 0 / 2 | -22 [-32, -12] | -1 [-5, +2] | +2 [+0, +5] | 189, -21, 3 / 24 | 2 / 1 / 0 | -21 (184) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 83 / 97 / 8 / 4 | 4.1 | +13 [+3, +23] | -2 [-8, +4] | +3 [-1, +8] | 177, +13 [+4, +22], 17 / 4 | 5 / 3 / 3 | 3 / 2 / 3 / 0 | +4 (153) |
| conf.md.p1 | 69 / 111 / 10 / 2 | 3.4 | -1 [-8, +5] | -2 [-5, +0] | +1 [+0, +3] | 180, +0 [-6, +6], 5 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +0 (175) |
| conf.rand.p1 | 71 / 112 / 8 / 1 | 3.5 | +1 [-5, +7] | +1 [+0, +3] | +0 [+0, +0] | 182, +0 [-5, +6], 4 / 4 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (175) |
| conf.rand.m1 | 68 / 112 / 10 / 2 | 3.3 | -2 [-7, +3] | -2 [-5, +0] | +1 [+0, +3] | 180, -1 [-5, +3], 2 / 3 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +13 (6.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +12 [+1, +22]; vs conf.rand.m1 +15 [+5, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-4.2, +2.1]; moved out / in 5 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +4 on 153 questions; mean words 3.4 -> 4.1
KEY family=E judge=pro layer=10 A=incomplete B=not_met dR=+13 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+12 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met


Other arms of the same layer and judge (arm | base R / W / H / X | arm R / W / H / X | dR | dAns | dX | keep: n, dR, W>R / R>W | out / in / toH | G_same (n)):

| caut.wq.p1 | 64 / 94 / 6 / 28 | 67 / 84 / 6 / 35 | +3 [-6, +12] | -7 [-14, +0] | +7 [+2, +12] | 148, +4, 9 / 5 | 10 / 3 / 3 | +4 (117) |
| caut_E1.wq.m1 | 76 / 80 / 4 / 32 | 63 / 97 / 6 / 26 | -13 [-22, -4] | +4 [-3, +11] | -6 [-11, -2] | 152, -12, 3 / 15 | 4 / 8 / 4 | -11 (148) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 69 / 113 / 8 / 2 | -1 [-4, +2] | +0 [-3, +3] | +1 [+0, +3] | 181, -2, 0 / 2 | 1 / 1 / 0 | -2 (179) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 69 / 111 / 10 / 2 | -1 [-8, +5] | -2 [-5, +0] | +1 [+0, +3] | 180, +0, 5 / 5 | 2 / 0 / 1 | +0 (175) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 68 / 112 / 10 / 2 | -2 [-7, +3] | -2 [-5, +0] | +1 [+0, +3] | 180, -1, 2 / 3 | 2 / 0 / 1 | -1 (177) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 71 / 112 / 8 / 1 | +1 [-5, +7] | +1 [+0, +3] | +0 [+0, +0] | 182, +0, 4 / 4 | 0 / 1 / 0 | +1 (175) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 83 / 97 / 8 / 4 | +13 [+3, +23] | -2 [-8, +4] | +3 [-1, +8] | 177, +13, 17 / 4 | 5 / 3 / 3 | +4 (153) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 86 / 95 / 8 / 3 | +16 [+6, +26] | -1 [-6, +4] | +2 [-2, +6] | 178, +15, 19 / 4 | 4 / 3 / 3 | +6 (152) |
| conf_E1.wq.m1 | 91 / 92 / 7 / 2 | 75 / 108 / 7 / 2 | -16 [-26, -6] | +0 [-5, +5] | +0 [-3, +3] | 180, -15, 6 / 21 | 3 / 3 / 2 | -15 (175) |


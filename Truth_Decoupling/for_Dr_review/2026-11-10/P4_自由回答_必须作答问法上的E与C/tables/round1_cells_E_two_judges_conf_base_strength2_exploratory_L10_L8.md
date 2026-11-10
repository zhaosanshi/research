# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 127 / 65 / 0 / 0 | 5.3 | +39 [+25, +52] | +3 [+0, +7] | -2 [-5, +0] | 189, +36 [+23, +49], 44 / 8 | 0 / 3 / 0 | 0 / 0 / 3 / 0 | +14 (118) |
| conf.md.p2 | 82 / 107 / 1 / 2 | 3.3 | -6 [-14, +2] | +0 [-3, +3] | +0 [+0, +0] | 188, -5 [-13, +3], 6 / 11 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | -4 (175) |
| conf.rand.p2 | 84 / 104 / 2 / 2 | 3.7 | -4 [-12, +4] | -1 [-5, +2] | +0 [+0, +0] | 187, -2 [-9, +5], 6 / 8 | 2 / 1 / 2 | 2 / 0 / 0 / 1 | +0 (175) |
| conf.rand.m2 | 88 / 101 / 1 / 2 | 3.2 | +0 [-6, +6] | +0 [+0, +0] | +0 [+0, +0] | 189, +0 [-6, +6], 5 / 5 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (188) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +39 (20.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +43 [+30, +56]; vs conf.rand.m2 +39 [+26, +52] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.6 pp [+0.0, +3.6]; moved out / in 0 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +14 on 118 questions; mean words 3.4 -> 5.3
KEY family=E judge=allenai layer=10 A=incomplete B=not_met dR=+39 dR_F=NA dAns=+3 dAns_F=NA rand_margin=+39 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 106 / 76 / 6 / 4 | 5.3 | +36 [+23, +49] | +0 [-6, +6] | +3 [-1, +8] | 177, +33 [+20, +46], 40 / 7 | 5 / 5 / 2 | 2 / 3 / 5 / 0 | +11 (114) |
| conf.md.p2 | 69 / 111 / 9 / 3 | 3.3 | -1 [-8, +5] | -2 [-5, +0] | +2 [+0, +5] | 180, +0 [-6, +6], 5 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (167) |
| conf.rand.p2 | 70 / 112 / 9 / 1 | 3.7 | +0 [-7, +7] | +0 [-3, +3] | +0 [+0, +0] | 181, +0 [-7, +7], 6 / 6 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +1 (169) |
| conf.rand.m2 | 69 / 112 / 9 / 2 | 3.2 | -1 [-7, +5] | -1 [-3, +0] | +1 [+0, +3] | 181, -1 [-7, +5], 4 / 5 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -1 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +36 (18.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +36 [+22, +50]; vs conf.rand.m2 +37 [+24, +50] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-3.1, +3.1]; moved out / in 5 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +11 on 114 questions; mean words 3.4 -> 5.3
KEY family=E judge=pro layer=10 A=incomplete B=not_met dR=+36 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+36 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met


# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 116 / 74 / 1 / 1 | 4.9 | +28 [+17, +39] | +1 [-2, +5] | -1 [-3, +0] | 188, +27 [+16, +38], 31 / 4 | 1 / 2 / 1 | 1 / 0 / 2 / 0 | +13 (133) |
| conf.md.p2 | 91 / 99 / 0 / 2 | 3.5 | +3 [-3, +9] | +1 [+0, +3] | +0 [+0, +0] | 189, +3 [-3, +9], 6 / 3 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +3 (182) |
| conf.rand.p2 | 85 / 104 / 1 / 2 | 3.3 | -3 [-10, +3] | +0 [+0, +0] | +0 [+0, +0] | 189, -3 [-10, +3], 4 / 7 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -2 (184) |
| conf.rand.m2 | 85 / 104 / 1 / 2 | 3.4 | -3 [-9, +3] | +0 [-3, +3] | +0 [+0, +0] | 188, -2 [-8, +3], 3 / 5 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | -1 (185) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +28 (14.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +31 [+20, +42]; vs conf.rand.m2 +31 [+20, +42] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-1.0, +2.6]; moved out / in 1 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +13 on 133 questions; mean words 3.4 -> 4.9
KEY family=E judge=allenai layer=8 A=incomplete B=incomplete dR=+28 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+31 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 96 / 83 / 8 / 5 | 4.9 | +26 [+15, +38] | -3 [-9, +3] | +4 [+0, +9] | 175, +25 [+15, +36], 28 / 3 | 7 / 4 / 4 | 3 / 4 / 4 / 0 | +12 (129) |
| conf.md.p2 | 71 / 112 / 8 / 1 | 3.5 | +1 [-6, +8] | +1 [-2, +5] | +0 [+0, +0] | 181, -1 [-8, +5], 5 / 6 | 1 / 2 / 1 | 0 / 1 / 2 / 0 | -2 (175) |
| conf.rand.p2 | 70 / 111 / 9 / 2 | 3.3 | +0 [-6, +5] | -1 [-5, +2] | +1 [+0, +3] | 180, -1 [-6, +4], 3 / 4 | 2 / 1 / 1 | 0 / 2 / 1 / 0 | -1 (175) |
| conf.rand.m2 | 70 / 112 / 9 / 1 | 3.4 | +0 [-6, +6] | +0 [-3, +3] | +0 [+0, +0] | 181, +0 [-5, +5], 3 / 3 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +0 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +26 (13.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +26 [+15, +38]; vs conf.rand.m2 +26 [+15, +38] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.7, +1.6]; moved out / in 7 / 4 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +12 on 129 questions; mean words 3.4 -> 4.9
KEY family=E judge=pro layer=8 A=incomplete B=not_met dR=+26 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+26 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met



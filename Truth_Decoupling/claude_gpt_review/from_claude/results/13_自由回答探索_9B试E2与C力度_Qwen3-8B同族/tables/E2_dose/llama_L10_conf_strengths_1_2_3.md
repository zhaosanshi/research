# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 118 / 73 / 0 / 1 | 4.1 | +30 [+18, +42] | +2 [+0, +5] | -1 [-3, +0] | 189, +28 [+16, +40], 34 / 6 | 0 / 2 / 0 | 0 / 0 / 2 / 0 | +15 (155) |
| conf.md.p1 | 88 / 102 / 0 / 2 | 3.5 | +0 [-7, +7] | +1 [+0, +3] | +0 [+0, +0] | 189, +0 [-7, +7], 6 / 6 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +2 (180) |
| conf.rand.p1 | 84 / 106 / 0 / 2 | 3.5 | -4 [-10, +1] | +1 [+0, +3] | +0 [+0, +0] | 189, -4 [-10, +1], 2 / 6 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -2 (181) |
| conf.rand.m1 | 90 / 98 / 2 / 2 | 3.3 | +2 [-3, +8] | -1 [-3, +0] | +0 [+0, +0] | 188, +3 [-2, +8], 5 / 2 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +2 (185) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +30 (15.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +34 [+22, +46]; vs conf.rand.m1 +28 [+15, +41] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.0 pp [+0.0, +2.6]; moved out / in 0 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +15 on 155 questions; mean words 3.4 -> 4.1
KEY family=E judge=allenai layer=10 A=incomplete B=incomplete dR=+30 dR_F=NA dAns=+2 dAns_F=NA rand_margin=+28 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 95 / 86 / 7 / 4 | 4.1 | +25 [+13, +37] | -1 [-7, +5] | +3 [-1, +8] | 177, +23 [+12, +34], 30 / 7 | 5 / 4 / 3 | 2 / 3 / 4 / 0 | +14 (148) |
| conf.md.p1 | 69 / 113 / 8 / 2 | 3.5 | -1 [-7, +5] | +0 [-3, +3] | +1 [+0, +3] | 181, -2 [-8, +3], 3 / 5 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | -2 (173) |
| conf.rand.p1 | 71 / 112 / 8 / 1 | 3.5 | +1 [-5, +7] | +1 [+0, +3] | +0 [+0, +0] | 182, +0 [-5, +6], 4 / 4 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (174) |
| conf.rand.m1 | 67 / 113 / 10 / 2 | 3.3 | -3 [-9, +3] | -2 [-5, +0] | +1 [+0, +3] | 180, -2 [-8, +3], 3 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -2 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +25 (13.0% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +24 [+11, +37]; vs conf.rand.m1 +28 [+16, +40] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-3.6, +2.6]; moved out / in 5 / 4 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +14 on 148 questions; mean words 3.4 -> 4.1
KEY family=E judge=pro layer=10 A=incomplete B=not_met dR=+25 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+24 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 139 / 50 / 3 / 0 | 4.7 | +51 [+36, +65] | +0 [-5, +5] | -2 [-5, +0] | 186, +48 [+34, +62], 57 / 9 | 3 / 3 / 3 | 0 / 3 / 3 / 0 | +18 (122) |
| conf.md.p2 | 85 / 106 / 1 / 0 | 3.5 | -3 [-11, +5] | +2 [-2, +6] | -2 [-5, +0] | 188, -2 [-10, +6], 7 / 9 | 1 / 3 / 1 | 1 / 0 / 0 / 3 | -3 (174) |
| conf.rand.p2 | 85 / 102 / 3 / 2 | 3.8 | -3 [-11, +5] | -2 [-6, +2] | +0 [+0, +0] | 186, +0 [-7, +7], 7 / 7 | 3 / 1 / 3 | 3 / 0 / 0 / 1 | +1 (172) |
| conf.rand.m2 | 90 / 100 / 1 / 1 | 3.2 | +2 [-5, +10] | +1 [+0, +3] | -1 [-3, +0] | 189, +1 [-6, +8], 7 / 6 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (187) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +51 (26.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +54 [+39, +69]; vs conf.rand.m2 +49 [+34, +64] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-2.6, +2.6]; moved out / in 3 / 3 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +18 on 122 questions; mean words 3.4 -> 4.7
KEY family=E judge=allenai layer=10 A=incomplete B=incomplete dR=+51 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+49 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 113 / 65 / 9 / 5 | 4.7 | +43 [+28, +58] | -4 [-12, +4] | +4 [+0, +9] | 171, +40 [+26, +54], 51 / 11 | 11 / 7 / 7 | 3 / 8 / 6 / 1 | +21 (117) |
| conf.md.p2 | 68 / 114 / 9 / 1 | 3.5 | -2 [-9, +5] | +0 [-6, +5] | +0 [-3, +3] | 178, -1 [-6, +4], 3 / 4 | 4 / 4 / 3 | 3 / 1 / 2 / 2 | -2 (168) |
| conf.rand.p2 | 70 / 111 / 9 / 2 | 3.8 | +0 [-8, +8] | -1 [-5, +2] | +1 [+0, +3] | 180, +1 [-6, +8], 7 / 6 | 2 / 1 / 1 | 2 / 0 / 1 / 0 | +0 (166) |
| conf.rand.m2 | 70 / 112 / 9 / 1 | 3.2 | +0 [-7, +7] | +0 [-3, +3] | +0 [-3, +3] | 181, -1 [-8, +6], 6 / 7 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | -1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +43 (22.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +43 [+27, +59]; vs conf.rand.m2 +43 [+28, +58] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.1 pp [-6.2, +2.1]; moved out / in 11 / 7 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +21 on 117 questions; mean words 3.4 -> 4.7
KEY family=E judge=pro layer=10 A=incomplete B=not_met dR=+43 dR_F=NA dAns=-4 dAns_F=NA rand_margin=+43 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 146 / 44 / 1 / 1 | 5.7 | +58 [+40, +75] | +1 [-3, +6] | -1 [-5, +2] | 187, +56 [+39, +73], 74 / 18 | 2 / 3 / 1 | 1 / 1 / 3 / 0 | +23 (88) |
| conf.md.p3 | 82 / 109 / 1 / 0 | 3.5 | -6 [-17, +4] | +2 [-2, +6] | -2 [-5, +0] | 188, -5 [-15, +5], 12 / 17 | 1 / 3 / 1 | 1 / 0 / 0 / 3 | -6 (165) |
| conf.rand.p3 | 87 / 101 / 2 / 2 | 3.7 | -1 [-10, +8] | -1 [-5, +2] | +0 [+0, +0] | 187, +1 [-8, +10], 11 / 10 | 2 / 1 / 2 | 2 / 0 / 0 / 1 | -1 (168) |
| conf.rand.m3 | 87 / 102 / 2 / 1 | 3.1 | -1 [-9, +7] | +0 [-3, +3] | -1 [-3, +0] | 188, +0 [-8, +8], 9 / 9 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | -1 (185) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +58 (30.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +59 [+42, +75]; vs conf.rand.m3 +59 [+41, +77] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-1.6, +3.1]; moved out / in 2 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +23 on 88 questions; mean words 3.4 -> 5.7
KEY family=E judge=allenai layer=10 A=incomplete B=not_met dR=+58 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+59 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 107 / 76 / 5 / 4 | 5.7 | +37 [+18, +55] | +1 [-7, +10] | +3 [-1, +8] | 173, +34 [+16, +52], 61 / 27 | 9 / 10 / 5 | 3 / 6 / 6 / 4 | +20 (82) |
| conf.md.p3 | 67 / 117 / 5 / 3 | 3.5 | -3 [-14, +8] | +2 [-5, +9] | +2 [-2, +6] | 177, -4 [-14, +6], 11 / 15 | 5 / 7 / 3 | 3 / 2 / 4 / 3 | -7 (162) |
| conf.rand.p3 | 68 / 113 / 9 / 2 | 3.7 | -2 [-12, +8] | -1 [-6, +3] | +1 [+0, +3] | 179, -1 [-9, +8], 9 / 10 | 3 / 2 / 3 | 3 / 0 / 2 / 0 | -4 (161) |
| conf.rand.m3 | 69 / 112 / 10 / 1 | 3.1 | -1 [-8, +6] | -1 [-6, +3] | +0 [-3, +3] | 179, +0 [-7, +7], 6 / 6 | 3 / 2 / 2 | 1 / 2 / 0 / 2 | +0 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +37 (19.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +39 [+20, +58]; vs conf.rand.m3 +38 [+20, +56] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-3.6, +5.2]; moved out / in 9 / 10 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +20 on 82 questions; mean words 3.4 -> 5.7
KEY family=E judge=pro layer=10 A=incomplete B=not_met dR=+37 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+38 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met


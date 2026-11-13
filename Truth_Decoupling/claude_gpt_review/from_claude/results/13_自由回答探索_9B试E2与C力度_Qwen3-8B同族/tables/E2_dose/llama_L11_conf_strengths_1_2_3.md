# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 117 / 72 / 2 / 1 | 4.0 | +29 [+17, +41] | +0 [-3, +3] | -1 [-3, +0] | 188, +29 [+17, +40], 35 / 6 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +19 (158) |
| conf.md.p1 | 88 / 102 / 0 / 2 | 3.4 | +0 [-7, +7] | +1 [+0, +3] | +0 [+0, +0] | 189, +0 [-7, +7], 6 / 6 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (180) |
| conf.rand.p1 | 81 / 108 / 1 / 2 | 3.4 | -7 [-12, -2] | +0 [+0, +0] | +0 [+0, +0] | 189, -7 [-12, -2], 0 / 7 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -6 (183) |
| conf.rand.m1 | 86 / 103 / 1 / 2 | 3.4 | -2 [-8, +4] | +0 [-3, +3] | +0 [+0, +0] | 188, -1 [-7, +5], 4 / 5 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (182) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +29 (15.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +36 [+23, +49]; vs conf.rand.m1 +31 [+20, +42] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.6, +1.6]; moved out / in 1 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +19 on 158 questions; mean words 3.4 -> 4.0
KEY family=E judge=allenai layer=11 A=incomplete B=incomplete dR=+29 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+31 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 99 / 81 / 8 / 4 | 4.0 | +29 [+17, +41] | -2 [-8, +3] | +3 [-1, +8] | 177, +29 [+18, +40], 33 / 4 | 5 / 3 / 3 | 3 / 2 / 3 / 0 | +17 (151) |
| conf.md.p1 | 73 / 108 / 10 / 1 | 3.4 | +3 [-4, +10] | -1 [-3, +0] | +0 [+0, +0] | 181, +4 [-3, +11], 8 / 4 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +1 (173) |
| conf.rand.p1 | 67 / 115 / 9 / 1 | 3.4 | -3 [-8, +1] | +0 [+0, +0] | +0 [+0, +0] | 182, -3 [-8, +1], 1 / 4 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -3 (176) |
| conf.rand.m1 | 72 / 108 / 10 / 2 | 3.4 | +2 [-4, +8] | -2 [-6, +2] | +1 [+0, +3] | 179, +2 [-3, +8], 5 / 3 | 3 / 1 / 2 | 1 / 2 / 1 / 0 | +1 (173) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +29 (15.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +32 [+20, +44]; vs conf.rand.m1 +27 [+15, +39] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-4.2, +1.6]; moved out / in 5 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +17 on 151 questions; mean words 3.4 -> 4.0
KEY family=E judge=pro layer=11 A=incomplete B=not_met dR=+29 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+27 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 132 / 55 / 5 / 0 | 4.8 | +44 [+28, +60] | -2 [-8, +3] | -2 [-5, +0] | 184, +41 [+25, +56], 57 / 16 | 5 / 3 / 5 | 0 / 5 / 3 / 0 | +20 (121) |
| conf.md.p2 | 82 / 109 / 0 / 1 | 3.3 | -6 [-15, +3] | +2 [+0, +5] | -1 [-3, +0] | 189, -6 [-15, +3], 7 / 13 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -4 (178) |
| conf.rand.p2 | 76 / 113 / 2 / 1 | 3.3 | -12 [-19, -5] | +0 [-3, +3] | -1 [-3, +0] | 188, -11 [-18, -5], 1 / 12 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | -9 (181) |
| conf.rand.m2 | 92 / 96 / 2 / 2 | 3.7 | +4 [-3, +12] | -1 [-5, +2] | +0 [+0, +0] | 187, +6 [-1, +14], 10 / 4 | 2 / 1 / 2 | 2 / 0 / 0 / 1 | +6 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +44 (22.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +56 [+40, +72]; vs conf.rand.m2 +40 [+24, +56] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-4.2, +1.6]; moved out / in 5 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +20 on 121 questions; mean words 3.4 -> 4.8
KEY family=E judge=allenai layer=11 A=incomplete B=not_met dR=+44 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+40 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 109 / 73 / 7 / 3 | 4.8 | +39 [+23, +55] | +0 [-7, +8] | +2 [-2, +6] | 175, +34 [+18, +49], 52 / 18 | 7 / 7 / 5 | 0 / 7 / 5 / 2 | +22 (119) |
| conf.md.p2 | 72 / 113 / 6 / 1 | 3.3 | +2 [-7, +11] | +3 [-1, +8] | +0 [-3, +3] | 181, +0 [-8, +8], 9 / 9 | 1 / 4 / 1 | 1 / 0 / 3 / 1 | -2 (172) |
| conf.rand.p2 | 66 / 114 / 11 / 1 | 3.3 | -4 [-10, +2] | -2 [-7, +3] | +0 [+0, +0] | 178, -3 [-8, +2], 2 / 5 | 4 / 2 / 4 | 2 / 2 / 1 / 1 | -4 (172) |
| conf.rand.m2 | 74 / 104 / 11 / 3 | 3.7 | +4 [-3, +11] | -4 [-9, +1] | +2 [+0, +5] | 177, +5 [-1, +12], 8 / 3 | 5 / 1 / 3 | 2 / 3 / 1 / 0 | +4 (169) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +39 (20.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +43 [+27, +59]; vs conf.rand.m2 +35 [+19, +51] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-3.6, +4.2]; moved out / in 7 / 7 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +22 on 119 questions; mean words 3.4 -> 4.8
KEY family=E judge=pro layer=11 A=incomplete B=not_met dR=+39 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+35 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 148 / 38 / 6 / 0 | 6.1 | +60 [+43, +77] | -3 [-9, +3] | -2 [-5, +0] | 183, +58 [+41, +75], 74 / 16 | 6 / 3 / 6 | 1 / 5 / 3 / 0 | +24 (85) |
| conf.md.p3 | 85 / 107 / 0 / 0 | 3.1 | -3 [-14, +8] | +3 [+0, +7] | -2 [-5, +0] | 189, -4 [-15, +7], 14 / 18 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | -1 (175) |
| conf.rand.p3 | 71 / 117 / 3 / 1 | 3.2 | -17 [-25, -9] | -1 [-5, +2] | -1 [-3, +0] | 187, -15 [-23, -8], 1 / 16 | 2 / 1 / 2 | 2 / 0 / 0 / 1 | -13 (179) |
| conf.rand.m3 | 90 / 99 / 1 / 2 | 3.5 | +2 [-6, +10] | +0 [-3, +3] | +0 [+0, +0] | 188, +3 [-5, +11], 10 / 7 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +3 (176) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +60 (31.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +77 [+60, +94]; vs conf.rand.m3 +58 [+40, +75] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.7, +1.6]; moved out / in 6 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +24 on 85 questions; mean words 3.4 -> 6.1
KEY family=E judge=allenai layer=11 A=incomplete B=not_met dR=+60 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+58 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 108 / 76 / 6 / 2 | 6.1 | +38 [+20, +56] | +2 [-5, +9] | +1 [-2, +5] | 176, +36 [+19, +53], 60 / 24 | 6 / 8 / 4 | 2 / 4 / 4 / 4 | +24 (82) |
| conf.md.p3 | 71 / 114 / 7 / 0 | 3.1 | +1 [-9, +11] | +3 [-2, +8] | -1 [-3, +0] | 180, +0 [-9, +9], 11 / 11 | 2 / 5 / 2 | 1 / 1 / 2 / 3 | +0 (169) |
| conf.rand.p3 | 63 / 117 / 11 / 1 | 3.2 | -7 [-14, -1] | -2 [-7, +3] | +0 [+0, +0] | 178, -5 [-11, +0], 2 / 7 | 4 / 2 / 4 | 2 / 2 / 0 / 2 | -5 (170) |
| conf.rand.m3 | 76 / 102 / 9 / 5 | 3.5 | +6 [-2, +14] | -4 [-9, +1] | +4 [+1, +8] | 177, +7 [+0, +15], 11 / 4 | 5 / 1 / 3 | 2 / 3 / 1 / 0 | +5 (168) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +38 (19.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +45 [+27, +63]; vs conf.rand.m3 +32 [+14, +50] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.0 pp [-2.6, +4.7]; moved out / in 6 / 8 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +24 on 82 questions; mean words 3.4 -> 6.1
KEY family=E judge=pro layer=11 A=incomplete B=not_met dR=+38 dR_F=NA dAns=+2 dAns_F=NA rand_margin=+32 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met


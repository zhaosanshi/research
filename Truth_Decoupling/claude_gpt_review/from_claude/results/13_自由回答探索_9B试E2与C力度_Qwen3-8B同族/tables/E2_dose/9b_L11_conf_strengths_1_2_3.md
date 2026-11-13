# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 128 / 58 / 3 / 3 | 2.9 | +17 [+8, +26] | -1 [-3, +0] | +1 [+0, +3] | 186, +18 [+9, +27], 20 / 2 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +17 (175) |
| conf.md.p1 | 113 / 74 / 3 / 2 | 3.3 | +2 [-3, +7] | +0 [-3, +3] | +0 [+0, +0] | 186, +2 [-2, +6], 3 / 1 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +1 (174) |
| conf.rand.p1 | 110 / 77 / 3 / 2 | 3.2 | -1 [-5, +2] | +0 [-3, +3] | +0 [+0, +0] | 186, -1 [-3, +0], 0 / 1 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +0 (181) |
| conf.rand.m1 | 114 / 73 / 2 / 3 | 3.1 | +3 [-2, +8] | +0 [+0, +0] | +1 [+0, +3] | 187, +3 [-2, +8], 5 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +18 [+9, +28]; vs conf.rand.m1 +14 [+5, +24] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +17 on 175 questions; mean words 3.1 -> 2.9
KEY family=E judge=allenai layer=11 A=incomplete B=incomplete dR=+17 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+14 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 117 / 70 / 1 / 4 | 2.9 | +17 [+8, +27] | +0 [-3, +3] | +0 [-3, +3] | 186, +17 [+8, +26], 20 / 3 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | +16 (176) |
| conf.md.p1 | 99 / 86 / 2 / 5 | 3.3 | -1 [-7, +5] | -2 [-5, +0] | +1 [+0, +3] | 185, +0 [-6, +5], 4 / 4 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +1 (174) |
| conf.rand.p1 | 100 / 88 / 1 / 3 | 3.2 | +0 [-3, +3] | +1 [+0, +3] | -1 [-3, +0] | 187, -1 [-3, +0], 0 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (182) |
| conf.rand.m1 | 102 / 84 / 2 / 4 | 3.1 | +2 [-3, +7] | -1 [-3, +0] | +0 [+0, +0] | 186, +2 [-3, +7], 4 / 2 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +17 [+8, +26]; vs conf.rand.m1 +15 [+6, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.6, +1.6]; moved out / in 1 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +16 on 176 questions; mean words 3.1 -> 2.9
KEY family=E judge=pro layer=11 A=incomplete B=incomplete dR=+17 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+15 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 136 / 51 / 2 / 3 | 2.5 | +25 [+15, +36] | +0 [-3, +3] | +1 [+0, +3] | 186, +25 [+15, +36], 28 / 3 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +22 (177) |
| conf.md.p2 | 111 / 76 / 2 / 3 | 3.6 | +0 [-8, +8] | +0 [-4, +4] | +1 [+0, +3] | 185, +1 [-6, +8], 7 / 6 | 2 / 2 / 1 | 2 / 0 / 1 / 1 | +2 (163) |
| conf.rand.p2 | 110 / 78 / 2 / 2 | 3.2 | -1 [-7, +5] | +1 [+0, +3] | +0 [+0, +0] | 187, -2 [-8, +4], 3 / 5 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (179) |
| conf.rand.m2 | 116 / 70 / 2 / 4 | 3.1 | +5 [-1, +12] | -1 [-3, +0] | +2 [+0, +5] | 186, +6 [+0, +12], 8 / 2 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +2 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +25 (13.0% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +26 [+15, +38]; vs conf.rand.m2 +20 [+9, +31] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.6, +1.6]; moved out / in 1 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +22 on 177 questions; mean words 3.1 -> 2.5
KEY family=E judge=allenai layer=11 A=incomplete B=incomplete dR=+25 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+20 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 128 / 58 / 1 / 5 | 2.5 | +28 [+16, +40] | -1 [-5, +2] | +1 [-2, +5] | 185, +29 [+18, +41], 34 / 5 | 2 / 1 / 0 | 2 / 0 / 1 / 0 | +25 (177) |
| conf.md.p2 | 99 / 86 / 2 / 5 | 3.6 | -1 [-9, +7] | -2 [-5, +0] | +1 [+0, +3] | 185, +0 [-8, +8], 8 / 8 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +2 (163) |
| conf.rand.p2 | 96 / 91 / 1 / 4 | 3.2 | -4 [-10, +2] | +0 [-3, +3] | +0 [-3, +3] | 186, -4 [-10, +1], 2 / 6 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | -1 (180) |
| conf.rand.m2 | 103 / 82 / 2 / 5 | 3.1 | +3 [-3, +9] | -2 [-5, +0] | +1 [+0, +3] | 185, +3 [-3, +9], 6 / 3 | 2 / 0 / 1 | 0 / 2 / 0 / 0 | +2 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +28 (14.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +32 [+20, +44]; vs conf.rand.m2 +25 [+13, +37] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +25 on 177 questions; mean words 3.1 -> 2.5
KEY family=E judge=pro layer=11 A=incomplete B=incomplete dR=+28 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+25 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 150 / 38 / 1 / 3 | 2.8 | +39 [+25, +53] | +1 [-2, +5] | +1 [+0, +3] | 186, +37 [+24, +51], 46 / 9 | 1 / 2 / 0 | 0 / 1 / 2 / 0 | +27 (165) |
| conf.md.p3 | 117 / 71 / 1 / 3 | 4.1 | +6 [-3, +15] | +1 [+0, +3] | +1 [+0, +3] | 187, +6 [-3, +15], 13 / 7 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +2 (153) |
| conf.rand.p3 | 109 / 78 / 3 / 2 | 3.3 | -2 [-9, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, -2 [-9, +5], 6 / 8 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (174) |
| conf.rand.m3 | 117 / 69 / 3 / 3 | 3.1 | +6 [-1, +13] | -1 [-3, +0] | +1 [+0, +3] | 186, +7 [+1, +14], 9 / 2 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +3 (175) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +39 (20.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +41 [+27, +55]; vs conf.rand.m3 +33 [+19, +47] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-1.0, +2.6]; moved out / in 1 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +27 on 165 questions; mean words 3.1 -> 2.8
KEY family=E judge=allenai layer=11 A=incomplete B=incomplete dR=+39 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+33 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 133 / 47 / 4 / 8 | 2.8 | +33 [+19, +47] | -7 [-14, -1] | +4 [-1, +10] | 178, +36 [+23, +49], 44 / 8 | 9 / 2 / 3 | 4 / 5 / 1 / 1 | +30 (162) |
| conf.md.p3 | 101 / 83 / 4 / 4 | 4.1 | +1 [-8, +10] | -3 [-7, +0] | +0 [+0, +0] | 184, +2 [-7, +11], 11 / 9 | 3 / 0 / 3 | 1 / 2 / 0 / 0 | +1 (151) |
| conf.rand.p3 | 96 / 91 / 1 / 4 | 3.3 | -4 [-11, +3] | +0 [-3, +3] | +0 [-3, +3] | 186, -4 [-10, +2], 3 / 7 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | +0 (175) |
| conf.rand.m3 | 105 / 80 / 2 / 5 | 3.1 | +5 [-1, +12] | -2 [-5, +0] | +1 [+0, +3] | 185, +5 [-1, +12], 8 / 3 | 2 / 0 / 1 | 0 / 2 / 0 / 0 | +4 (175) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +33 (17.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +37 [+23, +52]; vs conf.rand.m3 +28 [+14, +42] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -3.6 pp [-7.3, -0.5]; moved out / in 9 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +30 on 162 questions; mean words 3.1 -> 2.8
KEY family=E judge=pro layer=11 A=not_met B=not_met dR=+33 dR_F=NA dAns=-7 dAns_F=NA rand_margin=+28 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=not_met


# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 129 / 58 / 3 / 2 | 2.9 | +18 [+9, +27] | +0 [+0, +0] | +0 [+0, +0] | 187, +18 [+9, +27], 20 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +16 (175) |
| conf.md.p1 | 123 / 65 / 2 / 2 | 3.4 | +12 [+5, +20] | +1 [+0, +3] | +0 [+0, +0] | 187, +11 [+4, +19], 13 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +9 (168) |
| conf.rand.p1 | 106 / 81 / 3 / 2 | 3.2 | -5 [-10, -1] | +0 [+0, +0] | +0 [+0, +0] | 187, -5 [-10, -1], 0 / 5 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -3 (178) |
| conf.rand.m1 | 116 / 72 / 2 / 2 | 3.1 | +5 [+0, +10] | +1 [+0, +3] | +0 [+0, +0] | 187, +4 [+0, +9], 5 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +18 (9.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +23 [+14, +33]; vs conf.rand.m1 +13 [+5, +22] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.5, +1.5]; moved out / in 0 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +16 on 175 questions; mean words 3.1 -> 2.9
KEY family=E judge=allenai layer=15 A=incomplete B=incomplete dR=+18 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+13 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 118 / 69 / 1 / 4 | 2.9 | +18 [+10, +27] | +0 [-3, +3] | +0 [-3, +3] | 186, +18 [+10, +27], 19 / 1 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | +16 (175) |
| conf.md.p1 | 110 / 77 / 1 / 4 | 3.4 | +10 [+3, +18] | +0 [-3, +3] | +0 [-3, +3] | 186, +10 [+3, +18], 12 / 2 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | +9 (168) |
| conf.rand.p1 | 94 / 91 / 3 / 4 | 3.2 | -6 [-11, -2] | -2 [-5, +0] | +0 [+0, +0] | 185, -4 [-8, -1], 0 / 4 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | -2 (178) |
| conf.rand.m1 | 103 / 84 / 2 / 3 | 3.1 | +3 [+0, +7] | +0 [-3, +3] | -1 [-3, +0] | 186, +2 [+0, +5], 2 / 0 | 1 / 1 / 1 | 0 / 1 / 1 / 0 | +2 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +18 (9.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +24 [+14, +34]; vs conf.rand.m1 +15 [+7, +23] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.6, +1.6]; moved out / in 1 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +16 on 175 questions; mean words 3.1 -> 2.9
KEY family=E judge=pro layer=15 A=incomplete B=incomplete dR=+18 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+15 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 135 / 53 / 2 / 2 | 2.5 | +24 [+13, +36] | +1 [+0, +3] | +0 [+0, +0] | 187, +23 [+12, +35], 29 / 6 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +21 (178) |
| conf.md.p2 | 115 / 71 / 3 / 3 | 5.4 | +4 [-8, +16] | -1 [-3, +0] | +1 [+0, +3] | 186, +4 [-8, +16], 20 / 16 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +3 (147) |
| conf.rand.p2 | 103 / 86 / 0 / 3 | 3.3 | -8 [-15, -1] | +2 [+0, +5] | +1 [+0, +3] | 187, -8 [-15, -1], 3 / 11 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -3 (173) |
| conf.rand.m2 | 115 / 71 / 4 / 2 | 3.3 | +4 [-3, +11] | -1 [-5, +2] | +0 [+0, +0] | 185, +5 [-1, +11], 7 / 2 | 2 / 1 / 2 | 2 / 0 / 1 / 0 | +2 (172) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +24 (12.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +32 [+21, +44]; vs conf.rand.m2 +20 [+8, +32] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [+0.0, +1.6]; moved out / in 0 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +21 on 178 questions; mean words 3.1 -> 2.5
KEY family=E judge=allenai layer=15 A=incomplete B=incomplete dR=+24 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+20 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 126 / 58 / 2 / 6 | 2.5 | +26 [+15, +38] | -3 [-8, +1] | +2 [-2, +6] | 183, +28 [+18, +39], 30 / 2 | 4 / 1 / 1 | 3 / 1 / 1 / 0 | +26 (175) |
| conf.md.p2 | 108 / 78 / 2 / 4 | 5.4 | +8 [-4, +20] | -1 [-5, +2] | +0 [-3, +3] | 185, +8 [-4, +20], 22 / 14 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +6 (146) |
| conf.rand.p2 | 94 / 91 / 3 / 4 | 3.3 | -6 [-13, +1] | -2 [-5, +0] | +0 [+0, +0] | 185, -4 [-10, +2], 3 / 7 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | -1 (172) |
| conf.rand.m2 | 101 / 84 / 3 / 4 | 3.3 | +1 [-5, +7] | -2 [-6, +2] | +0 [-3, +3] | 184, +2 [-3, +7], 4 / 2 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +1 (173) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +26 (13.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +32 [+21, +44]; vs conf.rand.m2 +25 [+15, +36] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.2, +0.5]; moved out / in 4 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +26 on 175 questions; mean words 3.1 -> 2.5
KEY family=E judge=pro layer=15 A=incomplete B=not_met dR=+26 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+25 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 145 / 43 / 1 / 3 | 2.5 | +34 [+22, +47] | +1 [+0, +3] | +1 [+0, +3] | 187, +33 [+21, +46], 39 / 6 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +25 (170) |
| conf.md.p3 | 114 / 72 / 2 / 4 | 8.6 | +3 [-10, +17] | -1 [-3, +0] | +2 [+0, +5] | 186, +4 [-9, +18], 25 / 21 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (137) |
| conf.rand.p3 | 101 / 88 / 1 / 2 | 3.4 | -10 [-18, -2] | +2 [+0, +5] | +0 [+0, +0] | 187, -10 [-18, -2], 4 / 14 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -3 (170) |
| conf.rand.m3 | 115 / 71 / 4 / 2 | 3.4 | +4 [-3, +11] | -1 [-5, +2] | +0 [+0, +0] | 185, +5 [-1, +11], 7 / 2 | 2 / 1 / 2 | 2 / 0 / 1 / 0 | +2 (170) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +34 (17.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +44 [+31, +57]; vs conf.rand.m3 +30 [+17, +44] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [+0.0, +1.6]; moved out / in 0 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +25 on 170 questions; mean words 3.1 -> 2.5
KEY family=E judge=allenai layer=15 A=incomplete B=incomplete dR=+34 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+30 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 130 / 53 / 1 / 8 | 2.5 | +30 [+16, +44] | -4 [-9, +0] | +4 [+0, +9] | 182, +33 [+20, +46], 41 / 8 | 5 / 1 / 0 | 4 / 1 / 1 / 0 | +28 (166) |
| conf.md.p3 | 97 / 88 / 2 / 5 | 8.6 | -3 [-17, +11] | -2 [-6, +2] | +1 [-2, +5] | 184, -2 [-15, +11], 22 / 24 | 3 / 1 / 1 | 2 / 1 / 1 / 0 | +4 (137) |
| conf.rand.p3 | 95 / 90 / 3 / 4 | 3.4 | -5 [-14, +3] | -2 [-5, +0] | +0 [+0, +0] | 185, -3 [-11, +5], 7 / 10 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | +2 (169) |
| conf.rand.m3 | 102 / 81 / 4 / 5 | 3.4 | +2 [-5, +9] | -4 [-9, +0] | +1 [-2, +4] | 182, +4 [-2, +10], 7 / 3 | 5 / 1 / 3 | 3 / 2 / 1 / 0 | +3 (170) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +30 (15.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +35 [+20, +49]; vs conf.rand.m3 +28 [+15, +41] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.1 pp [-4.7, +0.0]; moved out / in 5 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +28 on 166 questions; mean words 3.1 -> 2.5
KEY family=E judge=pro layer=15 A=incomplete B=not_met dR=+30 dR_F=NA dAns=-4 dAns_F=NA rand_margin=+28 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met


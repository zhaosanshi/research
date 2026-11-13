# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 128 / 59 / 3 / 2 | 2.8 | +17 [+9, +26] | +0 [+0, +0] | +0 [+0, +0] | 187, +17 [+9, +26], 18 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +16 (177) |
| conf.md.p1 | 115 / 72 / 3 / 2 | 3.4 | +4 [-1, +10] | +0 [-3, +3] | +0 [+0, +0] | 186, +4 [+0, +9], 5 / 1 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +1 (171) |
| conf.rand.p1 | 111 / 77 / 2 / 2 | 3.3 | +0 [-5, +5] | +1 [+0, +3] | +0 [+0, +0] | 187, -1 [-6, +3], 2 / 3 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (177) |
| conf.rand.m1 | 112 / 74 / 2 / 4 | 2.9 | +1 [-3, +5] | -1 [-3, +0] | +2 [+0, +5] | 186, +2 [-2, +6], 3 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +2 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +17 [+8, +27]; vs conf.rand.m1 +16 [+8, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.5, +1.5]; moved out / in 0 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +16 on 177 questions; mean words 3.1 -> 2.8
KEY family=E judge=allenai layer=14 A=incomplete B=incomplete dR=+17 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+16 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 118 / 69 / 1 / 4 | 2.8 | +18 [+10, +27] | +0 [-3, +3] | +0 [-3, +3] | 186, +18 [+10, +27], 19 / 1 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | +16 (177) |
| conf.md.p1 | 101 / 84 / 3 / 4 | 3.4 | +1 [-5, +7] | -2 [-6, +2] | +0 [-3, +3] | 184, +2 [-2, +7], 4 / 2 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +2 (170) |
| conf.rand.p1 | 97 / 89 / 2 / 4 | 3.3 | -3 [-8, +1] | -1 [-5, +2] | +0 [-3, +3] | 185, -3 [-7, +0], 0 / 3 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -1 (178) |
| conf.rand.m1 | 104 / 82 / 1 / 5 | 2.9 | +4 [+1, +8] | -1 [-3, +0] | +1 [+0, +3] | 186, +4 [+1, +8], 4 / 0 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +4 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +18 (9.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +21 [+13, +30]; vs conf.rand.m1 +14 [+6, +22] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.6, +1.6]; moved out / in 1 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +16 on 177 questions; mean words 3.1 -> 2.8
KEY family=E judge=pro layer=14 A=incomplete B=incomplete dR=+18 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+14 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 135 / 53 / 2 / 2 | 2.4 | +24 [+14, +35] | +1 [+0, +3] | +0 [+0, +0] | 187, +23 [+13, +34], 27 / 4 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +21 (180) |
| conf.md.p2 | 113 / 72 / 3 / 4 | 3.4 | +2 [-4, +8] | -2 [-5, +0] | +2 [+0, +5] | 185, +4 [-1, +10], 6 / 2 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | +3 (166) |
| conf.rand.p2 | 111 / 76 / 3 / 2 | 3.5 | +0 [-7, +7] | +0 [-3, +3] | +0 [+0, +0] | 186, +0 [-6, +6], 5 / 5 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +0 (170) |
| conf.rand.m2 | 114 / 72 / 2 / 4 | 2.8 | +3 [-2, +8] | -1 [-3, +0] | +2 [+0, +5] | 186, +4 [+0, +9], 5 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +4 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +24 (12.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +24 [+12, +36]; vs conf.rand.m2 +21 [+11, +32] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [+0.0, +1.6]; moved out / in 0 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +21 on 180 questions; mean words 3.1 -> 2.4
KEY family=E judge=allenai layer=14 A=incomplete B=incomplete dR=+24 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+21 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 128 / 58 / 1 / 5 | 2.4 | +28 [+17, +40] | -1 [-5, +2] | +1 [-2, +5] | 185, +29 [+18, +40], 32 / 3 | 2 / 1 / 0 | 2 / 0 / 1 / 0 | +26 (179) |
| conf.md.p2 | 98 / 86 / 3 / 5 | 3.4 | -2 [-9, +5] | -3 [-7, +0] | +1 [+0, +3] | 184, +0 [-6, +6], 5 / 5 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +3 (166) |
| conf.rand.p2 | 95 / 91 / 2 / 4 | 3.5 | -5 [-11, +1] | -1 [-5, +2] | +0 [-3, +3] | 185, -5 [-10, +0], 1 / 6 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -3 (171) |
| conf.rand.m2 | 107 / 79 / 1 / 5 | 2.8 | +7 [+2, +13] | -1 [-3, +0] | +1 [+0, +3] | 186, +7 [+2, +13], 8 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +7 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +28 (14.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +33 [+21, +45]; vs conf.rand.m2 +21 [+10, +32] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +26 on 179 questions; mean words 3.1 -> 2.4
KEY family=E judge=pro layer=14 A=incomplete B=incomplete dR=+28 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+21 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 143 / 45 / 1 / 3 | 2.3 | +32 [+20, +44] | +1 [+0, +3] | +1 [+0, +3] | 187, +31 [+19, +43], 37 / 6 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +23 (174) |
| conf.md.p3 | 109 / 76 / 3 / 4 | 3.7 | -2 [-10, +6] | -2 [-5, +0] | +2 [+0, +5] | 185, +0 [-7, +7], 7 / 7 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | -2 (155) |
| conf.rand.p3 | 110 / 77 / 3 / 2 | 3.7 | -1 [-9, +7] | +0 [-3, +3] | +0 [+0, +0] | 186, -1 [-9, +7], 7 / 8 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +0 (161) |
| conf.rand.m3 | 116 / 70 / 2 / 4 | 2.6 | +5 [-1, +12] | -1 [-3, +0] | +2 [+0, +5] | 186, +6 [+0, +12], 8 / 2 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +6 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +32 (16.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +33 [+20, +47]; vs conf.rand.m3 +27 [+16, +39] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [+0.0, +1.6]; moved out / in 0 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +23 on 174 questions; mean words 3.1 -> 2.3
KEY family=E judge=allenai layer=14 A=incomplete B=incomplete dR=+32 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+27 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p3, strength 3 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p3 | 130 / 53 / 2 / 7 | 2.3 | +30 [+16, +44] | -4 [-9, +0] | +3 [-1, +8] | 182, +32 [+19, +45], 40 / 8 | 5 / 1 / 1 | 3 / 2 / 1 / 0 | +27 (171) |
| conf.md.p3 | 93 / 90 / 4 / 5 | 3.7 | -7 [-17, +2] | -4 [-8, -1] | +1 [+0, +3] | 183, -4 [-13, +5], 8 / 12 | 4 / 0 / 3 | 3 / 1 / 0 / 0 | +2 (155) |
| conf.rand.p3 | 94 / 92 / 2 / 4 | 3.7 | -6 [-14, +1] | -1 [-5, +2] | +0 [-3, +3] | 185, -6 [-13, +1], 3 / 9 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -3 (162) |
| conf.rand.m3 | 108 / 78 / 1 / 5 | 2.6 | +8 [+2, +15] | -1 [-3, +0] | +1 [+0, +3] | 186, +8 [+2, +15], 10 / 2 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +8 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +30 (15.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p3 +36 [+22, +50]; vs conf.rand.m3 +22 [+10, +35] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.1 pp [-4.7, +0.0]; moved out / in 5 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +27 on 171 questions; mean words 3.1 -> 2.3
KEY family=E judge=pro layer=14 A=incomplete B=not_met dR=+30 dR_F=NA dAns=-4 dAns_F=NA rand_margin=+22 EXPLORATORY base=conf main=conf.w.p3 strength=3 tested_A_without_F=met


# seed 1051
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 127 / 65 / 0 / 0 | 5.2 | +39 [+25, +52] | +3 [+0, +7] | -2 [-5, +0] | 189, +36 [+23, +49], 44 / 8 | 0 / 3 / 0 | 0 / 0 / 3 / 0 | +14 (121) |
| conf.md.p2 | 82 / 107 / 1 / 2 | 3.3 | -6 [-14, +2] | +0 [-3, +3] | +0 [+0, +0] | 188, -5 [-13, +3], 6 / 11 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | -4 (175) |
| conf.rand.p2 | 87 / 102 / 1 / 2 | 3.7 | -1 [-8, +7] | +0 [-3, +3] | +0 [+0, +0] | 188, +0 [-7, +7], 7 / 7 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +1 (175) |
| conf.rand.m2 | 89 / 100 / 2 / 1 | 3.2 | +1 [-5, +7] | +0 [-3, +3] | -1 [-3, +0] | 188, +1 [-5, +7], 5 / 4 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +1 (186) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +39 (20.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +40 [+27, +53]; vs conf.rand.m2 +38 [+25, +51] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.6 pp [+0.0, +3.6]; moved out / in 0 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +14 on 121 questions; mean words 3.4 -> 5.2
KEY family=E judge=allenai layer=10 A=incomplete B=not_met dR=+39 dR_F=NA dAns=+3 dAns_F=NA rand_margin=+38 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met


# seed 1051
# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 105 / 77 / 6 / 4 | 5.2 | +35 [+21, +49] | +0 [-6, +6] | +3 [-1, +8] | 177, +32 [+19, +45], 41 / 9 | 5 / 5 / 2 | 2 / 3 / 5 / 0 | +12 (117) |
| conf.md.p2 | 69 / 111 / 9 / 3 | 3.3 | -1 [-8, +5] | -2 [-5, +0] | +2 [+0, +5] | 180, +0 [-6, +6], 5 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (167) |
| conf.rand.p2 | 69 / 113 / 9 / 1 | 3.7 | -1 [-8, +7] | +0 [-3, +3] | +0 [+0, +0] | 181, -1 [-8, +6], 6 / 7 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | -1 (169) |
| conf.rand.m2 | 69 / 112 / 10 / 1 | 3.2 | -1 [-8, +5] | -1 [-5, +2] | +0 [-3, +3] | 180, -1 [-7, +5], 4 / 5 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -1 (179) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +35 (18.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +36 [+22, +50]; vs conf.rand.m2 +36 [+23, +49] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-3.1, +3.1]; moved out / in 5 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +12 on 117 questions; mean words 3.4 -> 5.2
KEY family=E judge=pro layer=10 A=incomplete B=not_met dR=+35 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+36 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met


# seed 2045
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 127 / 65 / 0 / 0 | 5.1 | +39 [+26, +52] | +3 [+0, +7] | -2 [-5, +0] | 189, +36 [+23, +48], 43 / 7 | 0 / 3 / 0 | 0 / 0 / 3 / 0 | +16 (123) |
| conf.md.p2 | 82 / 107 / 1 / 2 | 3.3 | -6 [-14, +2] | +0 [-3, +3] | +0 [+0, +0] | 188, -5 [-13, +3], 6 / 11 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | -4 (175) |
| conf.rand.p2 | 86 / 103 / 1 / 2 | 3.7 | -2 [-9, +5] | +0 [-3, +3] | +0 [+0, +0] | 188, -1 [-8, +6], 6 / 7 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +1 (175) |
| conf.rand.m2 | 89 / 101 / 1 / 1 | 3.2 | +1 [-5, +7] | +1 [+0, +3] | -1 [-3, +0] | 189, +0 [-6, +6], 5 / 5 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (188) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +39 (20.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +41 [+28, +54]; vs conf.rand.m2 +38 [+25, +50] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.6 pp [+0.0, +3.6]; moved out / in 0 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +16 on 123 questions; mean words 3.4 -> 5.1
KEY family=E judge=allenai layer=10 A=incomplete B=not_met dR=+39 dR_F=NA dAns=+3 dAns_F=NA rand_margin=+38 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met


# seed 2045
# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 106 / 76 / 6 / 4 | 5.1 | +36 [+23, +50] | +0 [-6, +6] | +3 [-1, +8] | 177, +33 [+20, +46], 41 / 8 | 5 / 5 / 2 | 2 / 3 / 5 / 0 | +14 (119) |
| conf.md.p2 | 69 / 111 / 9 / 3 | 3.3 | -1 [-8, +5] | -2 [-5, +0] | +2 [+0, +5] | 180, +0 [-6, +6], 5 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (167) |
| conf.rand.p2 | 68 / 114 / 9 / 1 | 3.7 | -2 [-9, +5] | +0 [-3, +3] | +0 [+0, +0] | 181, -2 [-9, +5], 5 / 7 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | -1 (169) |
| conf.rand.m2 | 71 / 111 / 9 / 1 | 3.2 | +1 [-5, +7] | +0 [-3, +3] | +0 [-3, +3] | 181, +0 [-6, +5], 4 / 4 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | +0 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +36 (18.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +38 [+24, +52]; vs conf.rand.m2 +35 [+22, +48] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-3.1, +3.1]; moved out / in 5 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +14 on 119 questions; mean words 3.4 -> 5.1
KEY family=E judge=pro layer=10 A=incomplete B=not_met dR=+36 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+35 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met



# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 120 / 49 / 0 / 23 | 6.1 | +17 [+8, +27] | +3 [-3, +9] | -2 [-8, +3] | 163, +12 [+4, +20], 15 / 3 | 3 / 6 / 0 | 0 / 3 / 5 / 1 | +6 (101) |
| x.w.m1 | 91 / 77 / 2 / 22 | 2.5 | -12 [-21, -3] | +2 [-5, +9] | -3 [-10, +3] | 161, -9 [-17, -1], 4 / 13 | 5 / 7 / 1 | 4 / 1 / 1 / 6 | -7 (156) |
| x.md.p1 | 101 / 62 / 1 / 28 | 3.3 | -2 [-8, +3] | -3 [-8, +1] | +3 [-1, +8] | 162, +0 [-5, +5], 3 / 3 | 4 / 1 / 1 | 2 / 2 / 0 / 1 | -1 (151) |
| x.rand.p1 | 102 / 64 / 2 / 24 | 3.3 | -1 [-6, +4] | +0 [-3, +3] | -1 [-3, +0] | 165, +0 [-5, +5], 3 / 3 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +2 (159) |
| x.rand.m1 | 106 / 57 / 1 / 28 | 3.2 | +3 [+0, +7] | -3 [-7, +0] | +3 [+0, +7] | 163, +3 [+0, +7], 3 / 0 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | +2 (161) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 139 / 41 / 2 / 10; dR +5 [-4, +14]; answered-rate change +2.6 pp [+0.0, +5.7]; out / in 1 / 6; mean words 10.1 -> 13.4
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +18 [+9, +28]; vs x.rand.m1 +14 [+5, +23] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.6 pp [-1.6, +4.7]; moved out / in 3 / 6 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 101 questions; mean words 3.2 -> 6.1
KEY family=E judge=allenai layer=9 A=met B=not_met dR=+17 dR_F=+5 dAns=+3 dAns_F=+5 rand_margin=+14

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 9

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 110 / 49 / 1 / 32 | +9 [+1, +17] | -1 [-8, +5] | +2 [-3, +8] | 154, +6, 9 / 3 | 6 / 5 / 1 | +6 (109) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 112 / 59 / 1 / 20 | -13 [-22, -4] | -5 [-11, +1] | +4 [-1, +10] | 169, -10, 3 / 13 | 7 / 2 / 1 | -9 (167) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 110 / 78 / 2 / 2 | -1 [-5, +2] | +1 [+0, +3] | +0 [+0, +0] | 187, -2, 0 / 2 | 0 / 1 / 0 | +0 (183) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 113 / 73 / 3 / 3 | +2 [-4, +8] | -1 [-5, +2] | +1 [+0, +3] | 185, +3, 5 / 2 | 2 / 1 / 1 | +1 (167) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 115 / 72 / 3 / 2 | +4 [-1, +10] | +0 [+0, +0] | +0 [+0, +0] | 187, +4, 6 / 2 | 0 / 0 / 0 | +1 (178) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 110 / 76 / 4 / 2 | -1 [-5, +3] | -1 [-3, +0] | +0 [+0, +0] | 186, +0, 2 / 2 | 1 / 0 / 1 | +0 (178) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 130 / 60 / 0 / 2 | +19 [+11, +28] | +3 [+0, +7] | +0 [+0, +0] | 187, +18, 19 / 1 | 0 / 3 / 0 | +8 (122) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 131 / 59 / 0 / 2 | +20 [+12, +29] | +3 [+0, +7] | +0 [+0, +0] | 187, +19, 20 / 1 | 0 / 3 / 0 | +9 (122) |
| conf_E1.wq.m1 | 136 / 54 / 0 / 2 | 115 / 71 / 2 / 4 | -21 [-31, -12] | -4 [-8, -1] | +2 [+0, +5] | 186, -19, 3 / 22 | 4 / 0 / 2 | -17 (181) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 139 / 41 / 2 / 10 | +5 [-4, +14] | +5 [+0, +11] | -5 [-11, +0] | 174, -1, 8 / 9 | 1 / 6 / 0 | +1 (86) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 130 / 60 / 0 / 2 | 5.4 | +19 [+11, +28] | +3 [+0, +7] | +0 [+0, +0] | 187, +18 [+10, +27], 19 / 1 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +8 (122) |
| conf.md.p1 | 113 / 73 / 3 / 3 | 3.4 | +2 [-4, +8] | -1 [-5, +2] | +1 [+0, +3] | 185, +3 [-2, +8], 5 / 2 | 2 / 1 / 1 | 2 / 0 / 1 / 0 | +1 (167) |
| conf.rand.p1 | 110 / 76 / 4 / 2 | 3.2 | -1 [-5, +3] | -1 [-3, +0] | +0 [+0, +0] | 186, +0 [-4, +4], 2 / 2 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (178) |
| conf.rand.m1 | 115 / 72 / 3 / 2 | 3.2 | +4 [-1, +10] | +0 [+0, +0] | +0 [+0, +0] | 187, +4 [-1, +10], 6 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +19 (9.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +20 [+11, +29]; vs conf.rand.m1 +15 [+7, +24] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.6 pp [+0.0, +3.6]; moved out / in 0 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +8 on 122 questions; mean words 3.1 -> 5.4
KEY family=E judge=allenai layer=9 A=incomplete B=not_met dR=+19 dR_F=NA dAns=+3 dAns_F=NA rand_margin=+15 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 104 / 59 / 6 / 23 | 6.1 | +5 [-5, +15] | +0 [-6, +6] | -2 [-8, +3] | 158, +2 [-7, +11], 11 / 9 | 5 / 5 / 2 | 1 / 4 / 4 / 1 | +3 (98) |
| x.w.m1 | 88 / 79 / 3 / 22 | 2.5 | -11 [-21, -1] | +4 [-3, +11] | -3 [-10, +3] | 158, -9 [-18, -1], 5 / 14 | 5 / 9 / 1 | 4 / 1 / 2 / 7 | -8 (152) |
| x.md.p1 | 93 / 67 / 4 / 28 | 3.3 | -6 [-12, +0] | -3 [-8, +1] | +3 [-1, +8] | 159, -5 [-11, +1], 2 / 7 | 4 / 1 / 0 | 1 / 3 / 0 / 1 | -3 (148) |
| x.rand.p1 | 94 / 70 / 4 / 24 | 3.3 | -5 [-11, +0] | +1 [+0, +3] | -1 [-3, +0] | 163, -5 [-11, +0], 1 / 6 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -3 (156) |
| x.rand.m1 | 102 / 59 / 3 / 28 | 3.2 | +3 [-2, +8] | -2 [-6, +2] | +3 [+0, +7] | 160, +2 [-3, +7], 4 / 2 | 3 / 1 / 0 | 0 / 3 / 1 / 0 | +1 (158) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 118 / 60 / 4 / 10; dR +7 [-2, +16]; answered-rate change +4.7 pp [+1.0, +8.3]; out / in 2 / 11; mean words 10.1 -> 13.4
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +10 [+0, +20]; vs x.rand.m1 +2 [-8, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-3.1, +3.1]; moved out / in 5 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 98 questions; mean words 3.2 -> 6.1
KEY family=E judge=pro layer=9 A=not_met B=not_met dR=+5 dR_F=+7 dAns=+0 dAns_F=+9 rand_margin=+2

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 9

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 99 / 57 / 4 / 32 | +7 [+0, +15] | -1 [-8, +5] | +2 [-3, +8] | 151, +3, 7 / 4 | 6 / 5 / 1 | +1 (107) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 106 / 64 / 1 / 21 | -8 [-17, +1] | -1 [-8, +5] | +4 [-1, +10] | 165, -9, 3 / 12 | 6 / 5 / 0 | -8 (163) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 98 / 89 / 1 / 4 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -2, 0 / 2 | 0 / 0 / 0 | +0 (183) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 100 / 85 / 2 / 5 | +0 [-6, +5] | -2 [-5, +0] | +1 [+0, +3] | 185, +1, 4 / 3 | 2 / 0 / 1 | +0 (169) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 101 / 85 / 2 / 4 | +1 [-2, +4] | -1 [-3, +0] | +0 [+0, +0] | 186, +1, 2 / 1 | 1 / 0 / 1 | +0 (178) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 102 / 86 / 1 / 3 | +2 [-3, +7] | +1 [+0, +3] | -1 [-3, +0] | 187, +1, 3 / 2 | 0 / 1 / 0 | +1 (179) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 107 / 78 / 2 / 5 | +7 [-2, +16] | -2 [-5, +0] | +1 [+0, +3] | 185, +8, 15 / 7 | 2 / 0 / 1 | +5 (120) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 107 / 78 / 2 / 5 | +7 [-2, +16] | -2 [-5, +0] | +1 [+0, +3] | 185, +8, 15 / 7 | 2 / 0 / 1 | +6 (120) |
| conf_E1.wq.m1 | 117 / 68 / 2 / 5 | 104 / 81 / 2 / 5 | -13 [-23, -3] | +0 [-4, +4] | +0 [-3, +3] | 183, -13, 6 / 19 | 2 / 2 / 1 | -11 (178) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 118 / 60 / 4 / 10 | +7 [-2, +16] | +9 [+2, +16] | -5 [-10, +0] | 167, +0, 8 / 8 | 2 / 11 / 1 | +1 (84) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 107 / 78 / 2 / 5 | 5.4 | +7 [-2, +16] | -2 [-5, +0] | +1 [+0, +3] | 185, +8 [-1, +17], 15 / 7 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +5 (120) |
| conf.md.p1 | 100 / 85 / 2 / 5 | 3.4 | +0 [-6, +5] | -2 [-5, +0] | +1 [+0, +3] | 185, +1 [-4, +6], 4 / 3 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +0 (169) |
| conf.rand.p1 | 102 / 86 / 1 / 3 | 3.2 | +2 [-3, +7] | +1 [+0, +3] | -1 [-3, +0] | 187, +1 [-3, +5], 3 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (179) |
| conf.rand.m1 | 101 / 85 / 2 / 4 | 3.2 | +1 [-2, +4] | -1 [-3, +0] | +0 [+0, +0] | 186, +1 [-2, +4], 2 / 1 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +0 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +5 [-5, +15]; vs conf.rand.m1 +6 [-3, +15] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +5 on 120 questions; mean words 3.1 -> 5.4
KEY family=E judge=pro layer=9 A=incomplete B=not_met dR=+7 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+5 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 124 / 46 / 0 / 22 | 7.0 | +21 [+12, +31] | +4 [-2, +10] | -3 [-9, +3] | 163, +15 [+7, +24], 18 / 3 | 3 / 7 / 0 | 0 / 3 / 6 / 1 | +5 (93) |
| x.w.m1 | 91 / 78 / 2 / 21 | 2.4 | -12 [-21, -3] | +3 [-4, +10] | -4 [-11, +2] | 161, -10 [-18, -3], 3 / 13 | 5 / 8 / 1 | 4 / 1 / 2 / 6 | -9 (157) |
| x.md.p1 | 112 / 61 / 0 / 19 | 3.9 | +9 [+2, +17] | +7 [+2, +13] | -6 [-12, -1] | 165, +4 [-2, +10], 7 / 3 | 1 / 8 / 0 | 0 / 1 / 5 / 3 | +2 (135) |
| x.rand.p1 | 102 / 64 / 2 / 24 | 3.1 | -1 [-7, +5] | +0 [-4, +4] | -1 [-5, +2] | 164, +0 [-6, +6], 4 / 4 | 2 / 2 / 1 | 1 / 1 / 0 / 2 | +1 (159) |
| x.rand.m1 | 104 / 59 / 0 / 29 | 3.5 | +1 [-2, +5] | -3 [-7, +0] | +4 [+1, +8] | 163, +1 [-2, +5], 2 / 1 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | +2 (157) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 145 / 39 / 1 / 7; dR +11 [+2, +21]; answered-rate change +4.7 pp [+2.1, +7.8]; out / in 0 / 9; mean words 10.1 -> 14.0
- main effect: dR +21 (10.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +22 [+12, +33]; vs x.rand.m1 +20 [+11, +30] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +2.1 pp [-1.0, +5.2]; moved out / in 3 / 7 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 93 questions; mean words 3.2 -> 7.0
KEY family=E judge=allenai layer=11 A=not_met B=not_met dR=+21 dR_F=+11 dAns=+4 dAns_F=+9 rand_margin=+20

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 114 / 48 / 1 / 29 | +13 [+5, +22] | +2 [-5, +9] | -1 [-7, +6] | 154, +7, 9 / 2 | 6 / 8 / 1 | +6 (109) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 109 / 57 / 1 / 25 | -16 [-26, -7] | -10 [-16, -4] | +9 [+4, +15] | 166, -9, 4 / 13 | 10 / 0 / 1 | -9 (164) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 110 / 78 / 2 / 2 | -1 [-5, +2] | +1 [+0, +3] | +0 [+0, +0] | 187, -2, 0 / 2 | 0 / 1 / 0 | +0 (183) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 117 / 72 / 1 / 2 | +6 [-1, +13] | +2 [+0, +5] | +0 [+0, +0] | 187, +5, 8 / 3 | 0 / 2 / 0 | +3 (156) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 114 / 73 / 2 / 3 | +3 [-3, +9] | +0 [+0, +0] | +1 [+0, +3] | 187, +3, 6 / 3 | 0 / 0 / 0 | +1 (178) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 110 / 78 / 2 / 2 | -1 [-6, +4] | +1 [+0, +3] | +0 [+0, +0] | 187, -2, 2 / 4 | 0 / 1 / 0 | -1 (182) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 133 / 57 / 0 / 2 | +22 [+12, +32] | +3 [+0, +7] | +0 [+0, +0] | 187, +21, 24 / 3 | 0 / 3 / 0 | +8 (115) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 132 / 58 / 0 / 2 | +21 [+12, +31] | +3 [+0, +7] | +0 [+0, +0] | 187, +20, 23 / 3 | 0 / 3 / 0 | +8 (118) |
| conf_E1.wq.m1 | 136 / 54 / 0 / 2 | 114 / 72 / 2 / 4 | -22 [-32, -13] | -4 [-8, -1] | +2 [+0, +5] | 186, -20, 2 / 22 | 4 / 0 / 2 | -18 (181) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 145 / 39 / 1 / 7 | +11 [+2, +21] | +9 [+4, +15] | -8 [-14, -3] | 175, +3, 10 / 7 | 0 / 9 / 0 | +4 (79) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 133 / 57 / 0 / 2 | 5.8 | +22 [+12, +32] | +3 [+0, +7] | +0 [+0, +0] | 187, +21 [+12, +31], 24 / 3 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +8 (115) |
| conf.md.p1 | 117 / 72 / 1 / 2 | 3.8 | +6 [-1, +13] | +2 [+0, +5] | +0 [+0, +0] | 187, +5 [-1, +12], 8 / 3 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +3 (156) |
| conf.rand.p1 | 110 / 78 / 2 / 2 | 3.2 | -1 [-6, +4] | +1 [+0, +3] | +0 [+0, +0] | 187, -2 [-7, +3], 2 / 4 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | -1 (182) |
| conf.rand.m1 | 114 / 73 / 2 / 3 | 3.2 | +3 [-3, +9] | +0 [+0, +0] | +1 [+0, +3] | 187, +3 [-3, +9], 6 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +22 (11.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +23 [+13, +34]; vs conf.rand.m1 +19 [+9, +29] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.6 pp [+0.0, +3.6]; moved out / in 0 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +8 on 115 questions; mean words 3.1 -> 5.8
KEY family=E judge=allenai layer=11 A=incomplete B=not_met dR=+22 dR_F=NA dAns=+3 dAns_F=NA rand_margin=+19 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 105 / 61 / 4 / 22 | 7.0 | +6 [-4, +16] | +3 [-3, +10] | -3 [-9, +3] | 159, +2 [-6, +10], 10 / 8 | 4 / 7 / 1 | 1 / 3 / 5 / 2 | +2 (91) |
| x.w.m1 | 81 / 87 / 3 / 21 | 2.4 | -18 [-28, -8] | +5 [-2, +13] | -4 [-11, +2] | 158, -16 [-25, -7], 3 / 19 | 5 / 10 / 1 | 4 / 1 / 2 / 8 | -15 (153) |
| x.md.p1 | 101 / 68 / 4 / 19 | 3.9 | +2 [-5, +9] | +6 [+0, +12] | -6 [-12, -1] | 161, -2 [-8, +4], 4 / 6 | 2 / 8 / 1 | 0 / 2 / 4 / 4 | -1 (131) |
| x.rand.p1 | 97 / 68 / 3 / 24 | 3.1 | -2 [-8, +3] | +2 [-2, +6] | -1 [-5, +2] | 162, -3 [-8, +2], 2 / 5 | 1 / 3 / 0 | 0 / 1 / 1 / 2 | -1 (156) |
| x.rand.m1 | 97 / 63 / 3 / 29 | 3.5 | -2 [-8, +3] | -3 [-8, +1] | +4 [+1, +8] | 159, -3 [-8, +2], 2 / 5 | 4 / 1 / 0 | 0 / 4 / 1 / 0 | -1 (153) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 124 / 58 / 3 / 7; dR +13 [+4, +22]; answered-rate change +6.8 pp [+3.1, +10.9]; out / in 1 / 14; mean words 10.1 -> 14.0
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +8 [-2, +18]; vs x.rand.m1 +8 [-1, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.6 pp [-1.6, +5.2]; moved out / in 4 / 7 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 91 questions; mean words 3.2 -> 7.0
KEY family=E judge=pro layer=11 A=not_met B=not_met dR=+6 dR_F=+13 dAns=+3 dAns_F=+13 rand_margin=+8

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 102 / 56 / 5 / 29 | +10 [+2, +18] | +1 [-7, +9] | -1 [-7, +6] | 150, +5, 8 / 3 | 7 / 8 / 2 | +3 (107) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 101 / 63 / 2 / 26 | -13 [-22, -4] | -7 [-14, +0] | +9 [+4, +15] | 161, -10, 3 / 13 | 10 / 3 / 1 | -9 (159) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 98 / 88 / 1 / 5 | -2 [-6, +2] | -1 [-3, +0] | +1 [+0, +3] | 186, -1, 1 / 2 | 1 / 0 / 0 | +1 (183) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 104 / 81 / 2 / 5 | +4 [-4, +12] | -2 [-5, +0] | +1 [+0, +3] | 185, +5, 10 / 5 | 2 / 0 / 1 | +4 (156) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 102 / 84 / 2 / 4 | +2 [-3, +8] | -1 [-3, +0] | +0 [+0, +0] | 186, +2, 5 / 3 | 1 / 0 / 1 | +1 (178) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 98 / 90 / 1 / 3 | -2 [-7, +3] | +1 [+0, +3] | -1 [-3, +0] | 187, -3, 1 / 4 | 0 / 1 / 0 | -2 (183) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 114 / 70 / 3 / 5 | +14 [+4, +24] | -3 [-7, +0] | +1 [+0, +3] | 184, +16, 21 / 5 | 3 / 0 / 2 | +7 (113) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 113 / 72 / 2 / 5 | +13 [+3, +23] | -2 [-5, +0] | +1 [+0, +3] | 185, +14, 19 / 5 | 2 / 0 / 1 | +6 (116) |
| conf_E1.wq.m1 | 117 / 68 / 2 / 5 | 109 / 77 / 1 / 5 | -8 [-18, +2] | +1 [-2, +4] | +0 [-3, +3] | 184, -9, 8 / 17 | 1 / 2 / 0 | -7 (179) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 124 / 58 / 3 / 7 | +13 [+4, +22] | +13 [+6, +21] | -8 [-14, -3] | 168, +5, 10 / 5 | 1 / 14 / 1 | +2 (76) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 114 / 70 / 3 / 5 | 5.8 | +14 [+4, +24] | -3 [-7, +0] | +1 [+0, +3] | 184, +16 [+6, +26], 21 / 5 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +7 (113) |
| conf.md.p1 | 104 / 81 / 2 / 5 | 3.8 | +4 [-4, +12] | -2 [-5, +0] | +1 [+0, +3] | 185, +5 [-2, +13], 10 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +4 (156) |
| conf.rand.p1 | 98 / 90 / 1 / 3 | 3.2 | -2 [-7, +3] | +1 [+0, +3] | -1 [-3, +0] | 187, -3 [-8, +1], 1 / 4 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | -2 (183) |
| conf.rand.m1 | 102 / 84 / 2 / 4 | 3.2 | +2 [-3, +8] | -1 [-3, +0] | +0 [+0, +0] | 186, +2 [-3, +8], 5 / 3 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +1 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +14 (7.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +16 [+6, +27]; vs conf.rand.m1 +12 [+2, +22] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +7 on 113 questions; mean words 3.1 -> 5.8
KEY family=E judge=pro layer=11 A=incomplete B=not_met dR=+14 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+12 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 124 / 45 / 0 / 23 | 6.7 | +21 [+12, +31] | +3 [-4, +10] | -2 [-9, +5] | 161, +14 [+6, +23], 16 / 2 | 5 / 8 / 0 | 0 / 5 / 7 / 1 | +5 (95) |
| x.w.m1 | 89 / 77 / 2 / 24 | 2.5 | -14 [-23, -6] | +0 [-6, +6] | -1 [-7, +5] | 161, -10 [-18, -3], 3 / 13 | 5 / 5 / 1 | 4 / 1 / 0 / 5 | -9 (156) |
| x.md.p1 | 119 / 55 / 0 / 18 | 4.1 | +16 [+7, +25] | +8 [+2, +15] | -7 [-14, -1] | 164, +10 [+3, +18], 13 / 3 | 2 / 10 / 0 | 0 / 2 / 6 / 4 | +5 (131) |
| x.rand.p1 | 108 / 56 / 1 / 27 | 3.3 | +5 [+1, +10] | -2 [-5, +0] | +2 [+0, +5] | 164, +5 [+1, +10], 5 / 0 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | +5 (159) |
| x.rand.m1 | 99 / 65 / 3 / 25 | 3.3 | -4 [-9, +0] | -2 [-6, +2] | +0 [-3, +3] | 163, -1 [-5, +2], 1 / 2 | 3 / 1 / 2 | 3 / 0 / 0 / 1 | +1 (160) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 141 / 42 / 1 / 8; dR +7 [-3, +17]; answered-rate change +4.2 pp [+1.6, +7.3]; out / in 0 / 8; mean words 10.1 -> 13.9
- main effect: dR +21 (10.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +16 [+7, +25]; vs x.rand.m1 +25 [+16, +35] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.6 pp [-2.1, +5.2]; moved out / in 5 / 8 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 95 questions; mean words 3.2 -> 6.7
KEY family=E judge=allenai layer=12 A=not_met B=not_met dR=+21 dR_F=+7 dAns=+3 dAns_F=+8 rand_margin=+16

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 116 / 48 / 0 / 28 | +15 [+8, +23] | +4 [-3, +11] | -2 [-8, +4] | 156, +8, 9 / 1 | 4 / 8 / 0 | +5 (105) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 109 / 59 / 2 / 22 | -16 [-25, -8] | -8 [-14, -2] | +6 [+1, +12] | 167, -9, 3 / 12 | 9 / 1 / 2 | -9 (165) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 77 / 2 / 2 | +0 [-3, +3] | +1 [+0, +3] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 1 / 0 | +0 (185) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 122 / 67 / 1 / 2 | +11 [+2, +20] | +2 [+0, +5] | +0 [+0, +0] | 187, +10, 15 / 5 | 0 / 2 / 0 | +6 (146) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 110 / 75 / 3 / 4 | -1 [-7, +5] | -2 [-5, +0] | +2 [+0, +5] | 185, +1, 4 / 3 | 2 / 0 / 1 | +1 (177) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 110 / 76 / 3 / 3 | -1 [-5, +3] | -1 [-3, +0] | +1 [+0, +3] | 186, +0, 2 / 2 | 1 / 0 / 0 | +0 (183) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 135 / 55 / 0 / 2 | +24 [+15, +34] | +3 [+0, +7] | +0 [+0, +0] | 187, +23, 24 / 1 | 0 / 3 / 0 | +9 (114) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 134 / 56 / 0 / 2 | +23 [+14, +33] | +3 [+0, +7] | +0 [+0, +0] | 187, +22, 23 / 1 | 0 / 3 / 0 | +9 (116) |
| conf_E1.wq.m1 | 136 / 54 / 0 / 2 | 114 / 72 / 2 / 4 | -22 [-32, -13] | -4 [-8, -1] | +2 [+0, +5] | 186, -20, 2 / 22 | 4 / 0 / 2 | -19 (182) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 141 / 42 / 1 / 8 | +7 [-3, +17] | +8 [+3, +14] | -7 [-12, -2] | 175, +0, 9 / 9 | 0 / 8 / 0 | +3 (82) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 135 / 55 / 0 / 2 | 5.8 | +24 [+15, +34] | +3 [+0, +7] | +0 [+0, +0] | 187, +23 [+14, +33], 24 / 1 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +9 (114) |
| conf.md.p1 | 122 / 67 / 1 / 2 | 4.1 | +11 [+2, +20] | +2 [+0, +5] | +0 [+0, +0] | 187, +10 [+1, +19], 15 / 5 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +6 (146) |
| conf.rand.p1 | 110 / 76 / 3 / 3 | 3.1 | -1 [-5, +3] | -1 [-3, +0] | +1 [+0, +3] | 186, +0 [-4, +4], 2 / 2 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (183) |
| conf.rand.m1 | 110 / 75 / 3 / 4 | 3.1 | -1 [-7, +5] | -2 [-5, +0] | +2 [+0, +5] | 185, +1 [-4, +6], 4 / 3 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | +1 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +24 (12.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +25 [+16, +35]; vs conf.rand.m1 +25 [+15, +35] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.6 pp [+0.0, +3.6]; moved out / in 0 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +9 on 114 questions; mean words 3.1 -> 5.8
KEY family=E judge=allenai layer=12 A=incomplete B=not_met dR=+24 dR_F=NA dAns=+3 dAns_F=NA rand_margin=+25 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 109 / 56 / 4 / 23 | 6.7 | +10 [+0, +20] | +2 [-6, +10] | -2 [-9, +5] | 156, +3 [-6, +12], 11 / 8 | 7 / 9 / 2 | 1 / 6 / 8 / 1 | +3 (94) |
| x.w.m1 | 81 / 84 / 3 / 24 | 2.5 | -18 [-28, -9] | +2 [-5, +9] | -1 [-7, +5] | 158, -15 [-24, -7], 3 / 18 | 5 / 7 / 1 | 4 / 1 / 1 / 6 | -14 (152) |
| x.md.p1 | 104 / 65 / 5 / 18 | 4.1 | +5 [-4, +14] | +6 [-1, +13] | -7 [-14, -1] | 159, +0 [-7, +7], 7 / 7 | 4 / 10 / 2 | 0 / 4 / 5 / 5 | +2 (127) |
| x.rand.p1 | 98 / 61 / 6 / 27 | 3.3 | -1 [-6, +3] | -4 [-8, -1] | +2 [+0, +5] | 159, +0 [-4, +4], 2 / 2 | 4 / 0 / 2 | 1 / 3 / 0 / 0 | +1 (155) |
| x.rand.m1 | 95 / 68 / 4 / 25 | 3.3 | -4 [-8, -1] | +0 [-3, +3] | +0 [-3, +3] | 162, -3 [-7, +0], 0 / 3 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | -1 (157) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 125 / 56 / 3 / 8; dR +14 [+4, +24]; answered-rate change +6.2 pp [+3.1, +9.9]; out / in 0 / 12; mean words 10.1 -> 13.9
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +11 [+2, +21]; vs x.rand.m1 +14 [+3, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.0 pp [-3.1, +5.2]; moved out / in 7 / 9 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 94 questions; mean words 3.2 -> 6.7
KEY family=E judge=pro layer=12 A=not_met B=not_met dR=+10 dR_F=+14 dAns=+2 dAns_F=+12 rand_margin=+11

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 105 / 54 / 5 / 28 | +13 [+4, +22] | +2 [-5, +9] | -2 [-8, +4] | 151, +8, 11 / 3 | 6 / 8 / 2 | +5 (102) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 100 / 68 / 2 / 22 | -14 [-23, -5] | -3 [-10, +3] | +5 [-1, +11] | 164, -11, 3 / 14 | 7 / 4 / 0 | -10 (162) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 98 / 88 / 1 / 5 | -2 [-5, +0] | -1 [-3, +0] | +1 [+0, +3] | 186, -1, 0 / 1 | 1 / 0 / 0 | +0 (185) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 105 / 80 / 2 / 5 | +5 [-4, +14] | -2 [-5, +0] | +1 [+0, +3] | 185, +6, 13 / 7 | 2 / 0 / 1 | +4 (146) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 98 / 87 / 2 / 5 | -2 [-7, +3] | -2 [-5, +0] | +1 [+0, +3] | 185, -2, 2 / 4 | 2 / 0 / 1 | +0 (177) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 101 / 86 / 1 / 4 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +1, 2 / 1 | 0 / 0 / 0 | +1 (184) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 112 / 72 / 3 / 5 | +12 [+2, +22] | -3 [-7, +0] | +1 [+0, +3] | 184, +14, 19 / 5 | 3 / 0 / 2 | +6 (112) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 111 / 74 / 2 / 5 | +11 [+2, +21] | -2 [-5, +0] | +1 [+0, +3] | 185, +12, 17 / 5 | 2 / 0 / 1 | +6 (114) |
| conf_E1.wq.m1 | 117 / 68 / 2 / 5 | 105 / 80 / 2 / 5 | -12 [-23, -2] | +0 [-4, +4] | +0 [-3, +3] | 183, -12, 7 / 19 | 2 / 2 / 1 | -11 (179) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 125 / 56 / 3 / 8 | +14 [+4, +24] | +12 [+6, +19] | -7 [-12, -3] | 169, +7, 13 / 6 | 0 / 12 / 0 | +4 (80) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 112 / 72 / 3 / 5 | 5.8 | +12 [+2, +22] | -3 [-7, +0] | +1 [+0, +3] | 184, +14 [+5, +24], 19 / 5 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +6 (112) |
| conf.md.p1 | 105 / 80 / 2 / 5 | 4.1 | +5 [-4, +14] | -2 [-5, +0] | +1 [+0, +3] | 185, +6 [-3, +15], 13 / 7 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +4 (146) |
| conf.rand.p1 | 101 / 86 / 1 / 4 | 3.1 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +1 [-2, +5], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (184) |
| conf.rand.m1 | 98 / 87 / 2 / 5 | 3.1 | -2 [-7, +3] | -2 [-5, +0] | +1 [+0, +3] | 185, -2 [-7, +3], 2 / 4 | 2 / 0 / 1 | 0 / 2 / 0 / 0 | +0 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +12 (6.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +11 [+1, +21]; vs conf.rand.m1 +14 [+5, +24] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 112 questions; mean words 3.1 -> 5.8
KEY family=E judge=pro layer=12 A=incomplete B=not_met dR=+12 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+11 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 123 / 45 / 0 / 24 | 7.0 | +20 [+11, +30] | +2 [-5, +9] | -1 [-8, +5] | 161, +14 [+6, +23], 17 / 3 | 5 / 7 / 0 | 0 / 5 / 6 / 1 | +5 (92) |
| x.w.m1 | 90 / 78 / 2 / 22 | 2.3 | -13 [-22, -5] | +2 [-5, +9] | -3 [-10, +3] | 161, -9 [-17, -2], 3 / 12 | 5 / 7 / 1 | 4 / 1 / 0 / 7 | -7 (157) |
| x.md.p1 | 115 / 54 / 1 / 22 | 4.0 | +12 [+4, +21] | +3 [-3, +9] | -3 [-8, +2] | 163, +8 [+1, +15], 11 / 3 | 3 / 6 / 1 | 1 / 2 / 5 / 1 | +4 (135) |
| x.rand.p1 | 103 / 62 / 2 / 25 | 3.4 | +0 [-4, +4] | -1 [-3, +0] | +0 [+0, +0] | 165, +1 [-2, +5], 2 / 1 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +1 (160) |
| x.rand.m1 | 102 / 64 / 1 / 25 | 3.3 | -1 [-6, +3] | +0 [-4, +4] | +0 [-4, +4] | 164, +0 [-4, +4], 2 / 2 | 2 / 2 / 0 | 1 / 1 / 0 / 2 | +0 (161) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 140 / 42 / 2 / 8; dR +6 [-2, +15]; answered-rate change +3.6 pp [+1.0, +6.2]; out / in 0 / 7; mean words 10.1 -> 14.0
- main effect: dR +20 (10.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +20 [+11, +30]; vs x.rand.m1 +21 [+12, +31] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.0 pp [-2.6, +4.7]; moved out / in 5 / 7 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 92 questions; mean words 3.2 -> 7.0
KEY family=E judge=allenai layer=13 A=not_met B=not_met dR=+20 dR_F=+6 dAns=+2 dAns_F=+7 rand_margin=+20

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 13

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 113 / 50 / 0 / 29 | +12 [+5, +20] | +3 [-3, +9] | -1 [-7, +5] | 156, +6, 8 / 2 | 4 / 7 / 0 | +4 (104) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 109 / 59 / 3 / 21 | -16 [-26, -7] | -8 [-15, -2] | +5 [-1, +11] | 166, -9, 3 / 12 | 10 / 2 / 3 | -9 (165) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 109 / 79 / 2 / 2 | -2 [-6, +2] | +1 [+0, +3] | +0 [+0, +0] | 187, -3, 0 / 3 | 0 / 1 / 0 | -1 (184) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 120 / 68 / 2 / 2 | +9 [+1, +17] | +1 [-2, +5] | +0 [+0, +0] | 186, +9, 13 / 4 | 1 / 2 / 1 | +7 (152) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 114 / 73 / 3 / 2 | +3 [-1, +8] | +0 [+0, +0] | +0 [+0, +0] | 187, +3, 4 / 1 | 0 / 0 / 0 | +3 (183) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 110 / 76 / 3 / 3 | -1 [-6, +4] | -1 [-3, +0] | +1 [+0, +3] | 186, +0, 3 / 3 | 1 / 0 / 1 | +0 (178) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 131 / 59 / 0 / 2 | +20 [+11, +30] | +3 [+0, +7] | +0 [+0, +0] | 187, +19, 22 / 3 | 0 / 3 / 0 | +6 (116) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 131 / 59 / 0 / 2 | +20 [+11, +30] | +3 [+0, +7] | +0 [+0, +0] | 187, +19, 22 / 3 | 0 / 3 / 0 | +7 (118) |
| conf_E1.wq.m1 | 136 / 54 / 0 / 2 | 118 / 68 / 2 / 4 | -18 [-28, -9] | -4 [-8, -1] | +2 [+0, +5] | 186, -16, 3 / 19 | 4 / 0 / 2 | -16 (183) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 140 / 42 / 2 / 8 | +6 [-2, +15] | +7 [+2, +12] | -7 [-12, -2] | 175, +0, 7 / 7 | 0 / 7 / 0 | -1 (83) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 131 / 59 / 0 / 2 | 6.1 | +20 [+11, +30] | +3 [+0, +7] | +0 [+0, +0] | 187, +19 [+10, +29], 22 / 3 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +6 (116) |
| conf.md.p1 | 120 / 68 / 2 / 2 | 3.9 | +9 [+1, +17] | +1 [-2, +5] | +0 [+0, +0] | 186, +9 [+1, +17], 13 / 4 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +7 (152) |
| conf.rand.p1 | 110 / 76 / 3 / 3 | 3.2 | -1 [-6, +4] | -1 [-3, +0] | +1 [+0, +3] | 186, +0 [-5, +5], 3 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (178) |
| conf.rand.m1 | 114 / 73 / 3 / 2 | 3.1 | +3 [-1, +8] | +0 [+0, +0] | +0 [+0, +0] | 187, +3 [-1, +8], 4 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +3 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +20 (10.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +21 [+11, +31]; vs conf.rand.m1 +17 [+6, +28] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.6 pp [+0.0, +3.6]; moved out / in 0 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 116 questions; mean words 3.1 -> 6.1
KEY family=E judge=allenai layer=13 A=incomplete B=not_met dR=+20 dR_F=NA dAns=+3 dAns_F=NA rand_margin=+17 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 105 / 59 / 4 / 24 | 7.0 | +6 [-4, +17] | +1 [-7, +9] | -1 [-8, +5] | 156, +0 [-9, +9], 10 / 10 | 7 / 8 / 2 | 1 / 6 / 7 / 1 | +3 (90) |
| x.w.m1 | 83 / 84 / 3 / 22 | 2.3 | -16 [-26, -7] | +4 [-3, +11] | -3 [-10, +3] | 158, -13 [-22, -5], 3 / 16 | 5 / 9 / 1 | 4 / 1 / 1 / 8 | -11 (153) |
| x.md.p1 | 106 / 60 / 4 / 22 | 4.0 | +7 [-1, +16] | +3 [-3, +9] | -3 [-8, +2] | 160, +3 [-5, +11], 9 / 6 | 3 / 6 / 1 | 0 / 3 / 4 / 2 | +3 (132) |
| x.rand.p1 | 95 / 67 / 5 / 25 | 3.4 | -4 [-9, +0] | -1 [-3, +0] | +0 [+0, +0] | 162, -3 [-8, +1], 1 / 4 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -2 (157) |
| x.rand.m1 | 97 / 66 / 4 / 25 | 3.3 | -2 [-6, +2] | +0 [-4, +4] | +0 [-4, +4] | 161, -1 [-4, +2], 1 / 2 | 2 / 2 / 0 | 1 / 1 / 0 / 2 | +0 (158) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 116 / 66 / 2 / 8; dR +5 [-5, +15]; answered-rate change +6.8 pp [+3.6, +10.4]; out / in 0 / 13; mean words 10.1 -> 14.0
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +10 [+0, +20]; vs x.rand.m1 +8 [-2, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.5 pp [-3.6, +4.7]; moved out / in 7 / 8 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 90 questions; mean words 3.2 -> 7.0
KEY family=E judge=pro layer=13 A=not_met B=not_met dR=+6 dR_F=+5 dAns=+1 dAns_F=+13 rand_margin=+8

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 13

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 103 / 56 / 4 / 29 | +11 [+3, +20] | +2 [-5, +9] | -1 [-7, +5] | 152, +6, 10 / 4 | 5 / 7 / 1 | +3 (101) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 101 / 68 / 1 / 22 | -13 [-22, -4] | -2 [-9, +5] | +5 [-1, +11] | 164, -11, 3 / 14 | 7 / 5 / 0 | -10 (163) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 99 / 88 / 1 / 4 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 1 / 2 | 0 / 0 / 0 | +1 (184) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 105 / 80 / 2 / 5 | +5 [-3, +13] | -2 [-5, +0] | +1 [+0, +3] | 185, +6, 12 / 6 | 2 / 0 / 1 | +5 (151) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 102 / 86 / 1 / 3 | +2 [-3, +7] | +1 [+0, +3] | -1 [-3, +0] | 187, +1, 3 / 2 | 0 / 1 / 0 | +1 (184) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 98 / 88 / 1 / 5 | -2 [-8, +3] | -1 [-3, +0] | +1 [+0, +3] | 186, -1, 3 / 4 | 1 / 0 / 0 | +0 (178) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 109 / 75 / 3 / 5 | +9 [-1, +19] | -3 [-7, +0] | +1 [+0, +3] | 184, +11, 17 / 6 | 3 / 0 / 2 | +5 (114) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 112 / 74 / 1 / 5 | +12 [+3, +22] | -1 [-3, +0] | +1 [+0, +3] | 186, +13, 18 / 5 | 1 / 0 / 0 | +6 (116) |
| conf_E1.wq.m1 | 117 / 68 / 2 / 5 | 109 / 76 / 2 / 5 | -8 [-18, +2] | +0 [-4, +4] | +0 [-3, +3] | 183, -8, 8 / 16 | 2 / 2 / 1 | -8 (180) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 116 / 66 / 2 / 8 | +5 [-5, +15] | +13 [+7, +20] | -7 [-12, -3] | 169, -2, 8 / 10 | 0 / 13 / 0 | -1 (80) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 109 / 75 / 3 / 5 | 6.1 | +9 [-1, +19] | -3 [-7, +0] | +1 [+0, +3] | 184, +11 [+2, +20], 17 / 6 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +5 (114) |
| conf.md.p1 | 105 / 80 / 2 / 5 | 3.9 | +5 [-3, +13] | -2 [-5, +0] | +1 [+0, +3] | 185, +6 [-2, +14], 12 / 6 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +5 (151) |
| conf.rand.p1 | 98 / 88 / 1 / 5 | 3.2 | -2 [-8, +3] | -1 [-3, +0] | +1 [+0, +3] | 186, -1 [-6, +4], 3 / 4 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (178) |
| conf.rand.m1 | 102 / 86 / 1 / 3 | 3.1 | +2 [-3, +7] | +1 [+0, +3] | -1 [-3, +0] | 187, +1 [-3, +5], 3 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +9 (4.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +11 [+2, +20]; vs conf.rand.m1 +7 [-3, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 114 questions; mean words 3.1 -> 6.1
KEY family=E judge=pro layer=13 A=incomplete B=not_met dR=+9 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+7 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 123 / 44 / 0 / 25 | 6.2 | +20 [+11, +30] | +1 [-6, +9] | +0 [-7, +7] | 159, +13 [+5, +22], 16 / 3 | 7 / 8 / 0 | 0 / 7 / 7 / 1 | +5 (101) |
| x.w.m1 | 92 / 74 / 2 / 24 | 2.4 | -11 [-19, -3] | +0 [-6, +6] | -1 [-7, +5] | 161, -7 [-14, +0], 3 / 10 | 5 / 5 / 1 | 4 / 1 / 0 / 5 | -6 (156) |
| x.md.p1 | 106 / 63 / 2 / 21 | 3.7 | +3 [-4, +10] | +3 [-3, +10] | -4 [-10, +1] | 162, +2 [-3, +8], 5 / 3 | 4 / 7 / 2 | 2 / 2 / 3 / 4 | -1 (138) |
| x.rand.p1 | 103 / 61 / 1 / 27 | 3.5 | +0 [-4, +4] | -2 [-6, +2] | +2 [-2, +6] | 163, +0 [-4, +4], 2 / 2 | 3 / 1 / 0 | 0 / 3 / 0 / 1 | +0 (154) |
| x.rand.m1 | 105 / 60 / 2 / 25 | 3.1 | +2 [-3, +7] | -1 [-3, +0] | +0 [+0, +0] | 165, +3 [-1, +8], 4 / 1 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +3 (164) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 140 / 43 / 1 / 8; dR +6 [-2, +14]; answered-rate change +4.2 pp [+1.6, +7.3]; out / in 0 / 8; mean words 10.1 -> 13.7
- main effect: dR +20 (10.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +20 [+11, +30]; vs x.rand.m1 +18 [+9, +28] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.5 pp [-3.1, +4.7]; moved out / in 7 / 8 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 101 questions; mean words 3.2 -> 6.2
KEY family=E judge=allenai layer=14 A=not_met B=not_met dR=+20 dR_F=+6 dAns=+1 dAns_F=+8 rand_margin=+18

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 14

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 112 / 48 / 0 / 32 | +11 [+4, +19] | +0 [-7, +7] | +2 [-5, +9] | 153, +5, 7 / 2 | 7 / 7 / 0 | +3 (101) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 110 / 60 / 3 / 19 | -15 [-25, -6] | -6 [-13, +1] | +3 [-3, +10] | 166, -8, 3 / 11 | 10 / 4 / 3 | -8 (165) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 76 / 3 / 2 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 1 / 1 | 0 / 0 / 0 | +0 (184) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 110 / 78 / 2 / 2 | -1 [-8, +5] | +1 [-2, +5] | +0 [+0, +0] | 186, -1, 4 / 5 | 1 / 2 / 1 | +0 (155) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 114 / 72 / 2 / 4 | +3 [-2, +8] | -1 [-3, +0] | +2 [+0, +5] | 186, +4, 5 / 1 | 1 / 0 / 0 | +4 (182) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 111 / 77 / 2 / 2 | +0 [-5, +5] | +1 [+0, +3] | +0 [+0, +0] | 187, -1, 2 / 3 | 0 / 1 / 0 | +0 (176) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 133 / 57 / 0 / 2 | +22 [+13, +32] | +3 [+0, +7] | +0 [+0, +0] | 187, +21, 23 / 2 | 0 / 3 / 0 | +7 (114) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 133 / 57 / 0 / 2 | +22 [+12, +32] | +3 [+0, +7] | +0 [+0, +0] | 187, +21, 24 / 3 | 0 / 3 / 0 | +7 (114) |
| conf_E1.wq.m1 | 136 / 54 / 0 / 2 | 114 / 72 / 2 / 4 | -22 [-32, -12] | -4 [-8, -1] | +2 [+0, +5] | 186, -20, 3 / 23 | 4 / 0 / 2 | -20 (182) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 140 / 43 / 1 / 8 | +6 [-2, +14] | +8 [+3, +14] | -7 [-12, -2] | 175, +0, 6 / 6 | 0 / 8 / 0 | +1 (88) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 133 / 57 / 0 / 2 | 6.2 | +22 [+13, +32] | +3 [+0, +7] | +0 [+0, +0] | 187, +21 [+12, +31], 23 / 2 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +7 (114) |
| conf.md.p1 | 110 / 78 / 2 / 2 | 3.8 | -1 [-8, +5] | +1 [-2, +5] | +0 [+0, +0] | 186, -1 [-7, +5], 4 / 5 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +0 (155) |
| conf.rand.p1 | 111 / 77 / 2 / 2 | 3.4 | +0 [-5, +5] | +1 [+0, +3] | +0 [+0, +0] | 187, -1 [-6, +3], 2 / 3 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (176) |
| conf.rand.m1 | 114 / 72 / 2 / 4 | 2.9 | +3 [-2, +8] | -1 [-3, +0] | +2 [+0, +5] | 186, +4 [+0, +9], 5 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +4 (182) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +22 (11.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +22 [+13, +32]; vs conf.rand.m1 +19 [+11, +28] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.6 pp [+0.0, +3.6]; moved out / in 0 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +7 on 114 questions; mean words 3.1 -> 6.2
KEY family=E judge=allenai layer=14 A=incomplete B=not_met dR=+22 dR_F=NA dAns=+3 dAns_F=NA rand_margin=+19 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 107 / 56 / 4 / 25 | 6.2 | +8 [-2, +19] | +0 [-8, +9] | +0 [-7, +7] | 154, +3 [-6, +12], 12 / 9 | 9 / 9 / 2 | 1 / 8 / 6 / 3 | +3 (98) |
| x.w.m1 | 85 / 80 / 3 / 24 | 2.4 | -14 [-24, -5] | +2 [-5, +9] | -1 [-7, +5] | 158, -11 [-19, -3], 3 / 14 | 5 / 7 / 1 | 4 / 1 / 1 / 6 | -10 (152) |
| x.md.p1 | 95 / 73 / 3 / 21 | 3.7 | -4 [-12, +4] | +5 [-1, +12] | -4 [-10, +1] | 160, -8 [-15, -2], 2 / 10 | 3 / 8 / 1 | 0 / 3 / 4 / 4 | -4 (136) |
| x.rand.p1 | 95 / 66 / 4 / 27 | 3.5 | -4 [-9, +0] | -2 [-6, +2] | +2 [-2, +6] | 160, -4 [-9, +0], 1 / 5 | 3 / 1 / 0 | 0 / 3 / 0 / 1 | +0 (151) |
| x.rand.m1 | 101 / 64 / 2 / 25 | 3.1 | +2 [-3, +7] | +2 [+0, +5] | +0 [+0, +0] | 163, +1 [-3, +5], 3 / 2 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +1 (161) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 117 / 64 / 3 / 8; dR +6 [-3, +15]; answered-rate change +6.2 pp [+3.1, +9.9]; out / in 0 / 12; mean words 10.1 -> 13.7
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +12 [+1, +23]; vs x.rand.m1 +6 [-3, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-4.2, +4.7]; moved out / in 9 / 9 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 98 questions; mean words 3.2 -> 6.2
KEY family=E judge=pro layer=14 A=not_met B=not_met dR=+8 dR_F=+6 dAns=+0 dAns_F=+12 rand_margin=+6

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 14

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 102 / 55 / 3 / 32 | +10 [+2, +18] | +0 [-8, +8] | +2 [-5, +9] | 149, +4, 8 / 4 | 8 / 8 / 1 | +2 (99) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 102 / 69 / 2 / 19 | -12 [-21, -3] | +0 [-7, +7] | +2 [-5, +9] | 164, -10, 3 / 13 | 7 / 7 / 0 | -9 (163) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 100 / 87 / 1 / 4 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 1 / 1 | 0 / 0 / 0 | +0 (184) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 96 / 90 / 2 / 4 | -4 [-12, +3] | -1 [-5, +2] | +0 [-3, +3] | 185, -4, 4 / 8 | 2 / 1 / 1 | +0 (155) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 107 / 79 / 1 / 5 | +7 [+2, +13] | -1 [-3, +0] | +1 [+0, +3] | 186, +7, 7 / 0 | 1 / 0 / 0 | +7 (182) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 96 / 90 / 2 / 4 | -4 [-9, +0] | -1 [-5, +2] | +0 [-3, +3] | 185, -4, 0 / 4 | 2 / 1 / 1 | -1 (177) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 107 / 77 / 3 / 5 | +7 [-2, +17] | -3 [-7, +0] | +1 [+0, +3] | 184, +9, 15 / 6 | 3 / 0 / 2 | +4 (112) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 111 / 75 / 1 / 5 | +11 [+2, +20] | -1 [-3, +0] | +1 [+0, +3] | 186, +12, 17 / 5 | 1 / 0 / 0 | +5 (112) |
| conf_E1.wq.m1 | 117 / 68 / 2 / 5 | 104 / 80 / 2 / 6 | -13 [-23, -3] | -1 [-5, +2] | +1 [+0, +3] | 183, -12, 6 / 18 | 2 / 1 / 1 | -12 (179) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 117 / 64 / 3 / 8 | +6 [-3, +15] | +12 [+6, +19] | -7 [-12, -3] | 169, +0, 8 / 8 | 0 / 12 / 0 | +3 (85) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 107 / 77 / 3 / 5 | 6.2 | +7 [-2, +17] | -3 [-7, +0] | +1 [+0, +3] | 184, +9 [+0, +18], 15 / 6 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +4 (112) |
| conf.md.p1 | 96 / 90 / 2 / 4 | 3.8 | -4 [-12, +3] | -1 [-5, +2] | +0 [-3, +3] | 185, -4 [-11, +2], 4 / 8 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +0 (155) |
| conf.rand.p1 | 96 / 90 / 2 / 4 | 3.4 | -4 [-9, +0] | -1 [-5, +2] | +0 [-3, +3] | 185, -4 [-8, -1], 0 / 4 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -1 (177) |
| conf.rand.m1 | 107 / 79 / 1 / 5 | 2.9 | +7 [+2, +13] | -1 [-3, +0] | +1 [+0, +3] | 186, +7 [+2, +13], 7 / 0 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +7 (182) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +11 [+2, +21]; vs conf.rand.m1 +0 [-9, +9] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +4 on 112 questions; mean words 3.1 -> 6.2
KEY family=E judge=pro layer=14 A=not_met B=not_met dR=+7 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+0 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 123 / 45 / 0 / 24 | 6.9 | +20 [+11, +30] | +2 [-5, +9] | -1 [-8, +5] | 161, +14 [+6, +23], 17 / 3 | 5 / 7 / 0 | 0 / 5 / 6 / 1 | +5 (95) |
| x.w.m1 | 91 / 78 / 2 / 21 | 2.3 | -12 [-20, -4] | +3 [-4, +10] | -4 [-11, +2] | 161, -8 [-15, -1], 3 / 11 | 5 / 8 / 1 | 4 / 1 / 0 / 8 | -7 (156) |
| x.md.p1 | 121 / 50 / 0 / 21 | 8.2 | +18 [+9, +28] | +5 [-1, +12] | -4 [-10, +2] | 163, +11 [+3, +19], 14 / 3 | 3 / 8 / 0 | 0 / 3 / 7 / 1 | +5 (85) |
| x.rand.p1 | 99 / 68 / 2 / 23 | 3.2 | -4 [-9, +0] | +1 [-2, +5] | -2 [-5, +0] | 165, -3 [-8, +1], 1 / 4 | 1 / 2 / 1 | 1 / 0 / 0 / 2 | -1 (159) |
| x.rand.m1 | 100 / 58 / 1 / 33 | 3.4 | -3 [-8, +2] | -8 [-14, -3] | +8 [+3, +14] | 158, +0 [-4, +4], 2 / 2 | 8 / 0 / 1 | 3 / 5 / 0 / 0 | +0 (152) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 140 / 42 / 1 / 9; dR +6 [-2, +14]; answered-rate change +3.6 pp [+1.0, +6.8]; out / in 0 / 7; mean words 10.1 -> 12.8
- main effect: dR +20 (10.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +24 [+14, +35]; vs x.rand.m1 +23 [+13, +33] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.0 pp [-2.6, +4.7]; moved out / in 5 / 7 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 95 questions; mean words 3.2 -> 6.9
KEY family=E judge=allenai layer=15 A=not_met B=not_met dR=+20 dR_F=+6 dAns=+2 dAns_F=+7 rand_margin=+23

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 15

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 112 / 51 / 0 / 29 | +11 [+4, +19] | +3 [-4, +10] | -1 [-8, +5] | 155, +4, 6 / 2 | 5 / 8 / 0 | +1 (100) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 108 / 62 / 2 / 20 | -17 [-27, -8] | -6 [-13, +1] | +4 [-3, +11] | 166, -9, 3 / 12 | 10 / 4 / 2 | -8 (163) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 110 / 77 / 3 / 2 | -1 [-4, +2] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 1 / 2 | 0 / 0 / 0 | +0 (183) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 124 / 66 / 0 / 2 | +13 [+3, +23] | +3 [+0, +7] | +0 [+0, +0] | 187, +12, 20 / 8 | 0 / 3 / 0 | +5 (103) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 114 / 73 / 3 / 2 | +3 [-2, +8] | +0 [-3, +3] | +0 [+0, +0] | 186, +3, 4 / 1 | 1 / 1 / 1 | +1 (179) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 106 / 80 / 4 / 2 | -5 [-10, +0] | -1 [-3, +0] | +0 [+0, +0] | 186, -4, 1 / 5 | 1 / 0 / 1 | -2 (175) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 133 / 56 / 1 / 2 | +22 [+12, +32] | +2 [-2, +6] | +0 [+0, +0] | 186, +22, 24 / 2 | 1 / 3 / 1 | +7 (112) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 135 / 54 / 1 / 2 | +24 [+14, +34] | +2 [-2, +6] | +0 [+0, +0] | 186, +24, 26 / 2 | 1 / 3 / 1 | +7 (112) |
| conf_E1.wq.m1 | 136 / 54 / 0 / 2 | 114 / 71 / 3 / 4 | -22 [-32, -13] | -5 [-10, -1] | +2 [+0, +5] | 185, -20, 2 / 22 | 5 / 0 / 3 | -20 (183) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 140 / 42 / 1 / 9 | +6 [-2, +14] | +7 [+2, +13] | -6 [-11, -2] | 175, +0, 6 / 6 | 0 / 7 / 0 | +3 (92) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 133 / 56 / 1 / 2 | 6.3 | +22 [+12, +32] | +2 [-2, +6] | +0 [+0, +0] | 186, +22 [+13, +32], 24 / 2 | 1 / 3 / 1 | 1 / 0 / 1 / 2 | +7 (112) |
| conf.md.p1 | 124 / 66 / 0 / 2 | 7.5 | +13 [+3, +23] | +3 [+0, +7] | +0 [+0, +0] | 187, +12 [+2, +22], 20 / 8 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +5 (103) |
| conf.rand.p1 | 106 / 80 / 4 / 2 | 3.3 | -5 [-10, +0] | -1 [-3, +0] | +0 [+0, +0] | 186, -4 [-9, +0], 1 / 5 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -2 (175) |
| conf.rand.m1 | 114 / 73 / 3 / 2 | 3.2 | +3 [-2, +8] | +0 [-3, +3] | +0 [+0, +0] | 186, +3 [-1, +8], 4 / 1 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +1 (179) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +22 (11.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +27 [+17, +38]; vs conf.rand.m1 +19 [+10, +28] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.0 pp [-1.0, +3.1]; moved out / in 1 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +7 on 112 questions; mean words 3.1 -> 6.3
KEY family=E judge=allenai layer=15 A=incomplete B=not_met dR=+22 dR_F=NA dAns=+2 dAns_F=NA rand_margin=+19 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 109 / 56 / 3 / 24 | 6.9 | +10 [-1, +21] | +2 [-5, +10] | -1 [-8, +5] | 157, +5 [-4, +14], 14 / 9 | 6 / 8 / 1 | 1 / 5 / 6 / 2 | +2 (92) |
| x.w.m1 | 86 / 83 / 2 / 21 | 2.3 | -13 [-23, -4] | +6 [-1, +13] | -4 [-11, +2] | 159, -11 [-20, -3], 4 / 15 | 4 / 10 / 0 | 3 / 1 / 1 / 9 | -10 (153) |
| x.md.p1 | 110 / 59 / 2 / 21 | 8.2 | +11 [+0, +22] | +6 [+0, +13] | -4 [-10, +2] | 160, +5 [-5, +15], 15 / 10 | 3 / 9 / 0 | 0 / 3 / 6 / 3 | +3 (83) |
| x.rand.p1 | 97 / 69 / 3 / 23 | 3.2 | -2 [-7, +3] | +3 [+0, +7] | -2 [-5, +0] | 163, -3 [-8, +1], 1 / 4 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | -1 (156) |
| x.rand.m1 | 94 / 60 / 5 / 33 | 3.4 | -5 [-12, +1] | -9 [-15, -4] | +8 [+3, +14] | 154, -2 [-8, +3], 3 / 5 | 9 / 0 / 1 | 3 / 6 / 0 / 0 | +0 (148) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 121 / 59 / 2 / 10; dR +10 [+2, +19]; answered-rate change +5.7 pp [+2.6, +9.4]; out / in 0 / 11; mean words 10.1 -> 12.8
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +12 [+1, +23]; vs x.rand.m1 +15 [+4, +26] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.0 pp [-2.6, +5.2]; moved out / in 6 / 8 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 92 questions; mean words 3.2 -> 6.9
KEY family=E judge=pro layer=15 A=not_met B=not_met dR=+10 dR_F=+10 dAns=+2 dAns_F=+11 rand_margin=+12

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 15

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 97 / 63 / 3 / 29 | +5 [-4, +14] | +3 [-4, +11] | -1 [-8, +5] | 151, +0, 8 / 8 | 6 / 9 / 1 | +1 (98) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 100 / 70 / 2 / 20 | -14 [-24, -5] | -1 [-9, +6] | +3 [-4, +10] | 163, -12, 3 / 15 | 8 / 7 / 0 | -10 (160) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 99 / 88 / 1 / 4 | -1 [-4, +2] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 1 / 2 | 0 / 0 / 0 | +0 (183) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 102 / 83 / 2 / 5 | +2 [-9, +13] | -2 [-5, +0] | +1 [+0, +3] | 185, +4, 17 / 13 | 2 / 0 / 1 | +3 (101) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 102 / 85 / 2 / 3 | +2 [-2, +6] | +0 [-3, +3] | -1 [-3, +0] | 186, +1, 2 / 1 | 1 / 1 / 1 | +1 (181) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 95 / 90 / 3 / 4 | -5 [-10, +0] | -2 [-5, +0] | +0 [+0, +0] | 185, -3, 1 / 4 | 2 / 0 / 2 | -1 (175) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 107 / 77 / 3 / 5 | +7 [-2, +17] | -3 [-7, +0] | +1 [+0, +3] | 184, +9, 15 / 6 | 3 / 0 / 2 | +4 (110) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 113 / 73 / 1 / 5 | +13 [+3, +23] | -1 [-3, +0] | +1 [+0, +3] | 186, +14, 19 / 5 | 1 / 0 / 0 | +5 (110) |
| conf_E1.wq.m1 | 117 / 68 / 2 / 5 | 103 / 81 / 2 / 6 | -14 [-24, -4] | -1 [-5, +2] | +1 [+0, +3] | 183, -13, 6 / 19 | 2 / 1 / 1 | -13 (181) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 121 / 59 / 2 / 10 | +10 [+2, +19] | +11 [+5, +18] | -5 [-10, -1] | 169, +4, 9 / 5 | 0 / 11 / 0 | +2 (90) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 107 / 77 / 3 / 5 | 6.3 | +7 [-2, +17] | -3 [-7, +0] | +1 [+0, +3] | 184, +9 [+0, +18], 15 / 6 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +4 (110) |
| conf.md.p1 | 102 / 83 / 2 / 5 | 7.5 | +2 [-9, +13] | -2 [-5, +0] | +1 [+0, +3] | 185, +4 [-7, +15], 17 / 13 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | +3 (101) |
| conf.rand.p1 | 95 / 90 / 3 / 4 | 3.3 | -5 [-10, +0] | -2 [-5, +0] | +0 [+0, +0] | 185, -3 [-8, +1], 1 / 4 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | -1 (175) |
| conf.rand.m1 | 102 / 85 / 2 / 3 | 3.2 | +2 [-2, +6] | +0 [-3, +3] | -1 [-3, +0] | 186, +1 [-2, +5], 2 / 1 | 1 / 1 / 1 | 0 / 1 / 1 / 0 | +1 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +12 [+3, +22]; vs conf.rand.m1 +5 [-4, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +4 on 110 questions; mean words 3.1 -> 6.3
KEY family=E judge=pro layer=15 A=incomplete B=not_met dR=+7 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+5 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 125 / 44 / 1 / 22 | 6.8 | +22 [+12, +32] | +3 [-4, +10] | -3 [-10, +3] | 161, +15 [+7, +24], 17 / 2 | 5 / 8 / 1 | 1 / 4 / 8 / 0 | +5 (103) |
| x.w.m1 | 94 / 72 / 3 / 23 | 2.3 | -9 [-17, -1] | +0 [-8, +8] | -2 [-9, +5] | 158, -4 [-10, +2], 3 / 7 | 8 / 8 / 2 | 6 / 2 / 1 / 7 | -4 (156) |
| x.md.p1 | 117 / 51 / 1 / 23 | 4.9 | +14 [+5, +24] | +2 [-3, +8] | -2 [-7, +3] | 163, +10 [+2, +18], 14 / 4 | 3 / 5 / 1 | 1 / 2 / 5 / 0 | +5 (125) |
| x.rand.p1 | 103 / 63 / 2 / 24 | 3.3 | +0 [-5, +5] | +0 [-3, +3] | -1 [-3, +0] | 165, +1 [-3, +6], 3 / 2 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (161) |
| x.rand.m1 | 101 / 60 / 1 / 30 | 3.2 | -2 [-6, +2] | -5 [-10, -1] | +5 [+1, +10] | 161, -1 [-5, +2], 1 / 2 | 5 / 0 / 0 | 1 / 4 / 0 / 0 | +0 (156) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 140 / 41 / 1 / 10; dR +6 [-2, +14]; answered-rate change +3.1 pp [+1.0, +5.7]; out / in 0 / 6; mean words 10.1 -> 13.0
- main effect: dR +22 (11.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +22 [+13, +32]; vs x.rand.m1 +24 [+15, +34] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.6 pp [-2.1, +5.2]; moved out / in 5 / 8 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 103 questions; mean words 3.2 -> 6.8
KEY family=E judge=allenai layer=18 A=not_met B=not_met dR=+22 dR_F=+6 dAns=+3 dAns_F=+6 rand_margin=+22

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 18

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 113 / 52 / 1 / 26 | +12 [+3, +21] | +5 [-2, +12] | -4 [-10, +2] | 156, +6, 10 / 4 | 4 / 9 / 1 | +2 (103) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 110 / 57 / 1 / 24 | -15 [-25, -5] | -9 [-16, -2] | +8 [+2, +15] | 165, -7, 4 / 11 | 11 / 2 / 1 | -7 (165) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 76 / 2 / 3 | +0 [+0, +0] | +0 [+0, +0] | +1 [+0, +3] | 187, +0, 0 / 0 | 0 / 0 / 0 | +0 (187) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 125 / 64 / 1 / 2 | +14 [+4, +24] | +2 [-2, +6] | +0 [+0, +0] | 186, +13, 18 / 5 | 1 / 3 / 1 | +6 (141) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 115 / 73 / 2 / 2 | +4 [+0, +9] | +1 [+0, +3] | +0 [+0, +0] | 187, +3, 4 / 1 | 0 / 1 / 0 | +2 (176) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 108 / 77 / 3 / 4 | -3 [-10, +3] | -2 [-5, +0] | +2 [+0, +5] | 185, -1, 4 / 5 | 2 / 0 / 1 | -1 (173) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 130 / 59 / 1 / 2 | +19 [+9, +29] | +2 [-2, +6] | +0 [+0, +0] | 186, +19, 23 / 4 | 1 / 3 / 1 | +5 (119) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 130 / 59 / 1 / 2 | +19 [+9, +29] | +2 [-2, +6] | +0 [+0, +0] | 186, +19, 22 / 3 | 1 / 3 / 1 | +6 (119) |
| conf_E1.wq.m1 | 136 / 54 / 0 / 2 | 117 / 68 / 3 / 4 | -19 [-29, -9] | -5 [-10, -1] | +2 [+0, +5] | 185, -17, 4 / 21 | 5 / 0 / 3 | -17 (182) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 140 / 41 / 1 / 10 | +6 [-2, +14] | +6 [+2, +11] | -5 [-10, -1] | 175, +1, 7 / 6 | 0 / 6 / 0 | +2 (103) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 130 / 59 / 1 / 2 | 5.9 | +19 [+9, +29] | +2 [-2, +6] | +0 [+0, +0] | 186, +19 [+9, +29], 23 / 4 | 1 / 3 / 1 | 1 / 0 / 1 / 2 | +5 (119) |
| conf.md.p1 | 125 / 64 / 1 / 2 | 4.8 | +14 [+4, +24] | +2 [-2, +6] | +0 [+0, +0] | 186, +13 [+4, +22], 18 / 5 | 1 / 3 / 1 | 1 / 0 / 2 / 1 | +6 (141) |
| conf.rand.p1 | 108 / 77 / 3 / 4 | 3.2 | -3 [-10, +3] | -2 [-5, +0] | +2 [+0, +5] | 185, -1 [-7, +5], 4 / 5 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | -1 (173) |
| conf.rand.m1 | 115 / 73 / 2 / 2 | 3.2 | +4 [+0, +9] | +1 [+0, +3] | +0 [+0, +0] | 187, +3 [-1, +8], 4 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (176) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +19 (9.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +22 [+13, +32]; vs conf.rand.m1 +15 [+6, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.0 pp [-1.0, +3.1]; moved out / in 1 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 119 questions; mean words 3.1 -> 5.9
KEY family=E judge=allenai layer=18 A=incomplete B=not_met dR=+19 dR_F=NA dAns=+2 dAns_F=NA rand_margin=+15 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 107 / 58 / 5 / 22 | 6.8 | +8 [-2, +19] | +2 [-6, +10] | -3 [-10, +3] | 156, +3 [-6, +12], 11 / 8 | 7 / 9 / 3 | 2 / 5 / 7 / 2 | +3 (100) |
| x.w.m1 | 87 / 78 / 4 / 23 | 2.3 | -12 [-21, -3] | +2 [-6, +10] | -2 [-9, +5] | 156, -7 [-15, +1], 4 / 11 | 7 / 9 / 1 | 6 / 1 / 1 / 8 | -7 (153) |
| x.md.p1 | 103 / 63 / 3 / 23 | 4.9 | +4 [-5, +13] | +3 [-3, +9] | -2 [-7, +3] | 160, +0 [-8, +8], 9 / 9 | 3 / 6 / 1 | 0 / 3 / 4 / 2 | +3 (122) |
| x.rand.p1 | 97 / 67 / 4 / 24 | 3.3 | -2 [-7, +3] | +1 [+0, +3] | -1 [-3, +0] | 163, -2 [-7, +3], 2 / 4 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -1 (158) |
| x.rand.m1 | 95 / 65 / 2 / 30 | 3.2 | -4 [-9, +0] | -3 [-8, +2] | +5 [+1, +10] | 158, -4 [-8, -1], 0 / 4 | 5 / 2 / 0 | 1 / 4 / 1 / 1 | -1 (153) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 122 / 58 / 2 / 10; dR +11 [+4, +19]; answered-rate change +5.7 pp [+2.6, +9.4]; out / in 0 / 11; mean words 10.1 -> 13.0
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +10 [+0, +20]; vs x.rand.m1 +12 [+1, +23] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.0 pp [-3.1, +5.2]; moved out / in 7 / 9 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 100 questions; mean words 3.2 -> 6.8
KEY family=E judge=pro layer=18 A=not_met B=not_met dR=+8 dR_F=+11 dAns=+2 dAns_F=+11 rand_margin=+10

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 18

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 99 / 63 / 4 / 26 | +7 [-2, +16] | +5 [-3, +13] | -4 [-10, +2] | 152, +2, 8 / 6 | 5 / 10 / 2 | +0 (101) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 103 / 62 / 2 / 25 | -11 [-20, -2] | -6 [-14, +2] | +8 [+2, +15] | 160, -8, 3 / 11 | 11 / 5 / 1 | -8 (160) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 102 / 85 / 1 / 4 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +2, 2 / 0 | 0 / 0 / 0 | +2 (187) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 104 / 80 / 3 / 5 | +4 [-5, +13] | -3 [-7, +0] | +1 [+0, +3] | 184, +6, 13 / 7 | 3 / 0 / 2 | +4 (139) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 99 / 85 / 3 / 5 | -1 [-5, +3] | -3 [-7, +0] | +1 [+0, +3] | 184, +1, 2 / 1 | 3 / 0 / 2 | +1 (176) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 101 / 85 / 2 / 4 | +1 [-5, +8] | -1 [-4, +2] | +0 [-3, +3] | 185, +1, 5 / 4 | 2 / 1 / 1 | +1 (174) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 106 / 78 / 3 / 5 | +6 [-4, +16] | -3 [-7, +0] | +1 [+0, +3] | 184, +8, 16 / 8 | 3 / 0 / 2 | +2 (117) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 109 / 76 / 2 / 5 | +9 [-1, +19] | -2 [-5, +0] | +1 [+0, +3] | 185, +10, 17 / 7 | 2 / 0 / 1 | +3 (117) |
| conf_E1.wq.m1 | 117 / 68 / 2 / 5 | 108 / 77 / 1 / 6 | -9 [-18, +0] | +0 [-3, +3] | +1 [+0, +3] | 184, -9, 6 / 15 | 1 / 1 / 0 | -9 (181) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 122 / 58 / 2 / 10 | +11 [+4, +19] | +11 [+5, +18] | -5 [-10, -1] | 169, +4, 6 / 2 | 0 / 11 / 0 | +1 (100) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 106 / 78 / 3 / 5 | 5.9 | +6 [-4, +16] | -3 [-7, +0] | +1 [+0, +3] | 184, +8 [-1, +18], 16 / 8 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +2 (117) |
| conf.md.p1 | 104 / 80 / 3 / 5 | 4.8 | +4 [-5, +13] | -3 [-7, +0] | +1 [+0, +3] | 184, +6 [-3, +15], 13 / 7 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +4 (139) |
| conf.rand.p1 | 101 / 85 / 2 / 4 | 3.2 | +1 [-5, +8] | -1 [-4, +2] | +0 [-3, +3] | 185, +1 [-5, +7], 5 / 4 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +1 (174) |
| conf.rand.m1 | 99 / 85 / 3 / 5 | 3.2 | -1 [-5, +3] | -3 [-7, +0] | +1 [+0, +3] | 184, +1 [-2, +5], 2 / 1 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +1 (176) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +5 [-5, +15]; vs conf.rand.m1 +7 [-3, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 117 questions; mean words 3.1 -> 5.9
KEY family=E judge=pro layer=18 A=incomplete B=not_met dR=+6 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+5 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 153 / 38 / 0 / 1 | 14.1 | +42 [+30, +55] | +4 [+1, +8] | -1 [-3, +0] | 187, +40 [+28, +53], 44 / 4 | 0 / 4 / 0 | 0 / 0 / 2 / 2 | +2 (27) |
| conf.md.p2 | 125 / 63 / 1 / 3 | 6.0 | +14 [+3, +25] | +1 [-3, +5] | +1 [+0, +3] | 185, +14 [+4, +24], 21 / 7 | 2 / 3 / 1 | 2 / 0 / 2 / 1 | +7 (105) |
| conf.rand.p2 | 111 / 74 / 3 / 4 | 3.3 | +0 [-6, +6] | -2 [-5, +0] | +2 [+0, +5] | 185, +2 [-3, +8], 5 / 3 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | +2 (175) |
| conf.rand.m2 | 114 / 73 / 3 / 2 | 3.1 | +3 [-2, +8] | +0 [+0, +0] | +0 [+0, +0] | 187, +3 [-2, +8], 5 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +2 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +42 (21.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +42 [+30, +55]; vs conf.rand.m2 +39 [+26, +52] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +2.1 pp [+0.5, +4.2]; moved out / in 0 / 4 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 27 questions; mean words 3.1 -> 14.1
KEY family=E judge=allenai layer=13 A=incomplete B=not_met dR=+42 dR_F=NA dAns=+4 dAns_F=NA rand_margin=+39 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 117 / 68 / 3 / 4 | 14.1 | +17 [+5, +29] | -2 [-6, +2] | +0 [-3, +3] | 184, +18 [+7, +30], 28 / 10 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +2 (27) |
| conf.md.p2 | 102 / 80 / 5 / 5 | 6.0 | +2 [-9, +13] | -5 [-10, -1] | +1 [+0, +3] | 182, +6 [-4, +16], 17 / 11 | 5 / 0 / 4 | 4 / 1 / 0 / 0 | +6 (103) |
| conf.rand.p2 | 97 / 87 / 2 / 6 | 3.3 | -3 [-8, +2] | -3 [-7, +0] | +2 [+0, +5] | 184, -1 [-5, +3], 2 / 3 | 3 / 0 / 1 | 2 / 1 / 0 / 0 | +0 (174) |
| conf.rand.m2 | 107 / 81 / 1 / 3 | 3.1 | +7 [+1, +14] | +1 [+0, +3] | -1 [-3, +0] | 187, +6 [+0, +12], 8 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +5 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +20 [+8, +32]; vs conf.rand.m2 +10 [-2, +23] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-3.1, +1.0]; moved out / in 3 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 27 questions; mean words 3.1 -> 14.1
KEY family=E judge=pro layer=13 A=incomplete B=not_met dR=+17 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+10 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 148 / 41 / 0 / 3 | 13.7 | +37 [+25, +49] | +2 [+0, +5] | +1 [+0, +3] | 187, +37 [+25, +49], 41 / 4 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | +1 (27) |
| conf.md.p2 | 134 / 56 / 0 / 2 | 25.1 | +23 [+11, +35] | +3 [+0, +7] | +0 [+0, +0] | 187, +22 [+10, +34], 31 / 9 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | -2 (20) |
| conf.rand.p2 | 102 / 87 / 1 / 2 | 3.3 | -9 [-17, -1] | +2 [+0, +5] | +0 [+0, +0] | 187, -9 [-17, -1], 4 / 13 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -3 (172) |
| conf.rand.m2 | 114 / 72 / 4 / 2 | 3.4 | +3 [-3, +10] | -1 [-5, +2] | +0 [+0, +0] | 185, +4 [-1, +10], 6 / 2 | 2 / 1 / 2 | 2 / 0 / 1 / 0 | +2 (172) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +37 (19.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +46 [+34, +59]; vs conf.rand.m2 +34 [+22, +47] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.0 pp [+0.0, +2.6]; moved out / in 0 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 27 questions; mean words 3.1 -> 13.7
KEY family=E judge=allenai layer=15 A=incomplete B=incomplete dR=+37 dR_F=NA dAns=+2 dAns_F=NA rand_margin=+34 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 119 / 65 / 4 / 4 | 13.7 | +19 [+7, +32] | -3 [-7, +0] | +0 [-3, +3] | 184, +21 [+9, +33], 31 / 10 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +2 (27) |
| conf.md.p2 | 97 / 90 / 1 / 4 | 25.1 | -3 [-16, +10] | +0 [-4, +4] | +0 [-3, +3] | 185, -2 [-15, +11], 21 / 23 | 2 / 2 / 1 | 2 / 0 / 1 / 1 | -2 (20) |
| conf.rand.p2 | 95 / 90 / 3 / 4 | 3.3 | -5 [-13, +3] | -2 [-5, +0] | +0 [+0, +0] | 185, -3 [-11, +5], 6 / 9 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | +1 (171) |
| conf.rand.m2 | 103 / 82 / 3 / 4 | 3.4 | +3 [-3, +10] | -2 [-6, +2] | +0 [-3, +3] | 184, +4 [-1, +10], 6 / 2 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +3 (173) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +19 (9.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +24 [+12, +36]; vs conf.rand.m2 +16 [+3, +29] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 27 questions; mean words 3.1 -> 13.7
KEY family=E judge=pro layer=15 A=incomplete B=not_met dR=+19 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+16 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

=== 2026-11-13 00:57:31 [qwen3.5-9b E tables gpunone] tables done ===

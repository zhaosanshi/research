TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 111 / 53 / 0 / 28 | 3.0 | +8 [+2, +15] | -2 [-6, +2] | +3 [-1, +8] | 163, +7 [+1, +14], 9 / 2 | 3 / 1 / 0 | 0 / 3 / 1 / 0 | +7 (160) |
| x.w.m1 | 98 / 69 / 3 / 22 | 3.3 | -5 [-12, +1] | +1 [-4, +6] | -3 [-8, +1] | 163, -4 [-10, +1], 2 / 6 | 3 / 4 / 2 | 2 / 1 / 1 / 3 | -4 (152) |
| x.md.p1 | 112 / 50 / 1 / 29 | 3.3 | +9 [+3, +16] | -4 [-9, +0] | +4 [+0, +9] | 161, +8 [+2, +14], 9 / 1 | 5 / 1 / 0 | 0 / 5 / 1 / 0 | +8 (153) |
| x.rand.p1 | 103 / 63 / 2 / 24 | 3.3 | +0 [-5, +5] | +0 [-3, +3] | -1 [-3, +0] | 165, +1 [-3, +6], 3 / 2 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +1 (162) |
| x.rand.m1 | 102 / 61 / 1 / 28 | 3.3 | -1 [-5, +2] | -3 [-7, +0] | +3 [+0, +7] | 163, +0 [-3, +3], 1 / 1 | 3 / 0 / 0 | 1 / 2 / 0 / 0 | +1 (158) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 132 / 43 / 1 / 16; dR -2 [-8, +4]; answered-rate change +0.0 pp [-1.6, +1.6]; out / in 1 / 1; mean words 10.1 -> 9.9
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +8 [+1, +15]; vs x.rand.m1 +9 [+2, +17] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.0 pp [-3.1, +1.0]; moved out / in 3 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +7 on 160 questions; mean words 3.2 -> 3.0
KEY family=E judge=allenai layer=18 A=not_met B=not_met dR=+8 dR_F=-2 dAns=-2 dAns_F=+0 rand_margin=+8

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 18

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 107 / 50 / 0 / 35 | +6 [-1, +13] | -3 [-10, +4] | +5 [-2, +12] | 152, +4, 5 / 1 | 8 / 5 / 0 | +4 (145) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 111 / 46 / 2 / 33 | -6 [-13, +1] | +5 [+0, +11] | -7 [-13, -2] | 150, -6, 2 / 8 | 2 / 7 / 1 | -6 (140) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 109 / 78 / 3 / 2 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -2, 0 / 2 | 0 / 0 / 0 | -1 (186) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 117 / 70 / 2 / 3 | +6 [+0, +13] | +0 [+0, +0] | +1 [+0, +3] | 187, +6, 9 / 3 | 0 / 0 / 0 | +5 (180) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 114 / 74 / 2 / 2 | +3 [+0, +7] | +1 [+0, +3] | +0 [+0, +0] | 187, +2, 2 / 0 | 0 / 1 / 0 | +2 (181) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 109 / 77 / 3 / 3 | -2 [-7, +3] | -1 [-3, +0] | +1 [+0, +3] | 186, -1, 2 / 3 | 1 / 0 / 1 | +0 (181) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 121 / 65 / 2 / 4 | +10 [+2, +18] | -1 [-3, +0] | +2 [+0, +5] | 186, +11, 13 / 2 | 1 / 0 / 0 | +11 (179) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 126 / 60 / 3 / 3 | +15 [+6, +24] | -1 [-3, +0] | +1 [+0, +3] | 186, +16, 18 / 2 | 1 / 0 / 0 | +14 (178) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 123 / 66 / 1 / 2 | -12 [-22, -3] | +0 [-3, +3] | +0 [+0, +0] | 188, -11, 6 / 17 | 1 / 1 / 1 | -8 (170) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 133 / 42 / 1 / 16 | -1 [-6, +4] | +0 [-3, +3] | +1 [+0, +3] | 174, -1, 2 / 3 | 1 / 1 / 0 | -2 (164) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 132 / 43 / 1 / 16 | -2 [-8, +4] | +0 [-3, +3] | +1 [+0, +3] | 174, -2, 3 / 5 | 1 / 1 / 0 | -2 (158) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 135 / 33 / 0 / 24 | +1 [-8, +10] | -7 [-14, -1] | +9 [+4, +15] | 166, +2, 8 / 6 | 9 / 2 / 0 | +0 (138) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 121 / 65 / 2 / 4 | 2.8 | +10 [+2, +18] | -1 [-3, +0] | +2 [+0, +5] | 186, +11 [+4, +19], 13 / 2 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +11 (179) |
| conf.md.p1 | 117 / 70 / 2 / 3 | 2.9 | +6 [+0, +13] | +0 [+0, +0] | +1 [+0, +3] | 187, +6 [+0, +13], 9 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +5 (180) |
| conf.rand.p1 | 109 / 77 / 3 / 3 | 3.2 | -2 [-7, +3] | -1 [-3, +0] | +1 [+0, +3] | 186, -1 [-5, +3], 2 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (181) |
| conf.rand.m1 | 114 / 74 / 2 / 2 | 3.2 | +3 [+0, +7] | +1 [+0, +3] | +0 [+0, +0] | 187, +2 [+0, +5], 2 / 0 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +12 [+4, +20]; vs conf.rand.m1 +7 [-1, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +11 on 179 questions; mean words 3.1 -> 2.8
KEY family=E judge=allenai layer=18 A=incomplete B=not_met dR=+10 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+7 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 107 / 54 / 4 / 27 | 3.0 | +8 [+0, +16] | -2 [-8, +3] | +2 [-2, +6] | 158, +8 [+1, +15], 11 / 3 | 5 / 3 / 2 | 2 / 3 / 2 / 1 | +7 (156) |
| x.w.m1 | 91 / 75 / 3 / 23 | 3.3 | -8 [-15, -2] | +3 [-2, +8] | -2 [-7, +3] | 161, -9 [-16, -3], 1 / 10 | 2 / 5 / 0 | 0 / 2 / 1 / 4 | -5 (149) |
| x.md.p1 | 104 / 56 / 2 / 30 | 3.3 | +5 [-1, +12] | -3 [-8, +2] | +5 [+0, +10] | 158, +4 [-2, +10], 7 / 3 | 5 / 2 / 0 | 0 / 5 / 1 / 1 | +3 (150) |
| x.rand.p1 | 96 / 68 / 4 / 24 | 3.3 | -3 [-8, +1] | +1 [+0, +3] | -1 [-3, +0] | 163, -3 [-8, +1], 1 / 4 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -1 (159) |
| x.rand.m1 | 95 / 65 / 4 / 28 | 3.3 | -4 [-8, -1] | -3 [-7, +0] | +3 [+0, +7] | 160, -3 [-7, +0], 0 / 3 | 3 / 0 / 0 | 1 / 2 / 0 / 0 | +0 (155) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 111 / 57 / 8 / 16; dR +0 [-7, +7]; answered-rate change -0.5 pp [-2.6, +1.0]; out / in 2 / 1; mean words 10.1 -> 9.9
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +11 [+3, +20]; vs x.rand.m1 +12 [+3, +21] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.0 pp [-4.2, +1.6]; moved out / in 5 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +7 on 156 questions; mean words 3.2 -> 3.0
KEY family=E judge=pro layer=18 A=not_met B=not_met dR=+8 dR_F=+0 dAns=-2 dAns_F=-1 rand_margin=+11

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 18

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 100 / 54 / 3 / 35 | +8 [+0, +16] | -3 [-11, +4] | +5 [-2, +12] | 148, +5, 7 / 2 | 9 / 6 / 1 | +4 (142) |
| caut_E2.wq.m1 | 109 / 39 / 3 / 41 | 104 / 52 / 2 / 34 | -5 [-12, +2] | +8 [+2, +14] | -7 [-13, -2] | 147, -7, 2 / 9 | 1 / 9 / 0 | -5 (137) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 99 / 88 / 1 / 4 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 108 / 79 / 1 / 4 | +8 [+2, +14] | +0 [+0, +0] | +0 [+0, +0] | 187, +8, 9 / 1 | 0 / 0 / 0 | +6 (180) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 99 / 87 / 1 / 5 | -1 [-5, +2] | -1 [-3, +0] | +1 [+0, +3] | 186, +0, 1 / 1 | 1 / 0 / 0 | +0 (181) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 101 / 86 / 1 / 4 | +1 [-3, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +1, 3 / 2 | 0 / 0 / 0 | +2 (181) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 112 / 74 / 1 / 5 | +12 [+5, +19] | -1 [-3, +0] | +1 [+0, +3] | 186, +12, 13 / 1 | 1 / 0 / 0 | +12 (179) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 115 / 71 / 1 / 5 | +15 [+8, +23] | -1 [-3, +0] | +1 [+0, +3] | 186, +15, 16 / 1 | 1 / 0 / 0 | +13 (178) |
| conf_E2.wq.m1 | 122 / 64 / 2 / 4 | 105 / 80 / 3 / 4 | -17 [-26, -8] | -1 [-3, +0] | +0 [+0, +0] | 185, -17, 3 / 20 | 1 / 0 / 1 | -11 (168) |
| x_F.w.p0.5 | 111 / 58 / 8 / 15 | 111 / 57 / 8 / 16 | +0 [-5, +5] | -1 [-5, +2] | +1 [+0, +3] | 167, +0, 2 / 2 | 2 / 1 / 1 | -2 (157) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 111 / 57 / 8 / 16 | +0 [-7, +7] | -1 [-5, +2] | +1 [+0, +3] | 167, +0, 5 / 5 | 2 / 1 / 1 | -1 (151) |
| x_F.w.p2 | 111 / 58 / 8 / 15 | 107 / 51 / 10 / 24 | -4 [-13, +5] | -11 [-18, -5] | +9 [+4, +15] | 157, -1, 8 / 9 | 12 / 1 / 3 | +2 (131) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 112 / 74 / 1 / 5 | 2.8 | +12 [+5, +19] | -1 [-3, +0] | +1 [+0, +3] | 186, +12 [+5, +19], 13 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +12 (179) |
| conf.md.p1 | 108 / 79 / 1 / 4 | 2.9 | +8 [+2, +14] | +0 [+0, +0] | +0 [+0, +0] | 187, +8 [+2, +14], 9 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +6 (180) |
| conf.rand.p1 | 101 / 86 / 1 / 4 | 3.2 | +1 [-3, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +1 [-3, +5], 3 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +2 (181) |
| conf.rand.m1 | 99 / 87 / 1 / 5 | 3.2 | -1 [-5, +2] | -1 [-3, +0] | +1 [+0, +3] | 186, +0 [-3, +3], 1 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +12 (6.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +11 [+4, +19]; vs conf.rand.m1 +13 [+7, +20] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +12 on 179 questions; mean words 3.1 -> 2.8
KEY family=E judge=pro layer=18 A=incomplete B=incomplete dR=+12 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+11 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 10:34:48 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

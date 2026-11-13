TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 85 / 76 / 3 / 28 | 4.4 | +7 [-3, +17] | -5 [-11, +1] | +5 [+0, +10] | 159, +8 [-2, +17], 16 / 8 | 7 / 2 / 1 | 3 / 4 / 2 / 0 | +7 (141) |
| x.w.m1 | 80 / 90 / 3 / 19 | 4.0 | +2 [-5, +9] | +4 [+0, +9] | -4 [-8, -1] | 165, +0 [-6, +5], 4 / 4 | 1 / 5 / 1 | 1 / 0 / 3 / 2 | +0 (159) |
| x.md.p1 | 79 / 82 / 2 / 29 | 4.2 | +1 [-9, +11] | -5 [-10, +0] | +6 [+2, +11] | 160, +2 [-7, +11], 12 / 10 | 6 / 1 / 0 | 2 / 4 / 1 / 0 | +0 (151) |
| x.rand.p1 | 81 / 85 / 3 / 23 | 4.2 | +3 [-3, +9] | +0 [+0, +0] | +0 [+0, +0] | 166, +3 [-3, +9], 6 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +2 (159) |
| x.rand.m1 | 77 / 88 / 1 / 26 | 4.1 | -1 [-7, +5] | -1 [-6, +3] | +3 [+0, +7] | 163, -2 [-7, +3], 2 / 4 | 3 / 2 / 0 | 1 / 2 / 2 / 0 | -2 (160) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 98 / 58 / 1 / 35; dR -4 [-14, +6]; answered-rate change -2.6 pp [-5.2, +0.0]; out / in 6 / 1; mean words 9.3 -> 9.4
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +4 [-7, +15]; vs x.rand.m1 +8 [-1, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.6 pp [-5.7, +0.5]; moved out / in 7 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +7 on 141 questions; mean words 4.0 -> 4.4
KEY family=E judge=allenai layer=17 A=not_met B=not_met dR=+7 dR_F=-4 dAns=-5 dAns_F=-5 rand_margin=+4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 86 / 72 / 2 / 32 | +4 [-5, +14] | -6 [-12, -1] | +5 [+0, +10] | 157, +4, 13 / 9 | 7 / 1 / 1 | -1 (139) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 93 / 67 / 1 / 31 | -9 [-18, +0] | +5 [-1, +12] | -6 [-13, +0] | 152, -8, 4 / 12 | 3 / 8 / 1 | -7 (143) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 88 / 101 / 1 / 2 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 189, +0, 1 / 1 | 0 / 0 / 0 | +0 (188) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 95 / 95 / 1 / 1 | +7 [-1, +16] | +1 [-2, +5] | -1 [-3, +0] | 188, +7, 12 / 5 | 1 / 2 / 1 | +1 (176) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 86 / 102 / 2 / 2 | -2 [-7, +2] | -1 [-3, +0] | +0 [+0, +0] | 188, -1, 2 / 3 | 1 / 0 / 1 | +0 (184) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 85 / 105 / 0 / 2 | -3 [-10, +3] | +1 [+0, +3] | +0 [+0, +0] | 189, -3, 4 / 7 | 0 / 1 / 0 | -2 (186) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 95 / 93 / 3 / 1 | +7 [-1, +16] | -1 [-5, +3] | -1 [-3, +0] | 186, +9, 12 / 3 | 3 / 2 / 3 | +5 (170) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 102 / 86 / 3 / 1 | +14 [+5, +24] | -1 [-5, +3] | -1 [-3, +0] | 186, +16, 19 / 3 | 3 / 2 / 3 | +10 (169) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 106 / 84 / 0 / 2 | -11 [-22, -1] | -1 [-5, +2] | +2 [+0, +5] | 189, -10, 8 / 18 | 2 / 1 / 0 | -9 (178) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 98 / 58 / 1 / 35 | -4 [-14, +6] | -5 [-10, +0] | +5 [+0, +10] | 155, -3, 9 / 12 | 6 / 1 / 0 | -2 (125) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 95 / 93 / 3 / 1 | 3.7 | +7 [-1, +16] | -1 [-5, +3] | -1 [-3, +0] | 186, +9 [+2, +17], 12 / 3 | 3 / 2 / 3 | 3 / 0 / 1 / 1 | +5 (170) |
| conf.md.p1 | 95 / 95 / 1 / 1 | 3.5 | +7 [-1, +16] | +1 [-2, +5] | -1 [-3, +0] | 188, +7 [-1, +15], 12 / 5 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +1 (176) |
| conf.rand.p1 | 85 / 105 / 0 / 2 | 3.3 | -3 [-10, +3] | +1 [+0, +3] | +0 [+0, +0] | 189, -3 [-10, +3], 4 / 7 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -2 (186) |
| conf.rand.m1 | 86 / 102 / 2 / 2 | 3.4 | -2 [-7, +2] | -1 [-3, +0] | +0 [+0, +0] | 188, -1 [-5, +3], 2 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +10 [+0, +20]; vs conf.rand.m1 +9 [+0, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.6]; moved out / in 3 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +5 on 170 questions; mean words 3.4 -> 3.7
KEY family=E judge=allenai layer=17 A=incomplete B=not_met dR=+7 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+9 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 67 / 90 / 5 / 30 | 4.4 | +3 [-7, +13] | -5 [-12, +1] | +6 [+1, +12] | 154, +2 [-7, +11], 12 / 10 | 8 / 3 / 2 | 2 / 6 / 3 / 0 | +2 (138) |
| x.w.m1 | 63 / 101 / 7 / 21 | 4.0 | -1 [-7, +5] | +2 [-2, +6] | -3 [-7, +0] | 161, -2 [-7, +3], 3 / 5 | 1 / 3 / 1 | 1 / 0 / 2 / 1 | -2 (155) |
| x.md.p1 | 69 / 88 / 5 / 30 | 4.2 | +5 [-3, +14] | -5 [-12, +1] | +6 [+2, +11] | 154, +5 [-2, +13], 10 / 5 | 8 / 3 / 2 | 2 / 6 / 2 / 1 | +4 (145) |
| x.rand.p1 | 65 / 97 / 6 / 24 | 4.2 | +1 [-4, +6] | +0 [-3, +3] | +0 [+0, +0] | 161, +1 [-3, +5], 3 / 2 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +3 (154) |
| x.rand.m1 | 65 / 94 / 6 / 27 | 4.1 | +1 [-3, +5] | -3 [-7, +0] | +3 [+0, +7] | 159, +1 [-3, +5], 3 / 2 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | +1 (156) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 82 / 72 / 3 / 35; dR +2 [-8, +12]; answered-rate change -3.1 pp [-6.8, +0.0]; out / in 8 / 2; mean words 9.3 -> 9.4
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +2 [-7, +11]; vs x.rand.m1 +2 [-8, +12] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.6 pp [-6.2, +0.5]; moved out / in 8 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 138 questions; mean words 4.0 -> 4.4
KEY family=E judge=pro layer=17 A=not_met B=not_met dR=+3 dR_F=+2 dAns=-5 dAns_F=-6 rand_margin=+2

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 66 / 86 / 7 / 33 | +2 [-8, +12] | -6 [-13, +1] | +5 [+0, +10] | 149, +2, 11 / 9 | 9 / 3 / 3 | -1 (134) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 77 / 79 / 3 / 33 | -4 [-11, +3] | +6 [+0, +12] | -7 [-13, -2] | 148, -4, 4 / 8 | 2 / 8 / 1 | -4 (139) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 71 / 111 / 9 / 1 | +1 [-2, +5] | +0 [-3, +3] | +0 [+0, +0] | 181, +0, 1 / 1 | 1 / 1 / 1 | +0 (180) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 76 / 102 / 11 / 3 | +6 [-1, +13] | -4 [-8, -1] | +2 [+0, +5] | 178, +7, 10 / 3 | 4 / 0 / 3 | +4 (170) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 70 / 111 / 10 / 1 | +0 [-4, +4] | -1 [-3, +0] | +0 [+0, +0] | 181, +1, 2 / 1 | 1 / 0 / 1 | +1 (177) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 69 / 114 / 8 / 1 | -1 [-8, +5] | +1 [+0, +3] | +0 [+0, +0] | 182, -2, 4 / 6 | 0 / 1 / 0 | -2 (180) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 75 / 102 / 12 / 3 | +5 [-3, +13] | -5 [-10, -1] | +2 [+0, +5] | 177, +7, 11 / 4 | 5 / 0 / 4 | +4 (164) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 81 / 97 / 10 / 4 | +11 [+2, +20] | -4 [-9, +1] | +3 [+0, +7] | 177, +12, 16 / 4 | 5 / 1 / 4 | +7 (162) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 79 / 104 / 7 / 2 | -19 [-30, -9] | -1 [-5, +2] | -2 [-5, +0] | 182, -17, 6 / 23 | 2 / 1 / 2 | -16 (172) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 82 / 72 / 3 / 35 | +2 [-8, +12] | -6 [-13, +0] | +5 [-1, +11] | 152, +3, 12 / 9 | 8 / 2 / 1 | +4 (124) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 75 / 102 / 12 / 3 | 3.7 | +5 [-3, +13] | -5 [-10, -1] | +2 [+0, +5] | 177, +7 [+0, +15], 11 / 4 | 5 / 0 / 4 | 2 / 3 / 0 / 0 | +4 (164) |
| conf.md.p1 | 76 / 102 / 11 / 3 | 3.5 | +6 [-1, +13] | -4 [-8, -1] | +2 [+0, +5] | 178, +7 [+0, +14], 10 / 3 | 4 / 0 / 3 | 1 / 3 / 0 / 0 | +4 (170) |
| conf.rand.p1 | 69 / 114 / 8 / 1 | 3.3 | -1 [-8, +5] | +1 [+0, +3] | +0 [+0, +0] | 182, -2 [-8, +4], 4 / 6 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | -2 (180) |
| conf.rand.m1 | 70 / 111 / 10 / 1 | 3.4 | +0 [-4, +4] | -1 [-3, +0] | +0 [+0, +0] | 181, +1 [-2, +4], 2 / 1 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +1 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +6 [-3, +15]; vs conf.rand.m1 +5 [-2, +13] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.6 pp [-5.2, -0.5]; moved out / in 5 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +4 on 164 questions; mean words 3.4 -> 3.7
KEY family=E judge=pro layer=17 A=not_met B=not_met dR=+5 dR_F=NA dAns=-5 dAns_F=NA rand_margin=+5 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 13:46:42 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

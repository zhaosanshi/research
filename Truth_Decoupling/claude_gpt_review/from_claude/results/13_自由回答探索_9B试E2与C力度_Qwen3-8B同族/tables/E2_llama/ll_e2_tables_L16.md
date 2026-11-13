TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 78 / 74 / 3 / 37 | 4.4 | +0 [-11, +11] | -14 [-22, -6] | +14 [+7, +22] | 150, +6 [-4, +15], 15 / 9 | 16 / 2 / 1 | 7 / 9 / 1 / 1 | +3 (130) |
| x.w.m1 | 78 / 93 / 4 / 17 | 4.1 | +0 [-9, +9] | +5 [-1, +11] | -6 [-11, -2] | 164, -2 [-9, +5], 6 / 8 | 2 / 7 / 2 | 2 / 0 / 4 / 3 | -2 (159) |
| x.md.p1 | 80 / 72 / 2 / 38 | 4.0 | +2 [-8, +12] | -14 [-22, -7] | +15 [+8, +23] | 151, +6 [-3, +15], 13 / 7 | 15 / 1 / 0 | 5 / 10 / 1 / 0 | +3 (142) |
| x.rand.p1 | 80 / 89 / 2 / 21 | 4.2 | +2 [-3, +8] | +3 [+0, +7] | -2 [-5, +0] | 166, +1 [-4, +6], 4 / 3 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +1 (156) |
| x.rand.m1 | 79 / 86 / 2 / 25 | 4.1 | +1 [-5, +7] | -1 [-5, +2] | +2 [+0, +5] | 164, +0 [-6, +6], 4 / 4 | 2 / 1 / 0 | 0 / 2 / 1 / 0 | +0 (157) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 93 / 55 / 2 / 42; dR -9 [-19, +0]; answered-rate change -6.8 pp [-10.9, -3.1]; out / in 14 / 1; mean words 9.3 -> 9.3
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -2 [-12, +8]; vs x.rand.m1 -1 [-12, +9] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -7.3 pp [-11.5, -3.1]; moved out / in 16 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +3 on 130 questions; mean words 4.0 -> 4.4
KEY family=E judge=allenai layer=16 A=not_met B=not_met dR=+0 dR_F=-9 dAns=-14 dAns_F=-13 rand_margin=-2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 91 / 62 / 1 / 38 | +9 [+0, +18] | -11 [-19, -4] | +11 [+4, +18] | 151, +8, 14 / 6 | 13 / 2 / 1 | +6 (133) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 97 / 71 / 1 / 23 | -5 [-15, +5] | +13 [+5, +21] | -14 [-22, -7] | 153, -10, 5 / 15 | 2 / 15 / 1 | -8 (143) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 89 / 100 / 1 / 2 | +1 [-2, +4] | +0 [+0, +0] | +0 [+0, +0] | 189, +1, 2 / 1 | 0 / 0 / 0 | +1 (188) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 94 / 95 / 2 / 1 | +6 [-3, +15] | +0 [-4, +4] | -1 [-3, +0] | 187, +7, 13 / 6 | 2 / 2 / 2 | +2 (174) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 88 / 101 / 1 / 2 | +0 [-5, +5] | +0 [+0, +0] | +0 [+0, +0] | 189, +0, 3 / 3 | 0 / 0 / 0 | -1 (183) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 89 / 100 / 1 / 2 | +1 [-5, +7] | +0 [-3, +3] | +0 [+0, +0] | 188, +2, 5 / 3 | 1 / 1 / 1 | +4 (181) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 100 / 88 / 3 / 1 | +12 [+3, +21] | -1 [-5, +3] | -1 [-3, +0] | 186, +14, 17 / 3 | 3 / 2 / 3 | +8 (165) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 107 / 79 / 4 / 2 | +19 [+8, +30] | -3 [-8, +1] | +0 [+0, +0] | 185, +22, 25 / 3 | 4 / 1 / 4 | +14 (165) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 98 / 91 / 1 / 2 | -19 [-30, -8] | -2 [-6, +2] | +2 [+0, +5] | 188, -17, 6 / 23 | 3 / 1 / 1 | -14 (178) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 93 / 55 / 2 / 42 | -9 [-19, +0] | -13 [-21, -6] | +12 [+6, +19] | 147, -2, 7 / 9 | 14 / 1 / 2 | -1 (117) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 88 / 3 / 1 | 3.9 | +12 [+3, +21] | -1 [-5, +3] | -1 [-3, +0] | 186, +14 [+6, +23], 17 / 3 | 3 / 2 / 3 | 3 / 0 / 1 / 1 | +8 (165) |
| conf.md.p1 | 94 / 95 / 2 / 1 | 3.5 | +6 [-3, +15] | +0 [-4, +4] | -1 [-3, +0] | 187, +7 [-1, +16], 13 / 6 | 2 / 2 / 2 | 2 / 0 / 1 / 1 | +2 (174) |
| conf.rand.p1 | 89 / 100 / 1 / 2 | 3.5 | +1 [-5, +7] | +0 [-3, +3] | +0 [+0, +0] | 188, +2 [-3, +8], 5 / 3 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +4 (181) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 3.5 | +0 [-5, +5] | +0 [+0, +0] | +0 [+0, +0] | 189, +0 [-5, +5], 3 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +12 (6.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +11 [+2, +20]; vs conf.rand.m1 +12 [+3, +22] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.6]; moved out / in 3 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +8 on 165 questions; mean words 3.4 -> 3.9
KEY family=E judge=allenai layer=16 A=incomplete B=incomplete dR=+12 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+11 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 61 / 88 / 4 / 39 | 4.4 | -3 [-14, +8] | -13 [-22, -5] | +15 [+8, +23] | 146, -2 [-11, +7], 10 / 12 | 16 / 3 / 1 | 4 / 12 / 3 / 0 | -4 (128) |
| x.w.m1 | 66 / 100 / 7 / 19 | 4.1 | +2 [-6, +10] | +4 [+0, +9] | -5 [-10, -1] | 161, +1 [-6, +8], 7 / 6 | 1 / 5 / 1 | 1 / 0 / 2 / 3 | +1 (156) |
| x.md.p1 | 68 / 82 / 3 / 39 | 4.0 | +4 [-4, +12] | -12 [-20, -4] | +15 [+8, +23] | 147, +4 [-3, +12], 9 / 5 | 15 / 3 / 1 | 2 / 13 / 2 / 1 | +4 (137) |
| x.rand.p1 | 65 / 99 / 6 / 22 | 4.2 | +1 [-4, +6] | +2 [+0, +5] | -2 [-5, +0] | 162, +0 [-5, +5], 3 / 3 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +2 (152) |
| x.rand.m1 | 60 / 99 / 7 / 26 | 4.1 | -4 [-10, +1] | -3 [-7, +0] | +2 [+0, +5] | 159, -3 [-8, +2], 2 / 5 | 3 / 0 / 1 | 1 / 2 / 0 / 0 | -2 (153) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 82 / 66 / 2 / 42; dR +2 [-7, +11]; answered-rate change -6.2 pp [-9.9, -2.6]; out / in 13 / 1; mean words 9.3 -> 9.3
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -4 [-14, +6]; vs x.rand.m1 +1 [-9, +11] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -6.8 pp [-11.5, -2.6]; moved out / in 16 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same -4 on 128 questions; mean words 4.0 -> 4.4
KEY family=E judge=pro layer=16 A=not_met B=not_met dR=-3 dR_F=+2 dAns=-13 dAns_F=-12 rand_margin=-4

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 76 / 72 / 6 / 38 | +12 [+2, +22] | -10 [-19, -2] | +10 [+3, +17] | 143, +10, 15 / 5 | 15 / 5 / 3 | +8 (128) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 77 / 85 / 5 / 25 | -4 [-13, +5] | +12 [+5, +20] | -15 [-23, -8] | 148, -7, 6 / 13 | 2 / 14 / 2 | -6 (139) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 70 / 112 / 8 / 2 | +0 [-4, +4] | +0 [-3, +3] | +1 [+0, +3] | 181, -1, 1 / 2 | 1 / 1 / 0 | -1 (180) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 77 / 100 / 12 / 3 | +7 [-1, +15] | -5 [-10, -1] | +2 [+0, +5] | 177, +9, 12 / 3 | 5 / 0 / 4 | +5 (168) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 71 / 111 / 9 / 1 | +1 [-4, +6] | +0 [-3, +3] | +0 [+0, +0] | 181, +0, 3 / 3 | 1 / 1 / 1 | -1 (175) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 73 / 107 / 11 / 1 | +3 [-4, +10] | -2 [-5, +0] | +0 [+0, +0] | 180, +4, 8 / 4 | 2 / 0 / 2 | +4 (173) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 80 / 98 / 12 / 2 | +10 [+1, +20] | -4 [-9, +1] | +1 [+0, +3] | 177, +11, 16 / 5 | 5 / 1 / 4 | +6 (159) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 88 / 91 / 10 / 3 | +18 [+8, +28] | -3 [-8, +2] | +2 [+0, +5] | 177, +18, 22 / 4 | 5 / 2 / 4 | +11 (160) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 74 / 107 / 9 / 2 | -24 [-36, -12] | -3 [-8, +1] | -2 [-5, +0] | 180, -21, 6 / 27 | 4 / 1 / 4 | -18 (171) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 82 / 66 / 2 / 42 | +2 [-7, +11] | -12 [-19, -5] | +12 [+5, +19] | 147, +5, 11 / 6 | 13 / 1 / 0 | +7 (118) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 80 / 98 / 12 / 2 | 3.9 | +10 [+1, +20] | -4 [-9, +1] | +1 [+0, +3] | 177, +11 [+2, +20], 16 / 5 | 5 / 1 / 4 | 2 / 3 / 1 / 0 | +6 (159) |
| conf.md.p1 | 77 / 100 / 12 / 3 | 3.5 | +7 [-1, +15] | -5 [-10, -1] | +2 [+0, +5] | 177, +9 [+2, +17], 12 / 3 | 5 / 0 / 4 | 2 / 3 / 0 / 0 | +5 (168) |
| conf.rand.p1 | 73 / 107 / 11 / 1 | 3.5 | +3 [-4, +10] | -2 [-5, +0] | +0 [+0, +0] | 180, +4 [-3, +11], 8 / 4 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | +4 (173) |
| conf.rand.m1 | 71 / 111 / 9 / 1 | 3.5 | +1 [-4, +6] | +0 [-3, +3] | +0 [+0, +0] | 181, +0 [-5, +5], 3 / 3 | 1 / 1 / 1 | 0 / 1 / 1 / 0 | -1 (175) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +7 [-2, +16]; vs conf.rand.m1 +9 [-1, +19] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.1 pp [-4.7, +0.5]; moved out / in 5 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 159 questions; mean words 3.4 -> 3.9
KEY family=E judge=pro layer=16 A=incomplete B=not_met dR=+10 dR_F=NA dAns=-4 dAns_F=NA rand_margin=+7 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 13:39:48 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

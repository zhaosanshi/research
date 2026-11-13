TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 86 / 53 / 0 / 53 | 4.6 | +8 [-4, +20] | -27 [-38, -16] | +30 [+20, +40] | 135, +17 [+7, +27], 22 / 5 | 31 / 4 / 0 | 11 / 20 / 2 / 2 | +8 (109) |
| x.w.m1 | 72 / 104 / 3 / 13 | 3.8 | -6 [-17, +4] | +10 [+3, +18] | -10 [-16, -4] | 164, -9 [-18, +0], 6 / 15 | 2 / 12 / 2 | 2 / 0 / 5 / 7 | -9 (154) |
| x.md.p1 | 77 / 90 / 1 / 24 | 4.1 | -1 [-9, +7] | +1 [-2, +4] | +1 [+0, +3] | 165, -2 [-10, +6], 7 / 9 | 1 / 2 / 0 | 0 / 1 / 1 / 1 | -3 (157) |
| x.rand.p1 | 80 / 88 / 2 / 22 | 4.0 | +2 [-2, +6] | +2 [+0, +5] | -1 [-3, +0] | 166, +1 [-2, +5], 2 / 1 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +1 (160) |
| x.rand.m1 | 81 / 84 / 1 / 26 | 4.1 | +3 [-3, +9] | -1 [-5, +3] | +3 [+0, +7] | 163, +4 [+0, +9], 5 / 1 | 3 / 2 / 0 | 2 / 1 / 1 / 1 | +4 (157) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 93 / 36 / 0 / 63; dR -9 [-22, +4]; answered-rate change -16.7 pp [-22.4, -10.9]; out / in 35 / 3; mean words 9.3 -> 8.3
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +6 [-5, +18]; vs x.rand.m1 +5 [-7, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -14.1 pp [-19.8, -8.3]; moved out / in 31 / 4 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 109 questions; mean words 4.0 -> 4.6
KEY family=E judge=allenai layer=11 A=not_met B=not_met dR=+8 dR_F=-9 dAns=-27 dAns_F=-32 rand_margin=+5

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 85 / 55 / 0 / 52 | +3 [-7, +14] | -24 [-34, -15] | +25 [+16, +34] | 139, +10, 15 / 5 | 25 / 1 / 0 | +2 (111) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 81 / 85 / 1 / 25 | -21 [-32, -11] | +11 [+5, +18] | -12 [-19, -6] | 154, -21, 4 / 25 | 1 / 12 / 1 | -19 (144) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 87 / 102 / 1 / 2 | -1 [-4, +2] | +0 [+0, +0] | +0 [+0, +0] | 189, -1, 1 / 2 | 0 / 0 / 0 | -1 (189) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 88 / 102 / 0 / 2 | +0 [-7, +7] | +1 [+0, +3] | +0 [+0, +0] | 189, +0, 6 / 6 | 0 / 1 / 0 | +0 (180) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 86 / 103 / 1 / 2 | -2 [-8, +4] | +0 [-3, +3] | +0 [+0, +0] | 188, -1, 4 / 5 | 1 / 1 / 1 | +0 (182) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 81 / 108 / 1 / 2 | -7 [-12, -2] | +0 [+0, +0] | +0 [+0, +0] | 189, -7, 0 / 7 | 0 / 0 / 0 | -6 (183) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 117 / 72 / 2 / 1 | +29 [+17, +41] | +0 [-3, +3] | -1 [-3, +0] | 188, +29, 35 / 6 | 1 / 1 / 1 | +19 (158) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 116 / 75 / 0 / 1 | +28 [+16, +39] | +2 [+0, +5] | -1 [-3, +0] | 189, +26, 32 / 6 | 0 / 2 / 0 | +18 (159) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 87 / 104 / 0 / 1 | -30 [-42, -18] | +0 [-3, +3] | +1 [+0, +3] | 190, -29, 7 / 36 | 1 / 1 / 0 | -25 (178) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 93 / 36 / 0 / 63 | -9 [-22, +4] | -32 [-43, -21] | +33 [+22, +45] | 126, +7, 15 / 8 | 35 / 3 / 0 | +1 (91) |

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

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 71 / 65 / 3 / 53 | 4.6 | +7 [-4, +18] | -26 [-38, -15] | +29 [+19, +39] | 130, +10 [+2, +19], 15 / 5 | 32 / 6 / 2 | 8 / 24 / 5 / 1 | +5 (108) |
| x.w.m1 | 62 / 110 / 7 / 13 | 3.8 | -2 [-11, +7] | +10 [+4, +17] | -11 [-18, -5] | 161, -4 [-12, +4], 7 / 11 | 1 / 11 / 1 | 1 / 0 / 3 / 8 | -3 (153) |
| x.md.p1 | 67 / 96 / 4 / 25 | 4.1 | +3 [-3, +9] | +1 [-2, +5] | +1 [+0, +3] | 161, +1 [-4, +6], 4 / 3 | 1 / 2 / 0 | 0 / 1 / 2 / 0 | +0 (154) |
| x.rand.p1 | 66 / 99 / 4 / 23 | 4.0 | +2 [-2, +6] | +3 [+0, +7] | -1 [-3, +0] | 162, +0 [-3, +3], 1 / 1 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | +1 (157) |
| x.rand.m1 | 65 / 95 / 5 / 27 | 4.1 | +1 [-5, +7] | -2 [-6, +2] | +3 [+0, +7] | 159, +0 [-6, +6], 4 / 4 | 3 / 1 / 0 | 0 / 3 / 1 / 0 | +2 (153) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 74 / 49 / 4 / 65; dR -6 [-19, +7]; answered-rate change -19.3 pp [-25.5, -13.5]; out / in 39 / 2; mean words 9.3 -> 8.3
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-6, +16]; vs x.rand.m1 +6 [-5, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -13.5 pp [-19.8, -7.8]; moved out / in 32 / 6 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 108 questions; mean words 4.0 -> 4.6
KEY family=E judge=pro layer=11 A=not_met B=not_met dR=+7 dR_F=-6 dAns=-26 dAns_F=-37 rand_margin=+5

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 65 / 69 / 5 / 53 | +1 [-10, +12] | -24 [-34, -14] | +25 [+16, +34] | 131, +4, 12 / 8 | 27 / 3 / 4 | +1 (109) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 67 / 95 / 3 / 27 | -14 [-25, -3] | +12 [+4, +20] | -13 [-20, -7] | 148, -13, 8 / 21 | 2 / 14 / 2 | -12 (137) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 70 / 112 / 8 / 2 | +0 [-4, +4] | +0 [-3, +3] | +1 [+0, +3] | 181, -1, 1 / 2 | 1 / 1 / 0 | -1 (181) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 73 / 108 / 10 / 1 | +3 [-4, +10] | -1 [-3, +0] | +0 [+0, +0] | 181, +4, 8 / 4 | 1 / 0 / 1 | +1 (173) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 72 / 108 / 10 / 2 | +2 [-4, +8] | -2 [-6, +2] | +1 [+0, +3] | 179, +2, 5 / 3 | 3 / 1 / 2 | +1 (173) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 67 / 115 / 9 / 1 | -3 [-8, +1] | +0 [+0, +0] | +0 [+0, +0] | 182, -3, 1 / 4 | 0 / 0 / 0 | -3 (176) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 99 / 81 / 8 / 4 | +29 [+17, +41] | -2 [-8, +3] | +3 [-1, +8] | 177, +29, 33 / 4 | 5 / 3 / 3 | +17 (151) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 99 / 82 / 7 / 4 | +29 [+17, +41] | -1 [-6, +4] | +3 [-1, +8] | 178, +28, 32 / 4 | 4 / 3 / 2 | +17 (152) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 75 / 107 / 8 / 2 | -23 [-36, -9] | -2 [-8, +4] | -2 [-6, +2] | 178, -18, 13 / 31 | 6 / 4 / 5 | -16 (167) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 74 / 49 / 4 / 65 | -6 [-19, +7] | -37 [-49, -26] | +35 [+24, +46] | 121, +7, 17 / 10 | 39 / 2 / 2 | +7 (89) |

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

=== 2026-11-13 13:03:29 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

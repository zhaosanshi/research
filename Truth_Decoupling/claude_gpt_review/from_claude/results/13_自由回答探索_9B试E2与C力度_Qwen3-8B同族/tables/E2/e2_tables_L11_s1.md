TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 109 / 46 / 0 / 37 | 3.0 | +6 [-1, +13] | -11 [-18, -5] | +12 [+6, +19] | 155, +8 [+2, +15], 10 / 2 | 11 / 0 / 0 | 2 / 9 / 0 / 0 | +7 (150) |
| x.w.m1 | 97 / 75 / 2 / 18 | 3.5 | -6 [-14, +2] | +6 [+0, +13] | -7 [-13, -2] | 164, -6 [-14, +1], 4 / 10 | 2 / 8 / 1 | 1 / 1 / 1 / 7 | -4 (146) |
| x.md.p1 | 105 / 59 / 1 / 27 | 3.6 | +2 [-3, +8] | -2 [-6, +2] | +2 [-2, +6] | 163, +3 [-1, +8], 4 / 1 | 3 / 1 / 0 | 2 / 1 / 1 / 0 | +1 (143) |
| x.rand.p1 | 102 / 64 / 2 / 24 | 3.1 | -1 [-6, +4] | +0 [-4, +4] | -1 [-5, +2] | 164, +0 [-5, +5], 3 / 3 | 2 / 2 / 1 | 1 / 1 / 0 / 2 | +1 (162) |
| x.rand.m1 | 104 / 59 / 1 / 28 | 3.4 | +1 [+0, +3] | -3 [-7, +0] | +3 [+0, +7] | 163, +1 [+0, +3], 1 / 0 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | +1 (160) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 138 / 36 / 1 / 17; dR +4 [-4, +12]; answered-rate change -0.5 pp [-2.6, +1.0]; out / in 2 / 1; mean words 10.1 -> 10.2
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +7 [+0, +15]; vs x.rand.m1 +5 [-2, +12] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -5.7 pp [-9.4, -2.6]; moved out / in 11 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +7 on 150 questions; mean words 3.2 -> 3.0
KEY family=E judge=allenai layer=11 A=not_met B=not_met dR=+6 dR_F=+4 dAns=-11 dAns_F=-1 rand_margin=+5

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 101 / 48 / 0 / 43 | +0 [-8, +8] | -11 [-19, -4] | +13 [+6, +21] | 147, +1, 6 / 5 | 13 / 2 / 0 | +1 (141) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 107 / 53 / 1 / 31 | -10 [-18, -2] | +8 [+3, +14] | -9 [-15, -4] | 152, -11, 3 / 14 | 0 / 8 / 0 | -10 (140) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 76 / 3 / 2 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 0 / 0 | 0 / 0 / 0 | +0 (187) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 113 / 74 / 3 / 2 | +2 [-3, +7] | +0 [-3, +3] | +0 [+0, +0] | 186, +2, 3 / 1 | 1 / 1 / 1 | +1 (174) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 114 / 73 / 2 / 3 | +3 [-2, +8] | +0 [+0, +0] | +1 [+0, +3] | 187, +3, 5 / 2 | 0 / 0 / 0 | +1 (180) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 110 / 77 / 3 / 2 | -1 [-5, +2] | +0 [-3, +3] | +0 [+0, +0] | 186, -1, 0 / 1 | 1 / 1 / 1 | +0 (181) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 128 / 58 / 3 / 3 | +17 [+8, +26] | -1 [-3, +0] | +1 [+0, +3] | 186, +18, 20 / 2 | 1 / 0 / 1 | +17 (175) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 127 / 59 / 3 / 3 | +16 [+7, +25] | -1 [-3, +0] | +1 [+0, +3] | 186, +17, 19 / 2 | 1 / 0 / 1 | +16 (175) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 117 / 70 / 2 / 3 | -18 [-28, -9] | -2 [-5, +0] | +1 [+0, +3] | 187, -16, 3 / 19 | 2 / 0 / 1 | -9 (161) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 135 / 40 / 1 / 16 | +1 [-5, +7] | +0 [-3, +3] | +1 [+0, +3] | 174, +1, 4 / 3 | 1 / 1 / 0 | +1 (161) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 138 / 36 / 1 / 17 | +4 [-4, +12] | -1 [-5, +2] | +2 [+0, +5] | 173, +4, 9 / 5 | 2 / 1 / 0 | +6 (146) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 147 / 16 / 0 / 29 | +13 [+2, +24] | -12 [-21, -4] | +14 [+7, +22] | 160, +17, 19 / 2 | 15 / 3 / 0 | +10 (116) |

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

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 101 / 51 / 3 / 37 | 3.0 | +2 [-6, +10] | -11 [-19, -4] | +12 [+6, +19] | 150, +4 [-3, +11], 8 / 4 | 13 / 2 / 1 | 3 / 10 / 1 / 1 | +3 (146) |
| x.w.m1 | 90 / 81 / 3 / 18 | 3.5 | -9 [-17, -2] | +8 [+2, +14] | -7 [-13, -2] | 162, -10 [-18, -3], 2 / 12 | 1 / 9 / 0 | 0 / 1 / 1 / 8 | -6 (143) |
| x.md.p1 | 94 / 68 / 3 / 27 | 3.6 | -5 [-12, +1] | -1 [-6, +3] | +2 [-2, +6] | 160, -4 [-10, +1], 2 / 6 | 3 / 2 / 0 | 2 / 1 / 1 / 1 | -1 (140) |
| x.rand.p1 | 96 / 68 / 4 / 24 | 3.1 | -3 [-8, +1] | +1 [-2, +5] | -1 [-5, +2] | 162, -3 [-8, +1], 1 / 4 | 1 / 2 / 0 | 0 / 1 / 0 / 2 | -1 (159) |
| x.rand.m1 | 99 / 62 / 3 / 28 | 3.4 | +0 [-4, +4] | -2 [-6, +2] | +3 [+0, +7] | 160, -1 [-4, +2], 1 / 2 | 3 / 1 / 0 | 0 / 3 / 1 / 0 | -1 (157) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 118 / 52 / 5 / 17; dR +7 [+0, +14]; answered-rate change +0.5 pp [-1.6, +2.6]; out / in 2 / 3; mean words 10.1 -> 10.2
- main effect: dR +2 (1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-3, +13]; vs x.rand.m1 +2 [-5, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -5.7 pp [-9.9, -2.1]; moved out / in 13 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +3 on 146 questions; mean words 3.2 -> 3.0
KEY family=E judge=pro layer=11 A=not_met B=not_met dR=+2 dR_F=+7 dAns=-11 dAns_F=+1 rand_margin=+2

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 91 / 54 / 4 / 43 | -1 [-9, +7] | -12 [-20, -4] | +13 [+6, +21] | 142, +1, 5 / 4 | 15 / 3 / 1 | +2 (137) |
| caut_E2.wq.m1 | 109 / 39 / 3 / 41 | 100 / 57 / 3 / 32 | -9 [-17, -2] | +9 [+3, +16] | -9 [-15, -4] | 147, -10, 2 / 12 | 1 / 10 / 1 | -7 (136) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 100 / 87 / 1 / 4 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 0 / 0 | 0 / 0 / 0 | +0 (187) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 99 / 86 / 2 / 5 | -1 [-7, +5] | -2 [-5, +0] | +1 [+0, +3] | 185, +0, 4 / 4 | 2 / 0 / 1 | +1 (174) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 102 / 84 / 2 / 4 | +2 [-3, +7] | -1 [-3, +0] | +0 [+0, +0] | 186, +2, 4 / 2 | 1 / 0 / 1 | +1 (180) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 100 / 88 / 1 / 3 | +0 [-3, +3] | +1 [+0, +3] | -1 [-3, +0] | 187, -1, 0 / 1 | 0 / 1 / 0 | +0 (182) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 117 / 70 / 1 / 4 | +17 [+8, +27] | +0 [-3, +3] | +0 [-3, +3] | 186, +17, 20 / 3 | 1 / 1 / 0 | +16 (176) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 116 / 71 / 1 / 4 | +16 [+7, +26] | +0 [-3, +3] | +0 [-3, +3] | 186, +16, 19 / 3 | 1 / 1 / 0 | +15 (176) |
| conf_E2.wq.m1 | 122 / 64 / 2 / 4 | 104 / 83 / 1 / 4 | -18 [-28, -8] | +1 [-2, +5] | +0 [-3, +3] | 185, -19, 3 / 22 | 1 / 2 / 0 | -9 (159) |
| x_F.w.p0.5 | 111 / 58 / 8 / 15 | 116 / 54 / 6 / 16 | +5 [+0, +10] | +1 [-2, +5] | +1 [+0, +3] | 168, +4, 5 / 1 | 1 / 2 / 0 | +2 (156) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 118 / 52 / 5 / 17 | +7 [+0, +14] | +1 [-3, +5] | +2 [+0, +5] | 167, +5, 9 / 4 | 2 / 3 / 0 | +6 (142) |
| x_F.w.p2 | 111 / 58 / 8 / 15 | 120 / 36 / 7 / 29 | +9 [-3, +21] | -13 [-22, -4] | +14 [+7, +22] | 151, +10, 18 / 8 | 18 / 5 / 3 | +8 (112) |

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

=== 2026-11-13 10:53:21 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

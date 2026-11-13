TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=2 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 113 / 38 / 0 / 41 | 3.3 | +10 [+1, +19] | -15 [-23, -8] | +16 [+9, +24] | 151, +14 [+7, +22], 15 / 1 | 15 / 0 / 0 | 4 / 11 / 0 / 0 | +10 (139) |
| x.rand.p2 | 101 / 68 / 2 / 21 | 3.2 | -2 [-8, +3] | +3 [-1, +8] | -4 [-8, -1] | 165, -1 [-6, +4], 3 / 4 | 1 / 4 / 1 | 1 / 0 / 0 / 4 | -1 (156) |
| x.rand.m2 | 105 / 56 / 2 / 29 | 3.1 | +2 [-3, +7] | -5 [-10, -1] | +4 [+1, +8] | 161, +3 [-1, +7], 4 / 1 | 5 / 0 / 1 | 1 / 4 / 0 / 0 | +2 (158) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 141 / 27 / 0 / 24; dR +7 [-2, +16]; answered-rate change -3.6 pp [-7.3, +0.0]; out / in 10 / 3; mean words 10.1 -> 9.9
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +12 [+2, +22]; vs x.rand.m2 +8 [-1, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -7.8 pp [-12.0, -4.2]; moved out / in 15 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +10 on 139 questions; mean words 3.2 -> 3.3
KEY family=E judge=allenai layer=9 A=not_met B=not_met dR=+10 dR_F=+7 dAns=-15 dAns_F=-7 rand_margin=+8 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 9

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 103 / 50 / 0 / 39 | +2 [-5, +9] | -7 [-14, -1] | +9 [+3, +16] | 151, +1, 5 / 4 | 9 / 2 / 0 | +1 (144) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 108 / 50 / 2 / 32 | -9 [-16, -2] | +6 [+1, +12] | -8 [-14, -3] | 151, -8, 2 / 10 | 1 / 7 / 1 | -8 (142) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 110 / 77 / 3 / 2 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 0 / 0 | +0 (185) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 108 / 78 / 3 / 3 | -3 [-10, +4] | -1 [-4, +2] | +1 [+0, +3] | 185, -2, 4 / 6 | 2 / 1 / 1 | -1 (171) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 114 / 73 / 3 / 2 | +3 [-2, +8] | +0 [+0, +0] | +0 [+0, +0] | 187, +3, 5 / 2 | 0 / 0 / 0 | +0 (179) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 111 / 75 / 4 / 2 | +0 [-4, +4] | -1 [-3, +0] | +0 [+0, +0] | 186, +1, 2 / 1 | 1 / 0 / 1 | +1 (182) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 126 / 61 / 2 / 3 | +15 [+8, +23] | +0 [+0, +0] | +1 [+0, +3] | 187, +15, 16 / 1 | 0 / 0 / 0 | +14 (177) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 127 / 60 / 2 / 3 | +16 [+8, +25] | +0 [+0, +0] | +1 [+0, +3] | 187, +16, 17 / 1 | 0 / 0 / 0 | +15 (177) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 121 / 67 / 2 / 2 | -14 [-22, -6] | -1 [-3, +0] | +0 [+0, +0] | 188, -13, 2 / 15 | 1 / 0 / 1 | -7 (169) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 133 / 42 / 1 / 16 | -1 [-7, +5] | +0 [-3, +3] | +1 [+0, +3] | 174, -1, 3 / 4 | 1 / 1 / 0 | +1 (162) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 134 / 40 / 1 / 17 | +0 [-8, +8] | -1 [-6, +3] | +2 [-2, +6] | 172, +0, 6 / 6 | 3 / 2 / 0 | +2 (148) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 141 / 27 / 0 / 24 | +7 [-2, +16] | -7 [-14, +0] | +9 [+3, +16] | 165, +7, 12 / 5 | 10 / 3 / 0 | +6 (130) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 103 / 44 / 4 / 41 | 3.3 | +4 [-5, +13] | -16 [-25, -7] | +16 [+9, +24] | 144, +6 [-2, +14], 11 / 5 | 19 / 3 / 3 | 4 / 15 / 2 / 1 | +5 (134) |
| x.rand.p2 | 93 / 75 / 3 / 21 | 3.2 | -6 [-13, +0] | +5 [+1, +10] | -4 [-8, -1] | 163, -7 [-13, -2], 1 / 8 | 0 / 5 / 0 | 0 / 0 / 1 / 4 | -3 (153) |
| x.rand.m2 | 101 / 59 / 3 / 29 | 3.1 | +2 [-4, +8] | -3 [-8, +1] | +4 [+1, +8] | 159, +1 [-5, +7], 5 / 4 | 4 / 1 / 0 | 0 / 4 / 1 / 0 | +0 (156) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 113 / 47 / 8 / 24; dR +2 [-9, +13]; answered-rate change -4.7 pp [-9.4, -0.5]; out / in 14 / 5; mean words 10.1 -> 9.9
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p2 +10 [+0, +20]; vs x.rand.m2 +2 [-7, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -8.3 pp [-13.0, -3.6]; moved out / in 19 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 134 questions; mean words 3.2 -> 3.3
KEY family=E judge=pro layer=9 A=not_met B=not_met dR=+4 dR_F=+2 dAns=-16 dAns_F=-9 rand_margin=+2 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 9

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 92 / 57 / 4 / 39 | +0 [-7, +7] | -8 [-15, -1] | +9 [+3, +16] | 146, +0, 5 / 5 | 11 / 3 / 1 | +1 (140) |
| caut_E2.wq.m1 | 109 / 39 / 3 / 41 | 101 / 56 / 2 / 33 | -8 [-15, -1] | +9 [+4, +15] | -8 [-14, -3] | 148, -9, 2 / 11 | 0 / 9 / 0 | -7 (139) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 99 / 88 / 1 / 4 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 0 / 0 | +0 (185) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 96 / 89 / 2 / 5 | -4 [-10, +1] | -2 [-5, +0] | +1 [+0, +3] | 185, -3, 2 / 5 | 2 / 0 / 1 | +0 (172) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 102 / 84 / 2 / 4 | +2 [-2, +6] | -1 [-3, +0] | +0 [+0, +0] | 186, +2, 3 / 1 | 1 / 0 / 1 | +1 (179) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 102 / 86 / 1 / 3 | +2 [-3, +7] | +1 [+0, +3] | -1 [-3, +0] | 187, +1, 3 / 2 | 0 / 1 / 0 | +1 (183) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 116 / 71 / 1 / 4 | +16 [+7, +25] | +0 [-3, +3] | +0 [-3, +3] | 186, +16, 18 / 2 | 1 / 1 / 0 | +15 (177) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 116 / 71 / 1 / 4 | +16 [+7, +26] | +0 [-3, +3] | +0 [-3, +3] | 186, +16, 19 / 3 | 1 / 1 / 0 | +16 (177) |
| conf_E2.wq.m1 | 122 / 64 / 2 / 4 | 107 / 80 / 1 / 4 | -15 [-24, -6] | +1 [-2, +5] | +0 [-3, +3] | 185, -16, 2 / 18 | 1 / 2 / 0 | -8 (166) |
| x_F.w.p0.5 | 111 / 58 / 8 / 15 | 115 / 55 / 6 / 16 | +4 [+0, +9] | +1 [-2, +5] | +1 [+0, +3] | 168, +3, 4 / 1 | 1 / 2 / 0 | +2 (157) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 112 / 58 / 5 / 17 | +1 [-6, +8] | +1 [-3, +6] | +2 [+0, +5] | 167, +0, 6 / 6 | 2 / 3 / 0 | +0 (145) |
| x_F.w.p2 | 111 / 58 / 8 / 15 | 113 / 47 / 8 / 24 | +2 [-9, +13] | -9 [-18, -1] | +9 [+3, +16] | 155, +2, 12 / 10 | 14 / 5 / 4 | +3 (126) |

=== 2026-11-13 10:50:35 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=2 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 109 / 33 / 0 / 50 | 2.8 | +6 [-3, +15] | -24 [-34, -15] | +25 [+16, +35] | 141, +11 [+5, +18], 12 / 1 | 25 / 1 / 0 | 6 / 19 / 1 / 0 | +11 (134) |
| x.rand.p2 | 98 / 72 / 2 / 20 | 3.3 | -5 [-11, +1] | +4 [+0, +9] | -5 [-10, -1] | 165, -5 [-10, +0], 1 / 6 | 1 / 5 / 1 | 1 / 0 / 1 / 4 | -3 (154) |
| x.rand.m2 | 102 / 56 / 1 / 33 | 3.5 | -1 [-6, +4] | -8 [-14, -3] | +8 [+3, +14] | 158, +2 [-2, +6], 3 / 1 | 8 / 0 / 1 | 3 / 5 / 0 / 0 | +1 (151) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 136 / 26 / 0 / 30; dR +2 [-8, +12]; answered-rate change -6.8 pp [-11.5, -2.6]; out / in 16 / 3; mean words 10.1 -> 9.7
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p2 +11 [+0, +22]; vs x.rand.m2 +7 [-2, +15] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -12.5 pp [-17.7, -7.8]; moved out / in 25 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +11 on 134 questions; mean words 3.2 -> 2.8
KEY family=E judge=allenai layer=15 A=not_met B=not_met dR=+6 dR_F=+2 dAns=-24 dAns_F=-13 rand_margin=+7 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 15

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 104 / 45 / 0 / 43 | +3 [-3, +10] | -11 [-19, -3] | +13 [+5, +21] | 146, +2, 4 / 2 | 14 / 3 / 0 | +2 (138) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 108 / 56 / 2 / 26 | -9 [-18, -1] | +12 [+5, +20] | -14 [-21, -7] | 151, -10, 3 / 13 | 1 / 13 / 1 | -10 (137) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 110 / 77 / 3 / 2 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 123 / 65 / 2 / 2 | +12 [+5, +20] | +1 [+0, +3] | +0 [+0, +0] | 187, +11, 13 / 2 | 0 / 1 / 0 | +9 (168) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 116 / 72 / 2 / 2 | +5 [+0, +10] | +1 [+0, +3] | +0 [+0, +0] | 187, +4, 5 / 1 | 0 / 1 / 0 | +2 (180) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 106 / 81 / 3 / 2 | -5 [-10, -1] | +0 [+0, +0] | +0 [+0, +0] | 187, -5, 0 / 5 | 0 / 0 / 0 | -3 (178) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 129 / 58 / 3 / 2 | +18 [+9, +27] | +0 [+0, +0] | +0 [+0, +0] | 187, +18, 20 / 2 | 0 / 0 / 0 | +16 (175) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 131 / 56 / 3 / 2 | +20 [+12, +29] | +0 [+0, +0] | +0 [+0, +0] | 187, +20, 21 / 1 | 0 / 0 / 0 | +17 (176) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 120 / 67 / 2 / 3 | -15 [-25, -5] | -2 [-5, +0] | +1 [+0, +3] | 187, -13, 5 / 18 | 2 / 0 / 1 | -9 (162) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 135 / 39 / 2 / 16 | +1 [-4, +6] | -1 [-3, +0] | +1 [+0, +3] | 174, +2, 4 / 2 | 1 / 0 / 0 | +3 (158) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 136 / 33 / 2 / 21 | +2 [-5, +9] | -6 [-11, -2] | +6 [+2, +11] | 169, +5, 8 / 3 | 6 / 0 / 0 | +5 (140) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 136 / 26 / 0 / 30 | +2 [-8, +12] | -13 [-22, -5] | +15 [+8, +23] | 159, +8, 11 / 3 | 16 / 3 / 0 | +4 (123) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 98 / 41 / 3 / 50 | 2.8 | -1 [-12, +10] | -24 [-35, -14] | +25 [+16, +35] | 136, +5 [-4, +14], 12 / 7 | 27 / 3 / 2 | 8 / 19 / 2 / 1 | +5 (129) |
| x.rand.p2 | 92 / 77 / 3 / 20 | 3.3 | -7 [-14, -1] | +6 [+2, +11] | -5 [-10, -1] | 163, -8 [-15, -2], 1 / 9 | 0 / 6 / 0 | 0 / 0 / 1 / 5 | -4 (151) |
| x.rand.m2 | 99 / 56 / 4 / 33 | 3.5 | +0 [-6, +6] | -8 [-14, -3] | +8 [+3, +14] | 155, +2 [-3, +8], 5 / 3 | 8 / 0 / 0 | 2 / 6 / 0 / 0 | +2 (147) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 118 / 40 / 5 / 29; dR +7 [-5, +19]; answered-rate change -5.7 pp [-10.4, -1.0]; out / in 17 / 6; mean words 10.1 -> 9.7
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p2 +6 [-6, +17]; vs x.rand.m2 -1 [-11, +9] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -12.5 pp [-18.2, -7.3]; moved out / in 27 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 129 questions; mean words 3.2 -> 2.8
KEY family=E judge=pro layer=15 A=not_met B=not_met dR=-1 dR_F=+7 dAns=-24 dAns_F=-11 rand_margin=-1 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 15

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 96 / 52 / 1 / 43 | +4 [-3, +11] | -9 [-18, -1] | +13 [+5, +21] | 143, +1, 5 / 4 | 14 / 5 / 0 | +1 (136) |
| caut_E2.wq.m1 | 109 / 39 / 3 / 41 | 100 / 62 / 3 / 27 | -9 [-17, -1] | +14 [+7, +22] | -14 [-21, -7] | 147, -11, 2 / 13 | 1 / 15 / 1 | -8 (134) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 99 / 88 / 1 / 4 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 110 / 77 / 1 / 4 | +10 [+3, +18] | +0 [-3, +3] | +0 [-3, +3] | 186, +10, 12 / 2 | 1 / 1 / 0 | +9 (168) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 103 / 84 / 2 / 3 | +3 [+0, +7] | +0 [-3, +3] | -1 [-3, +0] | 186, +2, 2 / 0 | 1 / 1 / 1 | +2 (181) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 94 / 91 / 3 / 4 | -6 [-11, -2] | -2 [-5, +0] | +0 [+0, +0] | 185, -4, 0 / 4 | 2 / 0 / 2 | -2 (178) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 118 / 69 / 1 / 4 | +18 [+10, +27] | +0 [-3, +3] | +0 [-3, +3] | 186, +18, 19 / 1 | 1 / 1 / 0 | +16 (175) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 119 / 67 / 1 / 5 | +19 [+11, +28] | -1 [-3, +0] | +1 [+0, +3] | 186, +20, 20 / 0 | 1 / 0 / 0 | +18 (176) |
| conf_E2.wq.m1 | 122 / 64 / 2 / 4 | 103 / 81 / 4 / 4 | -19 [-29, -10] | -2 [-6, +2] | +0 [-3, +3] | 183, -19, 2 / 21 | 3 / 1 / 2 | -9 (160) |
| x_F.w.p0.5 | 111 / 58 / 8 / 15 | 114 / 57 / 5 / 16 | +3 [-3, +9] | +2 [-2, +6] | +1 [+0, +3] | 168, +2, 5 / 3 | 1 / 3 / 0 | +3 (155) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 115 / 51 / 5 / 21 | +4 [-5, +13] | -3 [-10, +3] | +6 [+2, +11] | 162, +4, 10 / 6 | 7 / 4 / 1 | +6 (137) |
| x_F.w.p2 | 111 / 58 / 8 / 15 | 118 / 40 / 5 / 29 | +7 [-5, +19] | -11 [-20, -2] | +14 [+7, +22] | 152, +8, 15 / 7 | 17 / 6 / 3 | +4 (119) |

=== 2026-11-13 11:10:37 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

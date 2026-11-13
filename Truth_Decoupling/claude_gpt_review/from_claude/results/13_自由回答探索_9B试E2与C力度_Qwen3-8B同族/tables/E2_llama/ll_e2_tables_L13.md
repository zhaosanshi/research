TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 76 / 51 / 0 / 65 | 4.0 | -2 [-15, +11] | -39 [-51, -27] | +42 [+31, +54] | 125, +12 [+3, +22], 18 / 6 | 41 / 2 / 0 | 16 / 25 / 2 / 0 | +5 (105) |
| x.w.m1 | 78 / 101 / 2 / 11 | 3.9 | +0 [-11, +10] | +13 [+5, +21] | -12 [-19, -6] | 164, -5 [-14, +3], 7 / 12 | 2 / 15 / 2 | 2 / 0 / 7 / 8 | -4 (156) |
| x.md.p1 | 79 / 66 / 2 / 45 | 4.5 | +1 [-11, +13] | -21 [-31, -12] | +22 [+14, +31] | 143, +10 [+0, +20], 18 / 8 | 23 / 2 / 1 | 11 / 12 / 2 / 0 | +4 (125) |
| x.rand.p1 | 77 / 89 / 3 / 23 | 4.2 | -1 [-7, +5] | +0 [+0, +0] | +0 [+0, +0] | 166, -1 [-7, +5], 4 / 5 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -2 (158) |
| x.rand.m1 | 80 / 85 / 2 / 25 | 4.4 | +2 [-6, +10] | -1 [-5, +3] | +2 [-2, +6] | 163, +2 [-5, +9], 7 / 5 | 3 / 2 / 0 | 2 / 1 / 2 / 0 | -1 (154) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 80 / 37 / 0 / 75; dR -22 [-35, -9]; answered-rate change -22.9 pp [-29.2, -17.2]; out / in 44 / 0; mean words 9.3 -> 7.5
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -1 [-14, +11]; vs x.rand.m1 -4 [-17, +8] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -20.3 pp [-26.6, -14.1]; moved out / in 41 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 105 questions; mean words 4.0 -> 4.0
KEY family=E judge=allenai layer=13 A=not_met B=not_met dR=-2 dR_F=-22 dAns=-39 dAns_F=-44 rand_margin=-4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 13

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 72 / 45 / 1 / 74 | -10 [-22, +2] | -47 [-59, -35] | +47 [+36, +59] | 116, +9, 13 / 4 | 48 / 1 / 1 | +5 (98) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 94 / 79 / 1 / 18 | -8 [-20, +4] | +18 [+10, +27] | -19 [-27, -11] | 154, -15, 6 / 21 | 1 / 19 / 1 | -13 (147) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 87 / 102 / 1 / 2 | -1 [-4, +2] | +0 [-3, +3] | +0 [+0, +0] | 188, +0, 1 / 1 | 1 / 1 / 1 | +1 (186) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 95 / 93 / 2 / 2 | +7 [-2, +17] | -1 [-5, +2] | +0 [+0, +0] | 187, +9, 16 / 7 | 2 / 1 / 2 | +6 (168) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 85 / 103 / 2 / 2 | -3 [-8, +2] | -1 [-3, +0] | +0 [+0, +0] | 188, -2, 2 / 4 | 1 / 0 / 1 | -1 (182) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 82 / 107 / 1 / 2 | -6 [-12, +0] | +0 [-3, +3] | +0 [+0, +0] | 188, -5, 2 / 7 | 1 / 1 / 1 | -4 (187) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 104 / 85 / 2 / 1 | +16 [+5, +27] | +0 [-4, +4] | -1 [-3, +0] | 187, +16, 23 / 7 | 2 / 2 / 2 | +12 (165) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 110 / 81 / 1 / 0 | +22 [+11, +33] | +2 [-2, +6] | -2 [-5, +0] | 188, +20, 26 / 6 | 1 / 3 / 1 | +14 (164) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 93 / 99 / 0 / 0 | -24 [-36, -12] | +1 [+0, +3] | +0 [+0, +0] | 191, -24, 7 / 31 | 0 / 1 / 0 | -20 (176) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 80 / 37 / 0 / 75 | -22 [-35, -9] | -44 [-56, -33] | +45 [+34, +57] | 117, +4, 12 / 8 | 44 / 0 / 0 | +3 (88) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 104 / 85 / 2 / 1 | 3.9 | +16 [+5, +27] | +0 [-4, +4] | -1 [-3, +0] | 187, +16 [+6, +26], 23 / 7 | 2 / 2 / 2 | 2 / 0 / 2 / 0 | +12 (165) |
| conf.md.p1 | 95 / 93 / 2 / 2 | 3.9 | +7 [-2, +17] | -1 [-5, +2] | +0 [+0, +0] | 187, +9 [+0, +18], 16 / 7 | 2 / 1 / 2 | 2 / 0 / 0 / 1 | +6 (168) |
| conf.rand.p1 | 82 / 107 / 1 / 2 | 3.4 | -6 [-12, +0] | +0 [-3, +3] | +0 [+0, +0] | 188, -5 [-11, +1], 2 / 7 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | -4 (187) |
| conf.rand.m1 | 85 / 103 / 2 / 2 | 3.4 | -3 [-8, +2] | -1 [-3, +0] | +0 [+0, +0] | 188, -2 [-7, +3], 2 / 4 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -1 (182) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +16 (8.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +22 [+11, +33]; vs conf.rand.m1 +19 [+8, +30] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-2.1, +2.1]; moved out / in 2 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +12 on 165 questions; mean words 3.4 -> 3.9
KEY family=E judge=allenai layer=13 A=incomplete B=incomplete dR=+16 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+19 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 64 / 4 / 66 | 4.0 | -6 [-17, +5] | -40 [-52, -28] | +42 [+31, +53] | 119, +3 [-5, +11], 9 / 6 | 43 / 3 / 3 | 12 / 31 / 3 / 0 | +2 (102) |
| x.w.m1 | 63 / 109 / 8 / 12 | 3.9 | -1 [-10, +8] | +10 [+3, +18] | -12 [-19, -6] | 160, -2 [-10, +6], 7 / 9 | 2 / 12 / 2 | 2 / 0 / 3 / 9 | -2 (154) |
| x.md.p1 | 63 / 77 / 6 / 46 | 4.5 | -1 [-11, +9] | -22 [-32, -12] | +22 [+14, +31] | 137, +3 [-5, +11], 9 / 6 | 25 / 3 / 4 | 7 / 18 / 3 / 0 | +3 (121) |
| x.rand.p1 | 65 / 97 / 6 / 24 | 4.2 | +1 [-5, +7] | +0 [+0, +0] | +0 [+0, +0] | 162, +1 [-5, +7], 5 / 4 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +4 (154) |
| x.rand.m1 | 64 / 97 / 5 / 26 | 4.4 | +0 [-6, +6] | -1 [-5, +3] | +2 [-2, +6] | 159, -2 [-7, +3], 2 / 4 | 3 / 2 / 0 | 0 / 3 / 2 / 0 | -1 (151) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 65 / 47 / 4 / 76; dR -15 [-28, -2]; answered-rate change -25.0 pp [-31.2, -19.3]; out / in 48 / 0; mean words 9.3 -> 7.5
- main effect: dR -6 (-3.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -7 [-19, +5]; vs x.rand.m1 -6 [-16, +4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -20.8 pp [-27.1, -14.6]; moved out / in 43 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +2 on 102 questions; mean words 4.0 -> 4.0
KEY family=E judge=pro layer=13 A=not_met B=not_met dR=-6 dR_F=-15 dAns=-40 dAns_F=-48 rand_margin=-7

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 13

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 56 / 56 / 5 / 75 | -8 [-19, +3] | -46 [-59, -34] | +47 [+36, +59] | 109, +3, 10 / 7 | 49 / 3 / 4 | +3 (93) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 73 / 92 / 6 / 21 | -8 [-19, +3] | +15 [+7, +24] | -19 [-28, -11] | 148, -10, 9 / 19 | 2 / 17 / 2 | -9 (141) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 69 / 112 / 9 / 2 | -1 [-5, +3] | -1 [-5, +2] | +1 [+0, +3] | 180, -1, 1 / 2 | 2 / 1 / 1 | -1 (178) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 83 / 98 / 8 / 3 | +13 [+3, +23] | -1 [-6, +4] | +2 [+0, +5] | 178, +12, 18 / 6 | 4 / 3 / 3 | +4 (162) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 67 / 114 / 10 / 1 | -3 [-9, +3] | -1 [-3, +0] | +0 [+0, +0] | 181, -2, 3 / 5 | 1 / 0 / 1 | -4 (175) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 69 / 112 / 9 / 2 | -1 [-8, +5] | -1 [-5, +2] | +1 [+0, +3] | 180, -1, 4 / 5 | 2 / 1 / 1 | -1 (179) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 83 / 95 / 9 / 5 | +13 [+2, +24] | -4 [-10, +1] | +4 [+0, +9] | 176, +14, 22 / 8 | 6 / 2 / 3 | +6 (157) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 92 / 89 / 7 / 4 | +22 [+11, +33] | -1 [-7, +5] | +3 [-1, +8] | 177, +20, 26 / 6 | 5 / 4 / 2 | +11 (156) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 75 / 108 / 8 / 1 | -23 [-36, -10] | -1 [-6, +4] | -3 [-7, +0] | 180, -19, 12 / 31 | 4 / 3 / 4 | -18 (166) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 65 / 47 / 4 / 76 | -15 [-28, -2] | -48 [-60, -37] | +46 [+35, +58] | 112, +6, 16 / 10 | 48 / 0 / 2 | +6 (85) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 83 / 95 / 9 / 5 | 3.9 | +13 [+2, +24] | -4 [-10, +1] | +4 [+0, +9] | 176, +14 [+4, +24], 22 / 8 | 6 / 2 / 3 | 3 / 3 / 2 / 0 | +6 (157) |
| conf.md.p1 | 83 / 98 / 8 / 3 | 3.9 | +13 [+3, +23] | -1 [-6, +4] | +2 [+0, +5] | 178, +12 [+3, +21], 18 / 6 | 4 / 3 / 3 | 2 / 2 / 3 / 0 | +4 (162) |
| conf.rand.p1 | 69 / 112 / 9 / 2 | 3.4 | -1 [-8, +5] | -1 [-5, +2] | +1 [+0, +3] | 180, -1 [-7, +5], 4 / 5 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -1 (179) |
| conf.rand.m1 | 67 / 114 / 10 / 1 | 3.4 | -3 [-9, +3] | -1 [-3, +0] | +0 [+0, +0] | 181, -2 [-8, +3], 3 / 5 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -4 (175) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +13 (6.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +14 [+2, +26]; vs conf.rand.m1 +16 [+6, +27] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.1 pp [-5.2, +0.5]; moved out / in 6 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 157 questions; mean words 3.4 -> 3.9
KEY family=E judge=pro layer=13 A=incomplete B=not_met dR=+13 dR_F=NA dAns=-4 dAns_F=NA rand_margin=+14 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 13:23:18 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

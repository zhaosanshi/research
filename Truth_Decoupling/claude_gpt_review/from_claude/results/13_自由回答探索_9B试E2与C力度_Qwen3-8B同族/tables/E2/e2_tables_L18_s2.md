TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=2 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 120 / 45 / 0 / 27 | 3.0 | +17 [+9, +26] | -1 [-6, +4] | +2 [-3, +7] | 162, +14 [+6, +22], 16 / 2 | 4 / 3 / 0 | 0 / 4 / 3 / 0 | +13 (155) |
| x.rand.p2 | 103 / 63 / 2 / 24 | 3.3 | +0 [-5, +5] | +0 [-3, +3] | -1 [-3, +0] | 165, +1 [-3, +6], 3 / 2 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (161) |
| x.rand.m2 | 100 / 61 / 1 / 30 | 3.2 | -3 [-8, +1] | -5 [-10, -1] | +5 [+1, +10] | 161, -2 [-6, +2], 1 / 3 | 5 / 0 / 0 | 1 / 4 / 0 / 0 | +0 (154) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 135 / 33 / 0 / 24; dR +1 [-8, +10]; answered-rate change -3.6 pp [-7.3, -0.5]; out / in 9 / 2; mean words 10.1 -> 9.9
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +17 [+9, +26]; vs x.rand.m2 +20 [+11, +29] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-3.1, +2.1]; moved out / in 4 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +13 on 155 questions; mean words 3.2 -> 3.0
KEY family=E judge=allenai layer=18 A=not_met B=not_met dR=+17 dR_F=+1 dAns=-1 dAns_F=-7 rand_margin=+17 EXPLORATORY base=x main=x.w.p2 strength=2

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

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 113 / 49 / 3 / 27 | 3.0 | +14 [+4, +24] | -1 [-7, +5] | +2 [-3, +7] | 158, +12 [+3, +21], 17 / 5 | 5 / 4 / 1 | 1 / 4 / 3 / 1 | +9 (151) |
| x.rand.p2 | 97 / 67 / 4 / 24 | 3.3 | -2 [-7, +3] | +1 [+0, +3] | -1 [-3, +0] | 163, -2 [-7, +3], 2 / 4 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -1 (158) |
| x.rand.m2 | 93 / 67 / 2 / 30 | 3.2 | -6 [-12, -1] | -3 [-8, +2] | +5 [+1, +10] | 158, -6 [-11, -2], 0 / 6 | 5 / 2 / 0 | 1 / 4 / 1 / 1 | -3 (151) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 107 / 51 / 10 / 24; dR -4 [-13, +5]; answered-rate change -5.7 pp [-9.4, -2.6]; out / in 12 / 1; mean words 10.1 -> 9.9
- main effect: dR +14 (7.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +16 [+6, +26]; vs x.rand.m2 +20 [+11, +30] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-3.6, +2.6]; moved out / in 5 / 4 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +9 on 151 questions; mean words 3.2 -> 3.0
KEY family=E judge=pro layer=18 A=not_met B=not_met dR=+14 dR_F=-4 dAns=-1 dAns_F=-11 rand_margin=+16 EXPLORATORY base=x main=x.w.p2 strength=2

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

=== 2026-11-13 10:35:00 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

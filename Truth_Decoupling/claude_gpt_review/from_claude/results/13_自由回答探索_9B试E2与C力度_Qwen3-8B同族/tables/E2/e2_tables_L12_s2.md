TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=2 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 107 / 29 / 0 / 56 | 3.3 | +4 [-6, +14] | -30 [-40, -21] | +31 [+21, +41] | 136, +12 [+5, +20], 14 / 2 | 30 / 0 / 0 | 8 / 22 / 0 / 0 | +11 (123) |
| x.rand.p2 | 108 / 57 / 0 / 27 | 3.4 | +5 [+0, +11] | -1 [-5, +3] | +2 [-2, +6] | 163, +5 [+0, +11], 7 / 2 | 3 / 2 / 0 | 0 / 3 / 0 / 2 | +4 (153) |
| x.rand.m2 | 98 / 63 / 2 / 29 | 3.3 | -5 [-11, +0] | -5 [-11, +0] | +4 [+0, +9] | 160, -1 [-5, +2], 1 / 2 | 6 / 1 / 1 | 4 / 2 / 0 / 1 | +1 (157) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 144 / 20 / 0 / 28; dR +10 [-1, +21]; answered-rate change -5.7 pp [-10.4, -1.0]; out / in 16 / 5; mean words 10.1 -> 10.0
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p2 -1 [-10, +8]; vs x.rand.m2 +9 [-1, +19] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -15.6 pp [-20.8, -10.9]; moved out / in 30 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +11 on 123 questions; mean words 3.2 -> 3.3
KEY family=E judge=allenai layer=12 A=not_met B=not_met dR=+4 dR_F=+10 dAns=-30 dAns_F=-11 rand_margin=-1 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 100 / 47 / 0 / 45 | -1 [-9, +7] | -13 [-21, -6] | +15 [+8, +23] | 145, +1, 6 / 5 | 15 / 2 / 0 | +1 (139) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 108 / 53 / 1 / 30 | -9 [-17, -1] | +9 [+4, +15] | -10 [-16, -4] | 152, -10, 3 / 13 | 0 / 9 / 0 | -9 (137) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 76 / 3 / 2 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 0 / 0 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 112 / 74 / 3 / 3 | +1 [-4, +6] | -1 [-3, +0] | +1 [+0, +3] | 186, +2, 4 / 2 | 1 / 0 / 1 | +2 (178) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 112 / 74 / 2 / 4 | +1 [-4, +6] | -1 [-3, +0] | +2 [+0, +5] | 186, +2, 4 / 2 | 1 / 0 / 0 | +2 (180) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 113 / 75 / 2 / 2 | +2 [-3, +7] | +1 [+0, +3] | +0 [+0, +0] | 187, +1, 3 / 2 | 0 / 1 / 0 | +0 (183) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 126 / 60 / 3 / 3 | +15 [+8, +23] | -1 [-3, +0] | +1 [+0, +3] | 186, +16, 16 / 0 | 1 / 0 / 1 | +14 (178) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 127 / 59 / 3 / 3 | +16 [+8, +24] | -1 [-3, +0] | +1 [+0, +3] | 186, +17, 17 / 0 | 1 / 0 / 1 | +14 (175) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 118 / 69 / 2 / 3 | -17 [-27, -8] | -2 [-5, +0] | +1 [+0, +3] | 187, -15, 3 / 18 | 2 / 0 / 1 | -10 (163) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 136 / 39 / 1 / 16 | +2 [-2, +7] | +0 [-3, +3] | +1 [+0, +3] | 174, +2, 3 / 1 | 1 / 1 / 0 | +2 (163) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 137 / 37 / 2 / 16 | +3 [-4, +11] | -1 [-6, +3] | +1 [-2, +5] | 172, +2, 7 / 5 | 3 / 2 / 1 | +3 (147) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 144 / 20 / 0 / 28 | +10 [-1, +21] | -11 [-20, -2] | +13 [+5, +22] | 159, +12, 17 / 5 | 16 / 5 / 0 | +8 (124) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 99 / 33 / 4 / 56 | 3.3 | +0 [-11, +10] | -31 [-43, -20] | +31 [+21, +41] | 129, +9 [+2, +17], 12 / 3 | 34 / 3 / 3 | 11 / 23 / 2 / 1 | +8 (119) |
| x.rand.p2 | 100 / 62 / 3 / 27 | 3.4 | +1 [-5, +7] | -1 [-6, +4] | +2 [-2, +6] | 159, +0 [-5, +6], 4 / 4 | 4 / 3 / 1 | 0 / 4 / 1 / 2 | +1 (148) |
| x.rand.m2 | 95 / 64 / 4 / 29 | 3.3 | -4 [-9, +1] | -4 [-9, +0] | +4 [+0, +9] | 158, -1 [-5, +2], 1 / 2 | 5 / 1 / 0 | 3 / 2 / 0 / 1 | +0 (154) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 117 / 38 / 8 / 29; dR +6 [-6, +18]; answered-rate change -7.3 pp [-12.5, -2.1]; out / in 21 / 7; mean words 10.1 -> 10.0
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p2 -1 [-11, +9]; vs x.rand.m2 +4 [-6, +14] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -16.1 pp [-22.4, -10.4]; moved out / in 34 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 119 questions; mean words 3.2 -> 3.3
KEY family=E judge=pro layer=12 A=not_met B=not_met dR=+0 dR_F=+6 dAns=-31 dAns_F=-14 rand_margin=-1 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 92 / 52 / 3 / 45 | +0 [-8, +8] | -13 [-22, -5] | +15 [+8, +23] | 141, +2, 6 / 4 | 16 / 3 / 1 | +3 (136) |
| caut_E2.wq.m1 | 109 / 39 / 3 / 41 | 100 / 58 / 3 / 31 | -9 [-17, -2] | +10 [+4, +17] | -10 [-16, -4] | 147, -10, 2 / 12 | 1 / 11 / 1 | -6 (133) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 100 / 87 / 1 / 4 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 0 / 0 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 99 / 86 / 3 / 4 | -1 [-7, +5] | -2 [-6, +2] | +0 [-3, +3] | 184, +0, 3 / 3 | 3 / 1 / 2 | +2 (177) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 98 / 87 / 2 / 5 | -2 [-6, +2] | -2 [-5, +0] | +1 [+0, +3] | 185, -2, 1 / 3 | 2 / 0 / 1 | +0 (180) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 102 / 85 / 1 / 4 | +2 [-2, +6] | +0 [+0, +0] | +0 [+0, +0] | 187, +2, 3 / 1 | 0 / 0 / 0 | +1 (183) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 114 / 72 / 2 / 4 | +14 [+5, +24] | -1 [-5, +2] | +0 [-3, +3] | 185, +15, 18 / 3 | 2 / 1 / 1 | +14 (178) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 115 / 72 / 1 / 4 | +15 [+6, +25] | +0 [-3, +3] | +0 [-3, +3] | 186, +15, 19 / 4 | 1 / 1 / 0 | +14 (176) |
| conf_E2.wq.m1 | 122 / 64 / 2 / 4 | 105 / 81 / 2 / 4 | -17 [-28, -7] | +0 [-4, +4] | +0 [-3, +3] | 184, -18, 3 / 21 | 2 / 2 / 1 | -9 (161) |
| x_F.w.p0.5 | 111 / 58 / 8 / 15 | 114 / 57 / 5 / 16 | +3 [-2, +8] | +2 [-2, +6] | +1 [+0, +3] | 168, +2, 4 / 2 | 1 / 3 / 0 | +2 (159) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 114 / 54 / 7 / 17 | +3 [-5, +11] | -1 [-6, +4] | +2 [+0, +5] | 165, +3, 9 / 6 | 4 / 3 / 2 | +5 (143) |
| x_F.w.p2 | 111 / 58 / 8 / 15 | 117 / 38 / 8 / 29 | +6 [-6, +18] | -14 [-24, -4] | +14 [+6, +22] | 148, +6, 16 / 10 | 21 / 7 / 5 | +7 (118) |

=== 2026-11-13 11:09:07 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

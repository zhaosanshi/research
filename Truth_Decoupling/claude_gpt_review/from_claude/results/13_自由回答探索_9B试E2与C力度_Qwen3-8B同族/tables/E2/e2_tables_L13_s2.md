TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=2 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 104 / 30 / 0 / 58 | 3.4 | +1 [-9, +10] | -32 [-42, -22] | +33 [+23, +43] | 134, +10 [+3, +17], 12 / 2 | 32 / 0 / 0 | 9 / 23 / 0 / 0 | +10 (122) |
| x.rand.p2 | 103 / 61 / 1 / 27 | 3.3 | +0 [-3, +3] | -2 [-5, +0] | +2 [+0, +5] | 164, +0 [-3, +3], 1 / 1 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | +0 (157) |
| x.rand.m2 | 103 / 64 / 1 / 24 | 3.2 | +0 [-5, +5] | +1 [-2, +5] | -1 [-5, +2] | 165, +0 [-5, +5], 3 / 3 | 1 / 2 / 0 | 0 / 1 / 0 / 2 | +0 (162) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 145 / 19 / 0 / 28; dR +11 [+0, +22]; answered-rate change -5.7 pp [-10.4, -1.6]; out / in 15 / 4; mean words 10.1 -> 10.4
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p2 +1 [-8, +10]; vs x.rand.m2 +1 [-9, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -16.7 pp [-21.9, -11.5]; moved out / in 32 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +10 on 122 questions; mean words 3.2 -> 3.4
KEY family=E judge=allenai layer=13 A=not_met B=not_met dR=+1 dR_F=+11 dAns=-32 dAns_F=-11 rand_margin=+1 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 13

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 99 / 48 / 0 / 45 | -2 [-10, +6] | -13 [-21, -5] | +15 [+8, +23] | 145, +0, 5 / 5 | 15 / 2 / 0 | +0 (136) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 109 / 52 / 2 / 29 | -8 [-16, +0] | +9 [+3, +16] | -11 [-18, -5] | 151, -9, 3 / 12 | 1 / 10 / 1 | -10 (141) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 113 / 74 / 2 / 3 | +2 [+0, +5] | +0 [+0, +0] | +1 [+0, +3] | 187, +2, 2 / 0 | 0 / 0 / 0 | +1 (185) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 114 / 73 / 2 / 3 | +3 [-3, +9] | +0 [+0, +0] | +1 [+0, +3] | 187, +3, 6 / 3 | 0 / 0 / 0 | +3 (177) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 111 / 76 / 3 / 2 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 2 / 2 | 0 / 0 / 0 | +0 (184) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 111 / 75 / 3 / 3 | +0 [-5, +5] | -1 [-3, +0] | +1 [+0, +3] | 186, +1, 3 / 2 | 1 / 0 / 1 | +0 (183) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 126 / 60 / 3 / 3 | +15 [+7, +24] | -1 [-3, +0] | +1 [+0, +3] | 186, +16, 17 / 1 | 1 / 0 / 1 | +15 (176) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 127 / 60 / 2 / 3 | +16 [+8, +24] | +0 [+0, +0] | +1 [+0, +3] | 187, +16, 17 / 1 | 0 / 0 / 0 | +15 (176) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 119 / 68 / 2 / 3 | -16 [-26, -7] | -2 [-5, +0] | +1 [+0, +3] | 187, -14, 3 / 17 | 2 / 0 / 1 | -9 (164) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 135 / 39 / 2 / 16 | +1 [-3, +6] | -1 [-3, +0] | +1 [+0, +3] | 174, +2, 3 / 1 | 1 / 0 / 0 | +2 (163) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 135 / 35 / 1 / 21 | +1 [-7, +8] | -5 [-10, +0] | +6 [+2, +11] | 169, +3, 7 / 4 | 6 / 1 / 0 | +4 (141) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 145 / 19 / 0 / 28 | +11 [+0, +22] | -11 [-20, -3] | +13 [+5, +21] | 160, +14, 18 / 4 | 15 / 4 / 0 | +7 (121) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 98 / 33 / 3 / 58 | 3.4 | -1 [-11, +9] | -32 [-44, -21] | +33 [+23, +43] | 128, +8 [+1, +15], 11 / 3 | 35 / 3 / 2 | 11 / 24 / 2 / 1 | +8 (118) |
| x.rand.p2 | 97 / 63 / 5 / 27 | 3.3 | -2 [-6, +2] | -3 [-7, +0] | +2 [+0, +5] | 160, -1 [-5, +2], 1 / 2 | 3 / 0 / 1 | 1 / 2 / 0 / 0 | -1 (154) |
| x.rand.m2 | 99 / 65 / 4 / 24 | 3.2 | +0 [-5, +5] | +1 [-2, +5] | -1 [-5, +2] | 162, +0 [-5, +5], 3 / 3 | 1 / 2 / 0 | 0 / 1 / 0 / 2 | +0 (159) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 117 / 37 / 9 / 29; dR +6 [-6, +18]; answered-rate change -7.8 pp [-13.0, -3.1]; out / in 20 / 5; mean words 10.1 -> 10.4
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p2 +1 [-10, +11]; vs x.rand.m2 -1 [-12, +9] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -16.7 pp [-22.9, -10.9]; moved out / in 35 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 118 questions; mean words 3.2 -> 3.4
KEY family=E judge=pro layer=13 A=not_met B=not_met dR=-1 dR_F=+6 dAns=-32 dAns_F=-15 rand_margin=-1 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 13

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 91 / 53 / 3 / 45 | -1 [-9, +7] | -13 [-22, -5] | +15 [+8, +23] | 141, +2, 6 / 4 | 16 / 3 / 1 | +3 (133) |
| caut_E2.wq.m1 | 109 / 39 / 3 / 41 | 102 / 57 / 3 / 30 | -7 [-15, +0] | +11 [+5, +18] | -11 [-18, -5] | 147, -9, 2 / 11 | 1 / 12 / 1 | -7 (138) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 102 / 85 / 1 / 4 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +2, 2 / 0 | 0 / 0 / 0 | +1 (185) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 101 / 84 / 3 / 4 | +1 [-5, +8] | -2 [-6, +2] | +0 [-3, +3] | 184, +2, 5 / 3 | 3 / 1 / 2 | +4 (176) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 103 / 85 / 1 / 3 | +3 [-1, +8] | +1 [+0, +3] | -1 [-3, +0] | 187, +2, 3 / 1 | 0 / 1 / 0 | +2 (185) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 100 / 87 / 1 / 4 | +0 [-5, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 3 / 3 | 0 / 0 / 0 | +0 (183) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 114 / 73 / 1 / 4 | +14 [+5, +23] | +0 [-3, +3] | +0 [-3, +3] | 186, +14, 17 / 3 | 1 / 1 / 0 | +13 (177) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 115 / 72 / 1 / 4 | +15 [+7, +24] | +0 [-3, +3] | +0 [-3, +3] | 186, +15, 17 / 2 | 1 / 1 / 0 | +14 (176) |
| conf_E2.wq.m1 | 122 / 64 / 2 / 4 | 105 / 81 / 2 / 4 | -17 [-27, -8] | +0 [-4, +4] | +0 [-3, +3] | 184, -18, 2 / 20 | 2 / 2 / 1 | -9 (162) |
| x_F.w.p0.5 | 111 / 58 / 8 / 15 | 111 / 59 / 6 / 16 | +0 [-5, +6] | +1 [-2, +5] | +1 [+0, +3] | 168, +0, 4 / 4 | 1 / 2 / 0 | +1 (159) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 113 / 52 / 5 / 22 | +2 [-8, +11] | -4 [-11, +3] | +7 [+2, +12] | 161, +3, 11 / 8 | 8 / 4 / 1 | +4 (137) |
| x_F.w.p2 | 111 / 58 / 8 / 15 | 117 / 37 / 9 / 29 | +6 [-6, +18] | -15 [-25, -6] | +14 [+6, +22] | 149, +8, 18 / 10 | 20 / 5 / 4 | +8 (116) |

=== 2026-11-13 11:11:39 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

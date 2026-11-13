TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=0.5 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 111 / 50 / 0 / 31 | 3.1 | +8 [+2, +15] | -5 [-10, -1] | +6 [+2, +11] | 161, +9 [+3, +15], 10 / 1 | 5 / 0 / 0 | 1 / 4 / 0 / 0 | +10 (158) |
| x.rand.p0.5 | 103 / 63 / 2 / 24 | 3.3 | +0 [-3, +3] | +0 [-3, +3] | -1 [-3, +0] | 165, +1 [+0, +3], 1 / 0 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +1 (164) |
| x.rand.m0.5 | 103 / 62 / 1 / 26 | 3.5 | +0 [-3, +3] | -1 [-3, +0] | +1 [+0, +3] | 165, +0 [-3, +3], 1 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +1 (161) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 135 / 40 / 1 / 16; dR +1 [-5, +7]; answered-rate change +0.0 pp [-1.6, +1.6]; out / in 1 / 1; mean words 10.1 -> 10.1
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p0.5 +8 [+1, +15]; vs x.rand.m0.5 +8 [+2, +14] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.6 pp [-5.2, -0.5]; moved out / in 5 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +10 on 158 questions; mean words 3.2 -> 3.1
KEY family=E judge=allenai layer=11 A=met B=not_met dR=+8 dR_F=+1 dAns=-5 dAns_F=+0 rand_margin=+8 EXPLORATORY base=x main=x.w.p0.5 strength=0.5

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

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 102 / 54 / 5 / 31 | 3.1 | +3 [-4, +10] | -7 [-13, -2] | +6 [+2, +11] | 155, +4 [-2, +10], 7 / 3 | 8 / 1 / 2 | 2 / 6 / 1 / 0 | +5 (153) |
| x.rand.p0.5 | 98 / 66 / 4 / 24 | 3.3 | -1 [-3, +0] | +1 [+0, +3] | -1 [-3, +0] | 163, -1 [-3, +0], 0 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (161) |
| x.rand.m0.5 | 98 / 64 / 4 / 26 | 3.5 | -1 [-4, +2] | -1 [-3, +0] | +1 [+0, +3] | 162, -1 [-4, +2], 1 / 2 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +0 (158) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 116 / 54 / 6 / 16; dR +5 [+0, +10]; answered-rate change +0.5 pp [-1.0, +2.6]; out / in 1 / 2; mean words 10.1 -> 10.1
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p0.5 +4 [-3, +11]; vs x.rand.m0.5 +4 [-2, +10] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.6 pp [-6.8, -1.0]; moved out / in 8 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 153 questions; mean words 3.2 -> 3.1
KEY family=E judge=pro layer=11 A=not_met B=not_met dR=+3 dR_F=+5 dAns=-7 dAns_F=+1 rand_margin=+4 EXPLORATORY base=x main=x.w.p0.5 strength=0.5

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

=== 2026-11-13 10:53:29 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

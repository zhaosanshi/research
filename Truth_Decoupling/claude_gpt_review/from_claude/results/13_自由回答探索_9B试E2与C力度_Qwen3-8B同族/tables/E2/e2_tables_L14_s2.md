TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=2 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 106 / 33 / 0 / 53 | 3.1 | +3 [-5, +11] | -27 [-37, -18] | +28 [+19, +38] | 139, +8 [+2, +15], 10 / 2 | 27 / 0 / 0 | 5 / 22 / 0 / 0 | +9 (130) |
| x.rand.p2 | 104 / 59 / 2 / 27 | 3.7 | +1 [-5, +7] | -3 [-8, +1] | +2 [-2, +6] | 162, +2 [-3, +8], 5 / 3 | 4 / 1 / 1 | 1 / 3 / 0 / 1 | +0 (151) |
| x.rand.m2 | 102 / 61 / 3 / 26 | 3.2 | -1 [-8, +6] | -3 [-7, +0] | +1 [+0, +3] | 163, +1 [-5, +7], 5 / 4 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +3 (158) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 140 / 21 / 0 / 31; dR +6 [-5, +17]; answered-rate change -7.3 pp [-12.0, -3.1]; out / in 17 / 3; mean words 10.1 -> 9.9
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p2 +2 [-7, +11]; vs x.rand.m2 +4 [-5, +13] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -14.1 pp [-19.3, -9.4]; moved out / in 27 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +9 on 130 questions; mean words 3.2 -> 3.1
KEY family=E judge=allenai layer=14 A=not_met B=not_met dR=+3 dR_F=+6 dAns=-27 dAns_F=-14 rand_margin=+2 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 14

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 100 / 48 / 0 / 44 | -1 [-9, +6] | -12 [-20, -5] | +14 [+7, +22] | 146, +0, 5 / 5 | 14 / 2 / 0 | -1 (140) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 106 / 56 / 2 / 28 | -11 [-20, -3] | +10 [+4, +17] | -12 [-19, -6] | 151, -12, 2 / 14 | 1 / 11 / 1 | -11 (139) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 112 / 75 / 3 / 2 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 187, +1, 1 / 0 | 0 / 0 / 0 | +0 (185) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 115 / 72 / 3 / 2 | +4 [-1, +10] | +0 [-3, +3] | +0 [+0, +0] | 186, +4, 5 / 1 | 1 / 1 / 1 | +1 (171) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 112 / 74 / 2 / 4 | +1 [-3, +5] | -1 [-3, +0] | +2 [+0, +5] | 186, +2, 3 / 1 | 1 / 0 / 0 | +2 (183) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 111 / 77 / 2 / 2 | +0 [-5, +5] | +1 [+0, +3] | +0 [+0, +0] | 187, -1, 2 / 3 | 0 / 1 / 0 | +0 (177) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 128 / 59 / 3 / 2 | +17 [+9, +26] | +0 [+0, +0] | +0 [+0, +0] | 187, +17, 18 / 1 | 0 / 0 / 0 | +16 (177) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 127 / 60 / 3 / 2 | +16 [+8, +24] | +0 [+0, +0] | +0 [+0, +0] | 187, +16, 17 / 1 | 0 / 0 / 0 | +15 (176) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 119 / 68 / 2 / 3 | -16 [-25, -7] | -2 [-5, +0] | +1 [+0, +3] | 187, -14, 3 / 17 | 2 / 0 / 1 | -10 (164) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 136 / 39 / 1 / 16 | +2 [-3, +8] | +0 [-3, +3] | +1 [+0, +3] | 174, +2, 4 / 2 | 1 / 1 / 0 | +1 (161) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 135 / 34 / 1 / 22 | +1 [-6, +8] | -6 [-12, -1] | +7 [+2, +13] | 168, +3, 6 / 3 | 7 / 1 / 0 | +2 (146) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 140 / 21 / 0 / 31 | +6 [-5, +17] | -14 [-23, -6] | +16 [+8, +24] | 158, +12, 15 / 3 | 17 / 3 / 0 | +10 (122) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 99 / 37 / 3 / 53 | 3.1 | +0 [-10, +10] | -27 [-38, -17] | +28 [+19, +38] | 133, +4 [-4, +12], 10 / 6 | 30 / 3 / 2 | 6 / 24 / 2 / 1 | +4 (125) |
| x.rand.p2 | 97 / 65 / 3 / 27 | 3.7 | -2 [-8, +3] | -1 [-5, +3] | +2 [-2, +6] | 160, -2 [-8, +3], 3 / 5 | 3 / 2 / 0 | 0 / 3 / 0 / 2 | +0 (148) |
| x.rand.m2 | 99 / 65 / 2 / 26 | 3.2 | +0 [-6, +6] | +1 [-2, +4] | +1 [+0, +3] | 162, -1 [-6, +4], 3 / 4 | 1 / 2 / 0 | 0 / 1 / 1 / 1 | +0 (156) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 120 / 35 / 6 / 31; dR +9 [-3, +21]; answered-rate change -7.3 pp [-12.5, -2.6]; out / in 19 / 5; mean words 10.1 -> 9.9
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p2 +2 [-9, +12]; vs x.rand.m2 +0 [-9, +9] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -14.1 pp [-19.8, -8.9]; moved out / in 30 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +4 on 125 questions; mean words 3.2 -> 3.1
KEY family=E judge=pro layer=14 A=not_met B=not_met dR=+0 dR_F=+9 dAns=-27 dAns_F=-14 rand_margin=+0 EXPLORATORY base=x main=x.w.p2 strength=2

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 14

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 92 / 65 / 5 / 30 | 96 / 50 / 2 / 44 | +4 [-3, +11] | -11 [-19, -3] | +14 [+7, +22] | 143, +4, 7 / 3 | 14 / 3 / 0 | +3 (137) |
| caut_E2.wq.m1 | 109 / 39 / 3 / 41 | 100 / 61 / 2 / 29 | -9 [-17, -1] | +13 [+7, +20] | -12 [-19, -6] | 148, -11, 2 / 13 | 0 / 13 / 0 | -8 (136) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 101 / 86 / 1 / 4 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 187, +1, 1 / 0 | 0 / 0 / 0 | +0 (185) |
| conf.md.p1 | 100 / 87 / 1 / 4 | 101 / 84 / 3 / 4 | +1 [-5, +7] | -2 [-6, +2] | +0 [-3, +3] | 184, +2, 4 / 2 | 3 / 1 / 2 | +2 (170) |
| conf.rand.m1 | 100 / 87 / 1 / 4 | 104 / 82 / 1 / 5 | +4 [+1, +8] | -1 [-3, +0] | +1 [+0, +3] | 186, +4, 4 / 0 | 1 / 0 / 0 | +4 (183) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 97 / 89 / 2 / 4 | -3 [-8, +1] | -1 [-5, +2] | +0 [-3, +3] | 185, -3, 0 / 3 | 2 / 1 / 1 | -1 (178) |
| conf.w.p1 | 100 / 87 / 1 / 4 | 118 / 69 / 1 / 4 | +18 [+10, +27] | +0 [-3, +3] | +0 [-3, +3] | 186, +18, 19 / 1 | 1 / 1 / 0 | +16 (177) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 117 / 70 / 1 / 4 | +17 [+9, +26] | +0 [-3, +3] | +0 [-3, +3] | 186, +17, 18 / 1 | 1 / 1 / 0 | +15 (176) |
| conf_E2.wq.m1 | 122 / 64 / 2 / 4 | 105 / 81 / 2 / 4 | -17 [-27, -7] | +0 [-4, +4] | +0 [-3, +3] | 184, -18, 3 / 21 | 2 / 2 / 1 | -9 (162) |
| x_F.w.p0.5 | 111 / 58 / 8 / 15 | 112 / 58 / 6 / 16 | +1 [-6, +8] | +1 [-2, +5] | +1 [+0, +3] | 168, +0, 6 / 6 | 1 / 2 / 0 | +0 (156) |
| x_F.w.p1 | 111 / 58 / 8 / 15 | 111 / 52 / 7 / 22 | +0 [-9, +9] | -6 [-12, +0] | +7 [+2, +13] | 161, +0, 8 / 8 | 8 / 2 / 1 | +2 (143) |
| x_F.w.p2 | 111 / 58 / 8 / 15 | 120 / 35 / 6 / 31 | +9 [-3, +21] | -14 [-24, -5] | +16 [+8, +24] | 150, +11, 17 / 6 | 19 / 5 / 3 | +9 (117) |

=== 2026-11-13 10:49:35 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

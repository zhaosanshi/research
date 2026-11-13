TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 107 / 46 / 0 / 39 | 3.3 | +4 [-3, +11] | -13 [-20, -7] | +14 [+7, +22] | 153, +7 [+2, +13], 8 / 1 | 13 / 0 / 0 | 3 / 10 / 0 / 0 | +7 (150) |
| x.w.m1 | 96 / 79 / 2 / 15 | 3.3 | -7 [-15, +1] | +9 [+2, +16] | -10 [-17, -4] | 164, -7 [-15, +0], 4 / 11 | 2 / 11 / 1 | 1 / 1 / 1 / 10 | -6 (150) |
| x.md.p1 | 105 / 55 / 1 / 31 | 3.4 | +2 [-4, +8] | -6 [-12, -1] | +6 [+1, +12] | 159, +4 [+0, +9], 5 / 1 | 7 / 1 / 0 | 3 / 4 / 1 / 0 | +3 (146) |
| x.rand.p1 | 102 / 63 / 2 / 25 | 3.3 | -1 [-3, +0] | -1 [-3, +0] | +0 [+0, +0] | 165, +0 [+0, +0], 0 / 0 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (161) |
| x.rand.m1 | 101 / 64 / 1 / 26 | 3.3 | -2 [-6, +2] | -1 [-5, +2] | +1 [-2, +5] | 164, -1 [-4, +2], 1 / 2 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +0 (162) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 135 / 35 / 1 / 21; dR +1 [-7, +8]; answered-rate change -2.6 pp [-5.2, +0.0]; out / in 6 / 1; mean words 10.1 -> 10.1
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-2, +12]; vs x.rand.m1 +6 [+0, +13] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -6.8 pp [-10.4, -3.6]; moved out / in 13 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +7 on 150 questions; mean words 3.2 -> 3.3
KEY family=E judge=allenai layer=13 A=not_met B=not_met dR=+4 dR_F=+1 dAns=-13 dAns_F=-5 rand_margin=+5

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

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 126 / 60 / 3 / 3 | 2.8 | +15 [+7, +24] | -1 [-3, +0] | +1 [+0, +3] | 186, +16 [+8, +24], 17 / 1 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +15 (176) |
| conf.md.p1 | 114 / 73 / 2 / 3 | 3.1 | +3 [-3, +9] | +0 [+0, +0] | +1 [+0, +3] | 187, +3 [-3, +9], 6 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +3 (177) |
| conf.rand.p1 | 111 / 75 / 3 / 3 | 3.1 | +0 [-5, +5] | -1 [-3, +0] | +1 [+0, +3] | 186, +1 [-3, +6], 3 / 2 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (183) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 3.1 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 187, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +15 (7.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +15 [+7, +24]; vs conf.rand.m1 +15 [+7, +23] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +15 on 176 questions; mean words 3.1 -> 2.8
KEY family=E judge=allenai layer=13 A=incomplete B=incomplete dR=+15 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+15 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 99 / 52 / 2 / 39 | 3.3 | +0 [-8, +8] | -12 [-20, -5] | +14 [+7, +22] | 149, +2 [-5, +9], 7 / 5 | 14 / 2 / 0 | 3 / 11 / 1 / 1 | +2 (146) |
| x.w.m1 | 91 / 84 / 2 / 15 | 3.3 | -8 [-16, +0] | +12 [+5, +19] | -10 [-17, -4] | 162, -10 [-18, -3], 2 / 12 | 1 / 13 / 0 | 0 / 1 / 2 / 11 | -7 (147) |
| x.md.p1 | 96 / 61 / 3 / 32 | 3.4 | -3 [-10, +3] | -6 [-12, -1] | +7 [+2, +13] | 156, +0 [-5, +6], 4 / 4 | 7 / 1 / 0 | 3 / 4 / 0 / 1 | +1 (143) |
| x.rand.p1 | 95 / 67 / 5 / 25 | 3.3 | -4 [-8, -1] | -1 [-3, +0] | +0 [+0, +0] | 162, -3 [-7, +0], 0 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -2 (158) |
| x.rand.m1 | 97 / 65 / 4 / 26 | 3.3 | -2 [-6, +2] | -1 [-5, +2] | +1 [-2, +5] | 161, -1 [-4, +2], 1 / 2 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +0 (159) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 113 / 52 / 5 / 22; dR +2 [-8, +11]; answered-rate change -2.1 pp [-5.7, +1.6]; out / in 8 / 4; mean words 10.1 -> 10.1
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +4 [-3, +11]; vs x.rand.m1 +2 [-6, +10] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -6.2 pp [-10.4, -2.6]; moved out / in 14 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +2 on 146 questions; mean words 3.2 -> 3.3
KEY family=E judge=pro layer=13 A=not_met B=not_met dR=+0 dR_F=+2 dAns=-12 dAns_F=-4 rand_margin=+2

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

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 114 / 73 / 1 / 4 | 2.8 | +14 [+5, +23] | +0 [-3, +3] | +0 [-3, +3] | 186, +14 [+6, +23], 17 / 3 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | +13 (177) |
| conf.md.p1 | 101 / 84 / 3 / 4 | 3.1 | +1 [-5, +8] | -2 [-6, +2] | +0 [-3, +3] | 184, +2 [-3, +8], 5 / 3 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +4 (176) |
| conf.rand.p1 | 100 / 87 / 1 / 4 | 3.1 | +0 [-5, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +0 [-5, +5], 3 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (183) |
| conf.rand.m1 | 103 / 85 / 1 / 3 | 3.1 | +3 [-1, +8] | +1 [+0, +3] | -1 [-3, +0] | 187, +2 [-2, +6], 3 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (185) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +14 (7.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +14 [+6, +23]; vs conf.rand.m1 +11 [+2, +20] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.6, +1.6]; moved out / in 1 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +13 on 177 questions; mean words 3.1 -> 2.8
KEY family=E judge=pro layer=13 A=incomplete B=incomplete dR=+14 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+11 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 11:11:21 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 110 / 43 / 0 / 39 | 3.1 | +7 [+0, +15] | -13 [-20, -7] | +14 [+8, +21] | 153, +9 [+2, +16], 11 / 2 | 13 / 0 / 0 | 2 / 11 / 0 / 0 | +9 (147) |
| x.w.m1 | 96 / 80 / 2 / 14 | 3.4 | -7 [-15, +0] | +10 [+4, +17] | -11 [-18, -5] | 165, -6 [-14, +1], 4 / 10 | 1 / 11 / 1 | 1 / 0 / 0 / 11 | -5 (146) |
| x.md.p1 | 99 / 60 / 2 / 31 | 3.4 | -4 [-10, +1] | -7 [-13, -2] | +6 [+2, +11] | 159, +0 [-4, +4], 2 / 2 | 7 / 0 / 1 | 4 / 3 / 0 / 0 | +0 (150) |
| x.rand.p1 | 103 / 62 / 1 / 26 | 3.4 | +0 [-4, +4] | -1 [-4, +2] | +1 [-2, +4] | 164, +0 [-4, +4], 2 / 2 | 2 / 1 / 0 | 0 / 2 / 0 / 1 | +0 (158) |
| x.rand.m1 | 106 / 60 / 1 / 25 | 3.1 | +3 [+0, +7] | +0 [+0, +0] | +0 [+0, +0] | 166, +3 [+0, +7], 3 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +3 (166) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 135 / 34 / 1 / 22; dR +1 [-6, +8]; answered-rate change -3.1 pp [-6.2, -0.5]; out / in 7 / 1; mean words 10.1 -> 9.9
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +7 [-1, +15]; vs x.rand.m1 +4 [-4, +12] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -6.8 pp [-10.4, -3.6]; moved out / in 13 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +9 on 147 questions; mean words 3.2 -> 3.1
KEY family=E judge=allenai layer=14 A=not_met B=not_met dR=+7 dR_F=+1 dAns=-13 dAns_F=-6 rand_margin=+4

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

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 128 / 59 / 3 / 2 | 2.8 | +17 [+9, +26] | +0 [+0, +0] | +0 [+0, +0] | 187, +17 [+9, +26], 18 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +16 (177) |
| conf.md.p1 | 115 / 72 / 3 / 2 | 3.4 | +4 [-1, +10] | +0 [-3, +3] | +0 [+0, +0] | 186, +4 [+0, +9], 5 / 1 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +1 (171) |
| conf.rand.p1 | 111 / 77 / 2 / 2 | 3.3 | +0 [-5, +5] | +1 [+0, +3] | +0 [+0, +0] | 187, -1 [-6, +3], 2 / 3 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (177) |
| conf.rand.m1 | 112 / 74 / 2 / 4 | 2.9 | +1 [-3, +5] | -1 [-3, +0] | +2 [+0, +5] | 186, +2 [-2, +6], 3 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +2 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +17 [+8, +27]; vs conf.rand.m1 +16 [+8, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.5, +1.5]; moved out / in 0 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +16 on 177 questions; mean words 3.1 -> 2.8
KEY family=E judge=allenai layer=14 A=incomplete B=incomplete dR=+17 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+16 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 101 / 49 / 3 / 39 | 3.1 | +2 [-6, +10] | -13 [-21, -6] | +14 [+8, +21] | 148, +4 [-3, +11], 9 / 5 | 15 / 2 / 1 | 3 / 12 / 1 / 1 | +3 (143) |
| x.w.m1 | 90 / 86 / 2 / 14 | 3.4 | -9 [-17, -1] | +13 [+7, +20] | -11 [-18, -5] | 163, -11 [-19, -4], 2 / 13 | 0 / 13 / 0 | 0 / 0 / 2 / 11 | -6 (143) |
| x.md.p1 | 96 / 61 / 3 / 32 | 3.4 | -3 [-9, +3] | -6 [-11, -2] | +7 [+2, +13] | 157, +0 [-5, +5], 3 / 3 | 6 / 0 / 0 | 3 / 3 / 0 / 0 | +0 (147) |
| x.rand.p1 | 96 / 66 / 4 / 26 | 3.4 | -3 [-8, +1] | -1 [-4, +2] | +1 [-2, +4] | 161, -3 [-8, +1], 1 / 4 | 2 / 1 / 0 | 0 / 2 / 0 / 1 | +0 (155) |
| x.rand.m1 | 100 / 64 / 3 / 25 | 3.1 | +1 [-3, +5] | +1 [+0, +3] | +0 [+0, +0] | 163, +0 [-4, +4], 2 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (163) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 111 / 52 / 7 / 22; dR +0 [-9, +9]; answered-rate change -3.1 pp [-6.2, +0.0]; out / in 8 / 2; mean words 10.1 -> 9.9
- main effect: dR +2 (1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-4, +14]; vs x.rand.m1 +1 [-7, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -6.8 pp [-10.9, -3.1]; moved out / in 15 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +3 on 143 questions; mean words 3.2 -> 3.1
KEY family=E judge=pro layer=14 A=not_met B=not_met dR=+2 dR_F=+0 dAns=-13 dAns_F=-6 rand_margin=+1

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

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 118 / 69 / 1 / 4 | 2.8 | +18 [+10, +27] | +0 [-3, +3] | +0 [-3, +3] | 186, +18 [+10, +27], 19 / 1 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | +16 (177) |
| conf.md.p1 | 101 / 84 / 3 / 4 | 3.4 | +1 [-5, +7] | -2 [-6, +2] | +0 [-3, +3] | 184, +2 [-2, +7], 4 / 2 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +2 (170) |
| conf.rand.p1 | 97 / 89 / 2 / 4 | 3.3 | -3 [-8, +1] | -1 [-5, +2] | +0 [-3, +3] | 185, -3 [-7, +0], 0 / 3 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -1 (178) |
| conf.rand.m1 | 104 / 82 / 1 / 5 | 2.9 | +4 [+1, +8] | -1 [-3, +0] | +1 [+0, +3] | 186, +4 [+1, +8], 4 / 0 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +4 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +18 (9.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +21 [+13, +30]; vs conf.rand.m1 +14 [+6, +22] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.6, +1.6]; moved out / in 1 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +16 on 177 questions; mean words 3.1 -> 2.8
KEY family=E judge=pro layer=14 A=incomplete B=incomplete dR=+18 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+14 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 10:49:20 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

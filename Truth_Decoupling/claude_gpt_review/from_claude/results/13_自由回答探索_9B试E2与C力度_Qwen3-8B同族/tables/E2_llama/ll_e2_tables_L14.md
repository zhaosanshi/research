TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 73 / 58 / 1 / 60 | 4.1 | -5 [-16, +6] | -35 [-46, -24] | +37 [+26, +48] | 129, +7 [-1, +16], 13 / 6 | 37 / 2 / 0 | 13 / 24 / 1 / 1 | +2 (112) |
| x.w.m1 | 77 / 102 / 1 / 12 | 4.0 | -1 [-12, +10] | +13 [+6, +21] | -11 [-18, -5] | 165, -7 [-17, +2], 8 / 15 | 1 / 14 / 1 | 1 / 0 / 7 / 7 | -7 (152) |
| x.md.p1 | 83 / 62 / 2 / 45 | 4.6 | +5 [-6, +16] | -21 [-30, -12] | +22 [+14, +31] | 144, +13 [+4, +22], 18 / 5 | 22 / 1 / 0 | 9 / 13 / 1 / 0 | +6 (122) |
| x.rand.p1 | 78 / 86 / 1 / 27 | 3.8 | +0 [-6, +6] | -2 [-7, +3] | +4 [+1, +8] | 162, -1 [-6, +4], 3 / 4 | 4 / 2 / 0 | 1 / 3 / 2 / 0 | -1 (153) |
| x.rand.m1 | 77 / 89 / 3 / 23 | 4.5 | -1 [-7, +5] | +0 [-3, +3] | +0 [-3, +3] | 165, -1 [-7, +5], 4 / 5 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | -4 (151) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 91 / 44 / 1 / 56; dR -11 [-23, +0]; answered-rate change -13.5 pp [-18.8, -8.9]; out / in 27 / 1; mean words 9.3 -> 9.3
- main effect: dR -5 (-2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -5 [-17, +6]; vs x.rand.m1 -4 [-15, +7] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -18.2 pp [-24.0, -12.5]; moved out / in 37 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +2 on 112 questions; mean words 4.0 -> 4.1
KEY family=E judge=allenai layer=14 A=not_met B=not_met dR=-5 dR_F=-11 dAns=-35 dAns_F=-26 rand_margin=-5

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 14

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 71 / 53 / 1 / 67 | -11 [-22, +1] | -40 [-52, -29] | +40 [+29, +51] | 123, +6, 11 / 5 | 41 / 1 / 1 | +2 (107) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 95 / 81 / 0 / 16 | -7 [-19, +5] | +21 [+12, +30] | -21 [-30, -12] | 154, -15, 6 / 21 | 1 / 22 / 0 | -15 (145) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 87 / 102 / 1 / 2 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 189, -1, 0 / 1 | 0 / 0 / 0 | -1 (187) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 98 / 89 / 4 / 1 | +10 [+0, +20] | -2 [-6, +2] | -1 [-3, +0] | 186, +13, 19 / 6 | 3 / 1 / 3 | +7 (166) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 86 / 104 / 1 / 1 | -2 [-8, +4] | +1 [-2, +5] | -1 [-3, +0] | 188, -2, 3 / 5 | 1 / 2 / 1 | -1 (180) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 86 / 103 / 1 / 2 | -2 [-8, +4] | +0 [+0, +0] | +0 [+0, +0] | 189, -2, 4 / 6 | 0 / 0 / 0 | -2 (181) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 101 / 85 / 2 / 4 | +13 [+3, +24] | -3 [-8, +1] | +2 [+0, +5] | 185, +15, 20 / 5 | 4 / 1 / 2 | +10 (166) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 113 / 76 / 1 / 2 | +25 [+14, +36] | +0 [-5, +5] | +0 [-4, +4] | 186, +24, 27 / 3 | 3 / 3 / 1 | +14 (164) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 93 / 98 / 1 / 0 | -24 [-35, -13] | +0 [-3, +3] | +0 [+0, +0] | 190, -24, 4 / 28 | 1 / 1 / 1 | -22 (176) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 91 / 44 / 1 / 56 | -11 [-23, +0] | -26 [-36, -17] | +26 [+17, +36] | 134, +2, 11 / 9 | 27 / 1 / 1 | +0 (90) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 101 / 85 / 2 / 4 | 3.6 | +13 [+3, +24] | -3 [-8, +1] | +2 [+0, +5] | 185, +15 [+6, +25], 20 / 5 | 4 / 1 / 2 | 3 / 1 / 1 / 0 | +10 (166) |
| conf.md.p1 | 98 / 89 / 4 / 1 | 3.9 | +10 [+0, +20] | -2 [-6, +2] | -1 [-3, +0] | 186, +13 [+4, +23], 19 / 6 | 3 / 1 / 3 | 3 / 0 / 0 / 1 | +7 (166) |
| conf.rand.p1 | 86 / 103 / 1 / 2 | 3.4 | -2 [-8, +4] | +0 [+0, +0] | +0 [+0, +0] | 189, -2 [-8, +4], 4 / 6 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -2 (181) |
| conf.rand.m1 | 86 / 104 / 1 / 1 | 3.6 | -2 [-8, +4] | +1 [-2, +5] | -1 [-3, +0] | 188, -2 [-8, +3], 3 / 5 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | -1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +13 (6.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +15 [+5, +25]; vs conf.rand.m1 +15 [+5, +26] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.2, +0.5]; moved out / in 4 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +10 on 166 questions; mean words 3.4 -> 3.6
KEY family=E judge=allenai layer=14 A=incomplete B=not_met dR=+13 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+15 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 59 / 66 / 5 / 62 | 4.1 | -5 [-15, +5] | -37 [-49, -26] | +38 [+27, +49] | 123, +5 [-2, +12], 9 / 4 | 39 / 2 / 3 | 12 / 27 / 2 / 0 | +4 (109) |
| x.w.m1 | 69 / 106 / 4 / 13 | 4.0 | +5 [-4, +14] | +13 [+6, +21] | -11 [-18, -5] | 161, +0 [-8, +8], 8 / 8 | 1 / 14 / 1 | 1 / 0 / 6 / 8 | -1 (149) |
| x.md.p1 | 62 / 80 / 4 / 46 | 4.6 | -2 [-11, +7] | -20 [-29, -12] | +22 [+14, +31] | 141, +3 [-5, +11], 9 / 6 | 21 / 1 / 1 | 6 / 15 / 1 / 0 | +3 (121) |
| x.rand.p1 | 60 / 99 / 6 / 27 | 3.8 | -4 [-10, +1] | -3 [-7, +0] | +3 [+0, +7] | 159, -4 [-10, +1], 2 / 6 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | -1 (150) |
| x.rand.m1 | 66 / 96 / 6 / 24 | 4.5 | +2 [-4, +8] | +0 [-4, +4] | +0 [-3, +3] | 160, +1 [-4, +6], 4 / 3 | 2 / 2 / 1 | 1 / 1 / 2 / 0 | +0 (146) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 77 / 57 / 1 / 57; dR -3 [-15, +9]; answered-rate change -13.5 pp [-18.8, -8.3]; out / in 28 / 2; mean words 9.3 -> 9.3
- main effect: dR -5 (-2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -1 [-11, +9]; vs x.rand.m1 -7 [-17, +3] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -19.3 pp [-25.5, -13.5]; moved out / in 39 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +4 on 109 questions; mean words 4.0 -> 4.1
KEY family=E judge=pro layer=14 A=not_met B=not_met dR=-5 dR_F=-3 dAns=-37 dAns_F=-26 rand_margin=-7

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 14

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 58 / 63 / 3 / 68 | -6 [-17, +5] | -37 [-49, -25] | +40 [+29, +52] | 118, +2, 9 / 7 | 40 / 3 / 3 | +1 (105) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 75 / 91 / 8 / 18 | -6 [-17, +5] | +16 [+7, +25] | -22 [-31, -14] | 146, -8, 8 / 16 | 4 / 20 / 4 | -7 (137) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 69 / 113 / 8 / 2 | -1 [-5, +2] | +0 [-3, +3] | +1 [+0, +3] | 181, -2, 0 / 2 | 1 / 1 / 0 | -2 (179) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 82 / 96 / 11 / 3 | +12 [+2, +22] | -4 [-10, +1] | +2 [-2, +6] | 176, +13, 18 / 5 | 6 / 2 / 5 | +7 (160) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 71 / 111 / 10 / 0 | +1 [-5, +7] | +0 [-3, +3] | -1 [-3, +0] | 181, +1, 4 / 3 | 1 / 1 / 1 | +1 (173) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 68 / 113 / 9 / 2 | -2 [-7, +3] | -1 [-5, +2] | +1 [+0, +3] | 180, -3, 1 / 4 | 2 / 1 / 1 | -3 (173) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 81 / 97 / 9 / 5 | +11 [+1, +21] | -4 [-9, +1] | +4 [+1, +8] | 177, +13, 19 / 6 | 5 / 1 / 3 | +5 (160) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 91 / 88 / 10 / 3 | +21 [+10, +32] | -3 [-9, +3] | +2 [-2, +6] | 176, +21, 26 / 5 | 6 / 3 / 4 | +9 (156) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 79 / 105 / 8 / 0 | -19 [-32, -6] | +0 [-6, +6] | -4 [-8, -1] | 179, -16, 11 / 27 | 5 / 5 / 5 | -16 (165) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 77 / 57 / 1 / 57 | -3 [-15, +9] | -26 [-36, -16] | +27 [+18, +37] | 132, +5, 14 / 9 | 28 / 2 / 0 | +6 (88) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 81 / 97 / 9 / 5 | 3.6 | +11 [+1, +21] | -4 [-9, +1] | +4 [+1, +8] | 177, +13 [+4, +23], 19 / 6 | 5 / 1 / 3 | 3 / 2 / 1 / 0 | +5 (160) |
| conf.md.p1 | 82 / 96 / 11 / 3 | 3.9 | +12 [+2, +22] | -4 [-10, +1] | +2 [-2, +6] | 176, +13 [+4, +22], 18 / 5 | 6 / 2 / 5 | 2 / 4 / 1 / 1 | +7 (160) |
| conf.rand.p1 | 68 / 113 / 9 / 2 | 3.4 | -2 [-7, +3] | -1 [-5, +2] | +1 [+0, +3] | 180, -3 [-8, +1], 1 / 4 | 2 / 1 / 1 | 0 / 2 / 1 / 0 | -3 (173) |
| conf.rand.m1 | 71 / 111 / 10 / 0 | 3.6 | +1 [-5, +7] | +0 [-3, +3] | -1 [-3, +0] | 181, +1 [-4, +6], 4 / 3 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +1 (173) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +11 (5.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +13 [+3, +23]; vs conf.rand.m1 +10 [+0, +21] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.1 pp [-4.7, +0.5]; moved out / in 5 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 160 questions; mean words 3.4 -> 3.6
KEY family=E judge=pro layer=14 A=incomplete B=not_met dR=+11 dR_F=NA dAns=-4 dAns_F=NA rand_margin=+10 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 13:28:30 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

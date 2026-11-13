TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 87 / 59 / 0 / 46 | 4.4 | +9 [-2, +20] | -20 [-30, -10] | +23 [+14, +32] | 142, +13 [+4, +23], 19 / 6 | 24 / 4 / 0 | 7 / 17 / 3 / 1 | +8 (116) |
| x.w.m1 | 73 / 104 / 1 / 14 | 3.7 | -5 [-16, +5] | +11 [+4, +19] | -9 [-16, -3] | 164, -9 [-18, +0], 6 / 15 | 2 / 13 / 1 | 2 / 0 / 6 / 7 | -9 (158) |
| x.md.p1 | 78 / 89 / 1 / 24 | 3.9 | +0 [-5, +5] | +1 [-2, +4] | +1 [+0, +3] | 165, -1 [-6, +4], 3 / 4 | 1 / 2 / 0 | 0 / 1 / 1 / 1 | +1 (157) |
| x.rand.p1 | 79 / 90 / 1 / 22 | 4.2 | +1 [-5, +7] | +3 [+0, +7] | -1 [-3, +0] | 166, -1 [-6, +4], 3 / 4 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | -1 (158) |
| x.rand.m1 | 77 / 85 / 2 / 28 | 3.8 | -1 [-8, +6] | -4 [-10, +1] | +5 [+1, +10] | 160, +2 [-3, +8], 5 / 3 | 6 / 2 / 1 | 4 / 2 / 1 / 1 | +2 (155) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 99 / 33 / 0 / 60; dR -3 [-16, +10]; answered-rate change -15.1 pp [-20.8, -9.9]; out / in 31 / 2; mean words 9.3 -> 8.8
- main effect: dR +9 (4.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +8 [-4, +20]; vs x.rand.m1 +10 [-2, +22] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -10.4 pp [-15.6, -5.2]; moved out / in 24 / 4 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 116 questions; mean words 4.0 -> 4.4
KEY family=E judge=allenai layer=10 A=not_met B=not_met dR=+9 dR_F=-3 dAns=-20 dAns_F=-29 rand_margin=+8

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 10

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 86 / 56 / 1 / 49 | +4 [-7, +15] | -22 [-32, -13] | +22 [+14, +31] | 141, +9, 17 / 8 | 23 / 1 / 1 | +6 (117) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 81 / 89 / 1 / 21 | -21 [-33, -9] | +15 [+7, +23] | -16 [-24, -9] | 154, -24, 5 / 29 | 1 / 16 / 1 | -22 (143) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 86 / 103 / 1 / 2 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 189, -2, 0 / 2 | 0 / 0 / 0 | -2 (188) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 88 / 102 / 0 / 2 | +0 [-7, +7] | +1 [+0, +3] | +0 [+0, +0] | 189, +0, 6 / 6 | 0 / 1 / 0 | +2 (180) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 90 / 98 / 2 / 2 | +2 [-3, +8] | -1 [-3, +0] | +0 [+0, +0] | 188, +3, 5 / 2 | 1 / 0 / 1 | +2 (185) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 84 / 106 / 0 / 2 | -4 [-10, +1] | +1 [+0, +3] | +0 [+0, +0] | 189, -4, 2 / 6 | 0 / 1 / 0 | -2 (181) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 118 / 73 / 0 / 1 | +30 [+18, +42] | +2 [+0, +5] | -1 [-3, +0] | 189, +28, 34 / 6 | 0 / 2 / 0 | +15 (155) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 113 / 78 / 0 / 1 | +25 [+13, +36] | +2 [+0, +5] | -1 [-3, +0] | 189, +23, 30 / 7 | 0 / 2 / 0 | +14 (157) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 89 / 100 / 1 / 2 | -28 [-40, -16] | -2 [-6, +2] | +2 [+0, +5] | 188, -26, 7 / 33 | 3 / 1 / 1 | -23 (177) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 99 / 33 / 0 / 60 | -3 [-16, +10] | -29 [-40, -19] | +30 [+20, +41] | 130, +12, 18 / 6 | 31 / 2 / 0 | +2 (89) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 118 / 73 / 0 / 1 | 4.1 | +30 [+18, +42] | +2 [+0, +5] | -1 [-3, +0] | 189, +28 [+16, +40], 34 / 6 | 0 / 2 / 0 | 0 / 0 / 2 / 0 | +15 (155) |
| conf.md.p1 | 88 / 102 / 0 / 2 | 3.5 | +0 [-7, +7] | +1 [+0, +3] | +0 [+0, +0] | 189, +0 [-7, +7], 6 / 6 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +2 (180) |
| conf.rand.p1 | 84 / 106 / 0 / 2 | 3.5 | -4 [-10, +1] | +1 [+0, +3] | +0 [+0, +0] | 189, -4 [-10, +1], 2 / 6 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -2 (181) |
| conf.rand.m1 | 90 / 98 / 2 / 2 | 3.3 | +2 [-3, +8] | -1 [-3, +0] | +0 [+0, +0] | 188, +3 [-2, +8], 5 / 2 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +2 (185) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +30 (15.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +34 [+22, +46]; vs conf.rand.m1 +28 [+15, +41] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +1.0 pp [+0.0, +2.6]; moved out / in 0 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +15 on 155 questions; mean words 3.4 -> 4.1
KEY family=E judge=allenai layer=10 A=incomplete B=incomplete dR=+30 dR_F=NA dAns=+2 dAns_F=NA rand_margin=+28 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 67 / 76 / 3 / 46 | 4.4 | +3 [-8, +14] | -19 [-30, -9] | +22 [+13, +31] | 137, +5 [-4, +14], 13 / 8 | 25 / 6 / 2 | 7 / 18 / 5 / 1 | +5 (115) |
| x.w.m1 | 61 / 110 / 7 / 14 | 3.7 | -3 [-12, +6] | +9 [+2, +17] | -10 [-17, -4] | 159, -5 [-12, +2], 4 / 9 | 3 / 12 / 2 | 2 / 1 / 4 / 8 | -5 (154) |
| x.md.p1 | 66 / 97 / 4 / 25 | 3.9 | +2 [-3, +7] | +1 [-2, +5] | +1 [+0, +3] | 161, +0 [-5, +5], 3 / 3 | 1 / 2 / 0 | 0 / 1 / 2 / 0 | +2 (154) |
| x.rand.p1 | 64 / 99 / 6 / 23 | 4.2 | +0 [-6, +6] | +1 [-2, +4] | -1 [-3, +0] | 161, +0 [-5, +5], 3 / 3 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +2 (154) |
| x.rand.m1 | 63 / 93 / 8 / 28 | 3.8 | -1 [-7, +5] | -6 [-11, -2] | +4 [+1, +8] | 156, +1 [-4, +6], 4 / 3 | 6 / 0 / 2 | 2 / 4 / 0 / 0 | +2 (152) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 76 / 51 / 5 / 60; dR -4 [-17, +9]; answered-rate change -17.2 pp [-22.9, -12.0]; out / in 35 / 2; mean words 9.3 -> 8.8
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +3 [-7, +14]; vs x.rand.m1 +4 [-8, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -9.9 pp [-15.6, -4.7]; moved out / in 25 / 6 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 115 questions; mean words 4.0 -> 4.4
KEY family=E judge=pro layer=10 A=not_met B=not_met dR=+3 dR_F=-4 dAns=-19 dAns_F=-33 rand_margin=+3

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 10

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 69 / 65 / 9 / 49 | +5 [-7, +17] | -24 [-35, -14] | +21 [+13, +30] | 130, +8, 16 / 8 | 28 / 4 / 7 | +6 (112) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 68 / 95 / 6 / 23 | -13 [-24, -2] | +13 [+5, +22] | -17 [-25, -10] | 147, -13, 7 / 20 | 3 / 16 / 3 | -12 (136) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 69 / 113 / 8 / 2 | -1 [-4, +2] | +0 [-3, +3] | +1 [+0, +3] | 181, -2, 0 / 2 | 1 / 1 / 0 | -2 (180) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 69 / 113 / 8 / 2 | -1 [-7, +5] | +0 [-3, +3] | +1 [+0, +3] | 181, -2, 3 / 5 | 1 / 1 / 0 | -2 (173) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 67 / 113 / 10 / 2 | -3 [-9, +3] | -2 [-5, +0] | +1 [+0, +3] | 180, -2, 3 / 5 | 2 / 0 / 1 | -2 (178) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 71 / 112 / 8 / 1 | +1 [-5, +7] | +1 [+0, +3] | +0 [+0, +0] | 182, +0, 4 / 4 | 0 / 1 / 0 | +0 (174) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 95 / 86 / 7 / 4 | +25 [+13, +37] | -1 [-7, +5] | +3 [-1, +8] | 177, +23, 30 / 7 | 5 / 4 / 3 | +14 (148) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 95 / 86 / 7 / 4 | +25 [+13, +37] | -1 [-7, +5] | +3 [-1, +8] | 177, +23, 29 / 6 | 5 / 4 / 3 | +16 (151) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 80 / 102 / 8 / 2 | -18 [-31, -5] | -2 [-8, +4] | -2 [-6, +2] | 178, -15, 14 / 29 | 6 / 4 / 5 | -14 (168) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 76 / 51 / 5 / 60 | -4 [-17, +9] | -33 [-44, -23] | +30 [+20, +41] | 125, +9, 18 / 9 | 35 / 2 / 3 | +4 (87) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 95 / 86 / 7 / 4 | 4.1 | +25 [+13, +37] | -1 [-7, +5] | +3 [-1, +8] | 177, +23 [+12, +34], 30 / 7 | 5 / 4 / 3 | 2 / 3 / 4 / 0 | +14 (148) |
| conf.md.p1 | 69 / 113 / 8 / 2 | 3.5 | -1 [-7, +5] | +0 [-3, +3] | +1 [+0, +3] | 181, -2 [-8, +3], 3 / 5 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | -2 (173) |
| conf.rand.p1 | 71 / 112 / 8 / 1 | 3.5 | +1 [-5, +7] | +1 [+0, +3] | +0 [+0, +0] | 182, +0 [-5, +6], 4 / 4 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (174) |
| conf.rand.m1 | 67 / 113 / 10 / 2 | 3.3 | -3 [-9, +3] | -2 [-5, +0] | +1 [+0, +3] | 180, -2 [-8, +3], 3 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -2 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +25 (13.0% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +24 [+11, +37]; vs conf.rand.m1 +28 [+16, +40] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-3.6, +2.6]; moved out / in 5 / 4 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +14 on 148 questions; mean words 3.4 -> 4.1
KEY family=E judge=pro layer=10 A=incomplete B=not_met dR=+25 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+24 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 13:01:28 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

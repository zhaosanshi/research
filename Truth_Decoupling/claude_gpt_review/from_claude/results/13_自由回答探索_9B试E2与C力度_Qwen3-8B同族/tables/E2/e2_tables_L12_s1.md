TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 108 / 47 / 0 / 37 | 3.1 | +5 [-2, +13] | -11 [-18, -5] | +12 [+6, +19] | 155, +8 [+2, +15], 10 / 2 | 11 / 0 / 0 | 3 / 8 / 0 / 0 | +8 (149) |
| x.w.m1 | 98 / 76 / 2 / 16 | 3.5 | -5 [-13, +3] | +8 [+1, +15] | -9 [-16, -3] | 164, -5 [-13, +2], 5 / 10 | 2 / 10 / 1 | 1 / 1 / 1 / 9 | -6 (145) |
| x.md.p1 | 101 / 62 / 2 / 27 | 3.6 | -2 [-8, +4] | -3 [-8, +2] | +2 [-3, +7] | 161, +0 [-5, +5], 3 / 3 | 5 / 2 / 1 | 3 / 2 / 1 / 1 | +1 (149) |
| x.rand.p1 | 105 / 59 / 1 / 27 | 3.3 | +2 [+0, +5] | -2 [-5, +0] | +2 [+0, +5] | 164, +2 [+0, +5], 2 / 0 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | +2 (160) |
| x.rand.m1 | 99 / 65 / 3 / 25 | 3.3 | -4 [-9, +0] | -2 [-6, +2] | +0 [-3, +3] | 163, -1 [-5, +2], 1 / 2 | 3 / 1 / 2 | 3 / 0 / 0 / 1 | +1 (161) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 137 / 37 / 2 / 16; dR +3 [-4, +11]; answered-rate change -0.5 pp [-3.1, +1.6]; out / in 3 / 2; mean words 10.1 -> 10.1
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +3 [-4, +10]; vs x.rand.m1 +9 [+1, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -5.7 pp [-9.4, -2.6]; moved out / in 11 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 149 questions; mean words 3.2 -> 3.1
KEY family=E judge=allenai layer=12 A=not_met B=not_met dR=+5 dR_F=+3 dAns=-11 dAns_F=-1 rand_margin=+3

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

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 126 / 60 / 3 / 3 | 2.8 | +15 [+8, +23] | -1 [-3, +0] | +1 [+0, +3] | 186, +16 [+9, +24], 16 / 0 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +14 (178) |
| conf.md.p1 | 112 / 74 / 3 / 3 | 3.2 | +1 [-4, +6] | -1 [-3, +0] | +1 [+0, +3] | 186, +2 [-3, +7], 4 / 2 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +2 (178) |
| conf.rand.p1 | 113 / 75 / 2 / 2 | 3.1 | +2 [-3, +7] | +1 [+0, +3] | +0 [+0, +0] | 187, +1 [-3, +6], 3 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (183) |
| conf.rand.m1 | 112 / 74 / 2 / 4 | 3.1 | +1 [-4, +6] | -1 [-3, +0] | +2 [+0, +5] | 186, +2 [-3, +7], 4 / 2 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +2 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +15 (7.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +13 [+5, +22]; vs conf.rand.m1 +14 [+6, +23] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +14 on 178 questions; mean words 3.1 -> 2.8
KEY family=E judge=allenai layer=12 A=incomplete B=incomplete dR=+15 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+13 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 99 / 53 / 3 / 37 | 3.1 | +0 [-9, +9] | -11 [-19, -4] | +12 [+6, +19] | 150, +3 [-4, +11], 9 / 6 | 13 / 2 / 1 | 4 / 9 / 1 / 1 | +2 (145) |
| x.w.m1 | 87 / 85 / 4 / 16 | 3.5 | -12 [-20, -5] | +9 [+2, +16] | -9 [-16, -3] | 161, -13 [-21, -6], 1 / 14 | 2 / 11 / 1 | 0 / 2 / 1 / 10 | -9 (142) |
| x.md.p1 | 94 / 68 / 3 / 27 | 3.6 | -5 [-12, +1] | -1 [-6, +4] | +2 [-3, +7] | 159, -3 [-9, +3], 3 / 6 | 4 / 3 / 0 | 2 / 2 / 0 / 3 | +0 (146) |
| x.rand.p1 | 97 / 63 / 5 / 27 | 3.3 | -2 [-6, +2] | -3 [-7, +0] | +2 [+0, +5] | 160, -1 [-5, +2], 1 / 2 | 3 / 0 / 1 | 1 / 2 / 0 / 0 | +0 (157) |
| x.rand.m1 | 96 / 67 / 4 / 25 | 3.3 | -3 [-7, +0] | +0 [-3, +3] | +0 [-3, +3] | 162, -2 [-5, +0], 0 / 2 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +0 (158) |

- F, complete-sentence background (base x_F): x_F 111 / 58 / 8 / 15 -> with the E write 114 / 54 / 7 / 17; dR +3 [-5, +11]; answered-rate change -0.5 pp [-3.1, +2.1]; out / in 4 / 3; mean words 10.1 -> 10.1
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +2 [-6, +10]; vs x.rand.m1 +3 [-6, +12] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -5.7 pp [-9.9, -2.1]; moved out / in 13 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +2 on 145 questions; mean words 3.2 -> 3.1
KEY family=E judge=pro layer=12 A=not_met B=not_met dR=+0 dR_F=+3 dAns=-11 dAns_F=-1 rand_margin=+2

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

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 87 / 1 / 4; rule abstentions 2; mean words 3.1

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 114 / 72 / 2 / 4 | 2.8 | +14 [+5, +24] | -1 [-5, +2] | +0 [-3, +3] | 185, +15 [+7, +24], 18 / 3 | 2 / 1 / 1 | 2 / 0 / 1 / 0 | +14 (178) |
| conf.md.p1 | 99 / 86 / 3 / 4 | 3.2 | -1 [-7, +5] | -2 [-6, +2] | +0 [-3, +3] | 184, +0 [-5, +5], 3 / 3 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +2 (177) |
| conf.rand.p1 | 102 / 85 / 1 / 4 | 3.1 | +2 [-2, +6] | +0 [+0, +0] | +0 [+0, +0] | 187, +2 [-2, +6], 3 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (183) |
| conf.rand.m1 | 98 / 87 / 2 / 5 | 3.1 | -2 [-6, +2] | -2 [-5, +0] | +1 [+0, +3] | 185, -2 [-6, +2], 1 / 3 | 2 / 0 / 1 | 0 / 2 / 0 / 0 | +0 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +14 (7.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +12 [+3, +22]; vs conf.rand.m1 +16 [+7, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +14 on 178 questions; mean words 3.1 -> 2.8
KEY family=E judge=pro layer=12 A=incomplete B=incomplete dR=+14 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+12 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 11:08:50 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

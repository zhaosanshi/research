TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 82 / 56 / 2 / 52 | 4.5 | +4 [-8, +16] | -28 [-38, -18] | +29 [+19, +39] | 136, +14 [+4, +25], 22 / 8 | 30 / 2 / 0 | 10 / 20 / 0 / 2 | +6 (111) |
| x.w.m1 | 77 / 100 / 4 / 11 | 4.1 | -1 [-11, +9] | +11 [+4, +19] | -12 [-19, -6] | 164, -5 [-13, +3], 7 / 12 | 2 / 13 / 2 | 2 / 0 / 6 / 7 | -4 (155) |
| x.md.p1 | 78 / 88 / 1 / 25 | 4.0 | +0 [-8, +8] | +0 [-4, +4] | +2 [+0, +5] | 164, -1 [-9, +6], 7 / 8 | 2 / 2 / 0 | 0 / 2 / 1 / 1 | -1 (156) |
| x.rand.p1 | 77 / 87 / 1 / 27 | 3.9 | -1 [-7, +5] | -2 [-7, +3] | +4 [+1, +8] | 162, -1 [-6, +4], 3 / 4 | 4 / 2 / 0 | 2 / 2 / 2 / 0 | -1 (156) |
| x.rand.m1 | 81 / 88 / 2 / 21 | 4.2 | +3 [-3, +9] | +3 [+0, +7] | -2 [-5, +0] | 166, +2 [-3, +8], 5 / 3 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +1 (160) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 92 / 34 / 0 / 66; dR -10 [-23, +3]; answered-rate change -18.2 pp [-24.0, -12.5]; out / in 36 / 1; mean words 9.3 -> 8.2
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-7, +17]; vs x.rand.m1 +1 [-11, +13] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -14.6 pp [-19.8, -9.4]; moved out / in 30 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +6 on 111 questions; mean words 4.0 -> 4.5
KEY family=E judge=allenai layer=12 A=not_met B=not_met dR=+4 dR_F=-10 dAns=-28 dAns_F=-35 rand_margin=+1

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 83 / 54 / 1 / 54 | +1 [-9, +12] | -27 [-37, -18] | +27 [+18, +37] | 136, +8, 14 / 6 | 28 / 1 / 1 | +3 (113) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 86 / 82 / 0 / 24 | -16 [-27, -5] | +13 [+6, +20] | -13 [-20, -6] | 155, -19, 5 / 24 | 0 / 13 / 0 | -16 (144) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 87 / 102 / 1 / 2 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 189, -1, 1 / 2 | 0 / 0 / 0 | +0 (188) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 87 / 103 / 0 / 2 | -1 [-8, +6] | +1 [+0, +3] | +0 [+0, +0] | 189, -1, 6 / 7 | 0 / 1 / 0 | +2 (181) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 87 / 101 / 2 / 2 | -1 [-6, +4] | -1 [-3, +0] | +0 [+0, +0] | 188, +0, 3 / 3 | 1 / 0 / 1 | +1 (181) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 88 / 103 / 0 / 1 | +0 [-5, +5] | +2 [+0, +5] | -1 [-3, +0] | 189, +0, 3 / 3 | 0 / 2 / 0 | -1 (186) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 112 / 77 / 2 / 1 | +24 [+13, +35] | +0 [-3, +3] | -1 [-3, +0] | 188, +24, 29 / 5 | 1 / 1 / 1 | +13 (158) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 112 / 78 / 1 / 1 | +24 [+13, +35] | +1 [-2, +5] | -1 [-3, +0] | 188, +23, 28 / 5 | 1 / 2 / 1 | +13 (158) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 88 / 103 / 0 / 1 | -29 [-41, -17] | +0 [-3, +3] | +1 [+0, +3] | 190, -28, 6 / 34 | 1 / 1 / 0 | -25 (177) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 92 / 34 / 0 / 66 | -10 [-23, +3] | -35 [-46, -24] | +36 [+25, +47] | 125, +9, 16 / 7 | 36 / 1 / 0 | +1 (87) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 112 / 77 / 2 / 1 | 4.1 | +24 [+13, +35] | +0 [-3, +3] | -1 [-3, +0] | 188, +24 [+13, +35], 29 / 5 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +13 (158) |
| conf.md.p1 | 87 / 103 / 0 / 2 | 3.5 | -1 [-8, +6] | +1 [+0, +3] | +0 [+0, +0] | 189, -1 [-8, +6], 6 / 7 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +2 (181) |
| conf.rand.p1 | 88 / 103 / 0 / 1 | 3.4 | +0 [-5, +5] | +2 [+0, +5] | -1 [-3, +0] | 189, +0 [-5, +5], 3 / 3 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -1 (186) |
| conf.rand.m1 | 87 / 101 / 2 / 2 | 3.5 | -1 [-6, +4] | -1 [-3, +0] | +0 [+0, +0] | 188, +0 [-5, +5], 3 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +1 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +24 (12.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +24 [+13, +35]; vs conf.rand.m1 +25 [+14, +36] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.6, +1.6]; moved out / in 1 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +13 on 158 questions; mean words 3.4 -> 4.1
KEY family=E judge=allenai layer=12 A=incomplete B=incomplete dR=+24 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+24 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 69 / 65 / 5 / 53 | 4.5 | +5 [-6, +16] | -28 [-39, -17] | +29 [+19, +39] | 130, +10 [+2, +19], 15 / 5 | 32 / 4 / 3 | 8 / 24 / 3 / 1 | +7 (109) |
| x.w.m1 | 64 / 107 / 8 / 13 | 4.1 | +0 [-9, +9] | +9 [+2, +17] | -11 [-18, -5] | 159, +0 [-8, +8], 8 / 8 | 3 / 12 / 3 | 3 / 0 / 3 / 9 | +0 (152) |
| x.md.p1 | 64 / 97 / 5 / 26 | 4.0 | +0 [-6, +6] | -1 [-5, +3] | +2 [+0, +5] | 159, -1 [-6, +4], 3 / 4 | 3 / 2 / 1 | 1 / 2 / 2 / 0 | -1 (152) |
| x.rand.p1 | 65 / 94 / 5 / 28 | 3.9 | +1 [-5, +7] | -3 [-8, +2] | +4 [+1, +8] | 157, +0 [-5, +5], 3 / 3 | 5 / 2 / 1 | 1 / 4 / 2 / 0 | +1 (153) |
| x.rand.m1 | 65 / 99 / 5 / 23 | 4.2 | +1 [-3, +6] | +2 [+0, +5] | -1 [-3, +0] | 162, +0 [-4, +4], 2 / 2 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +2 (156) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 74 / 48 / 3 / 67; dR -6 [-19, +7]; answered-rate change -19.8 pp [-25.5, -14.1]; out / in 39 / 1; mean words 9.3 -> 8.2
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +4 [-8, +15]; vs x.rand.m1 +4 [-7, +15] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -14.6 pp [-20.3, -8.9]; moved out / in 32 / 4 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +7 on 109 questions; mean words 4.0 -> 4.5
KEY family=E judge=pro layer=12 A=not_met B=not_met dR=+5 dR_F=-6 dAns=-28 dAns_F=-38 rand_margin=+4

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 68 / 66 / 4 / 54 | +4 [-7, +15] | -24 [-35, -13] | +26 [+17, +36] | 129, +6, 14 / 8 | 29 / 5 / 3 | +3 (110) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 72 / 91 / 3 / 26 | -9 [-20, +2] | +13 [+5, +21] | -14 [-21, -7] | 148, -11, 8 / 19 | 2 / 15 / 2 | -11 (137) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 70 / 112 / 8 / 2 | +0 [-4, +4] | +0 [-3, +3] | +1 [+0, +3] | 181, -1, 1 / 2 | 1 / 1 / 0 | -1 (180) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 68 / 114 / 9 / 1 | -2 [-9, +5] | +0 [-3, +3] | +0 [+0, +0] | 181, -2, 5 / 7 | 1 / 1 / 1 | -1 (173) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 72 / 110 / 9 / 1 | +2 [-4, +8] | +0 [-3, +3] | +0 [+0, +0] | 181, +2, 5 / 3 | 1 / 1 / 1 | +2 (174) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 67 / 114 / 11 / 0 | -3 [-9, +3] | -1 [-4, +2] | -1 [-3, +0] | 180, -3, 3 / 6 | 2 / 1 / 2 | -3 (178) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 89 / 90 / 9 / 4 | +19 [+7, +31] | -3 [-9, +3] | +3 [-1, +8] | 176, +19, 26 / 7 | 6 / 3 / 4 | +6 (150) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 90 / 90 / 8 / 4 | +20 [+8, +32] | -2 [-8, +3] | +3 [-1, +8] | 177, +20, 27 / 7 | 5 / 3 / 3 | +8 (151) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 76 / 108 / 7 / 1 | -22 [-35, -9] | +0 [-6, +6] | -3 [-8, +1] | 179, -19, 11 / 30 | 5 / 5 / 4 | -18 (167) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 74 / 48 / 3 / 67 | -6 [-19, +7] | -38 [-49, -27] | +37 [+26, +48] | 121, +9, 17 / 8 | 39 / 1 / 1 | +6 (86) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 89 / 90 / 9 / 4 | 4.1 | +19 [+7, +31] | -3 [-9, +3] | +3 [-1, +8] | 176, +19 [+8, +30], 26 / 7 | 6 / 3 / 4 | 3 / 3 / 3 / 0 | +6 (150) |
| conf.md.p1 | 68 / 114 / 9 / 1 | 3.5 | -2 [-9, +5] | +0 [-3, +3] | +0 [+0, +0] | 181, -2 [-9, +5], 5 / 7 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | -1 (173) |
| conf.rand.p1 | 67 / 114 / 11 / 0 | 3.4 | -3 [-9, +3] | -1 [-4, +2] | -1 [-3, +0] | 180, -3 [-9, +3], 3 / 6 | 2 / 1 / 2 | 0 / 2 / 0 / 1 | -3 (178) |
| conf.rand.m1 | 72 / 110 / 9 / 1 | 3.5 | +2 [-4, +8] | +0 [-3, +3] | +0 [+0, +0] | 181, +2 [-3, +8], 5 / 3 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +2 (174) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +19 (9.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +22 [+10, +34]; vs conf.rand.m1 +17 [+5, +29] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.7, +1.6]; moved out / in 6 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 150 questions; mean words 3.4 -> 4.1
KEY family=E judge=pro layer=12 A=incomplete B=not_met dR=+19 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+17 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 13:15:01 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

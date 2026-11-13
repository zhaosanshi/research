TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 88 / 60 / 0 / 44 | 4.4 | +10 [-2, +22] | -18 [-27, -9] | +21 [+13, +30] | 145, +15 [+5, +25], 21 / 6 | 21 / 3 / 0 | 8 / 13 / 3 / 0 | +8 (122) |
| x.w.m1 | 77 / 97 / 1 / 17 | 3.8 | -1 [-11, +9] | +8 [+2, +15] | -6 [-13, +0] | 164, -4 [-13, +5], 8 / 12 | 2 / 10 / 0 | 2 / 0 / 5 / 5 | -4 (156) |
| x.md.p1 | 85 / 80 / 2 / 25 | 4.2 | +7 [+0, +14] | -1 [-5, +2] | +2 [+0, +5] | 164, +7 [+1, +13], 9 / 2 | 2 / 1 / 0 | 1 / 1 / 1 / 0 | +5 (151) |
| x.rand.p1 | 79 / 85 / 2 / 26 | 4.0 | +1 [-5, +7] | -2 [-6, +2] | +3 [+0, +7] | 163, +3 [-2, +8], 5 / 2 | 3 / 1 / 0 | 2 / 1 / 0 / 1 | +2 (160) |
| x.rand.m1 | 82 / 85 / 2 / 23 | 4.2 | +4 [+0, +9] | +1 [+0, +3] | +0 [+0, +0] | 166, +3 [-1, +8], 4 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (154) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 104 / 41 / 0 / 47; dR +2 [-10, +14]; answered-rate change -8.3 pp [-13.0, -4.2]; out / in 18 / 2; mean words 9.3 -> 9.8
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +9 [-3, +21]; vs x.rand.m1 +6 [-6, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -9.4 pp [-14.1, -4.7]; moved out / in 21 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 122 questions; mean words 4.0 -> 4.4
KEY family=E judge=allenai layer=9 A=not_met B=not_met dR=+10 dR_F=+2 dAns=-18 dAns_F=-16 rand_margin=+6

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 9

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 87 / 57 / 3 / 45 | +5 [-5, +15] | -20 [-29, -12] | +18 [+11, +26] | 143, +9, 15 / 6 | 21 / 1 / 3 | +4 (120) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 83 / 80 / 1 / 28 | -19 [-30, -8] | +8 [+2, +14] | -9 [-15, -4] | 154, -19, 6 / 25 | 1 / 9 / 1 | -17 (144) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 88 / 101 / 1 / 2 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 189, +0, 1 / 1 | 0 / 0 / 0 | +0 (189) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 86 / 104 / 0 / 2 | -2 [-9, +5] | +1 [+0, +3] | +0 [+0, +0] | 189, -2, 5 / 7 | 0 / 1 / 0 | -1 (177) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 83 / 106 / 1 / 2 | -5 [-11, +1] | +0 [+0, +0] | +0 [+0, +0] | 189, -5, 2 / 7 | 0 / 0 / 0 | -4 (181) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 89 / 100 / 1 / 2 | +1 [-4, +6] | +0 [-3, +3] | +0 [+0, +0] | 188, +2, 4 / 2 | 1 / 1 / 1 | +2 (187) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 113 / 77 / 1 / 1 | +25 [+14, +36] | +1 [-2, +5] | -1 [-3, +0] | 188, +25, 29 / 4 | 1 / 2 / 1 | +14 (160) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 110 / 80 / 1 / 1 | +22 [+10, +34] | +1 [-2, +5] | -1 [-3, +0] | 188, +21, 28 / 7 | 1 / 2 / 1 | +12 (160) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 99 / 92 / 1 / 0 | -18 [-29, -7] | +0 [-3, +3] | +0 [+0, +0] | 190, -18, 8 / 26 | 1 / 1 / 1 | -17 (180) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 104 / 41 / 0 / 47 | +2 [-10, +14] | -16 [-25, -8] | +17 [+9, +26] | 143, +11, 18 / 7 | 18 / 2 / 0 | +6 (92) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 113 / 77 / 1 / 1 | 4.1 | +25 [+14, +36] | +1 [-2, +5] | -1 [-3, +0] | 188, +25 [+14, +36], 29 / 4 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +14 (160) |
| conf.md.p1 | 86 / 104 / 0 / 2 | 3.7 | -2 [-9, +5] | +1 [+0, +3] | +0 [+0, +0] | 189, -2 [-9, +5], 5 / 7 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -1 (177) |
| conf.rand.p1 | 89 / 100 / 1 / 2 | 3.3 | +1 [-4, +6] | +0 [-3, +3] | +0 [+0, +0] | 188, +2 [-3, +7], 4 / 2 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +2 (187) |
| conf.rand.m1 | 83 / 106 / 1 / 2 | 3.5 | -5 [-11, +1] | +0 [+0, +0] | +0 [+0, +0] | 189, -5 [-11, +1], 2 / 7 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -4 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +25 (13.0% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +24 [+13, +35]; vs conf.rand.m1 +30 [+19, +41] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-1.0, +2.6]; moved out / in 1 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +14 on 160 questions; mean words 3.4 -> 4.1
KEY family=E judge=allenai layer=9 A=incomplete B=incomplete dR=+25 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+24 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 67 / 76 / 4 / 45 | 4.4 | +3 [-8, +14] | -19 [-29, -10] | +21 [+13, +30] | 140, +8 [-1, +17], 15 / 7 | 22 / 3 / 3 | 8 / 14 / 3 / 0 | +8 (119) |
| x.w.m1 | 60 / 107 / 8 / 17 | 3.8 | -4 [-13, +5] | +5 [-1, +12] | -7 [-13, -2] | 159, -4 [-12, +4], 6 / 10 | 3 / 8 / 2 | 2 / 1 / 2 / 6 | -4 (153) |
| x.md.p1 | 65 / 95 / 6 / 26 | 4.2 | +1 [-6, +8] | -2 [-6, +2] | +2 [+0, +5] | 159, +2 [-3, +8], 5 / 3 | 3 / 1 / 1 | 2 / 1 / 1 / 0 | +3 (147) |
| x.rand.p1 | 65 / 94 / 6 / 27 | 4.0 | +1 [-3, +5] | -3 [-7, +0] | +3 [+0, +7] | 159, +1 [-3, +5], 3 / 2 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | +2 (156) |
| x.rand.m1 | 66 / 97 / 5 / 24 | 4.2 | +2 [-3, +7] | +1 [+0, +3] | +0 [+0, +0] | 162, +1 [-3, +5], 3 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +3 (151) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 81 / 60 / 4 / 47; dR +1 [-11, +13]; answered-rate change -9.9 pp [-15.1, -5.2]; out / in 22 / 3; mean words 9.3 -> 9.8
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +2 [-8, +12]; vs x.rand.m1 +1 [-11, +13] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -9.9 pp [-15.1, -5.2]; moved out / in 22 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 119 questions; mean words 4.0 -> 4.4
KEY family=E judge=pro layer=9 A=not_met B=not_met dR=+3 dR_F=+1 dAns=-19 dAns_F=-19 rand_margin=+1

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 9

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 74 / 64 / 8 / 46 | +10 [-1, +21] | -20 [-30, -11] | +18 [+11, +26] | 135, +13, 18 / 5 | 23 / 3 / 5 | +8 (114) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 69 / 89 / 4 / 30 | -12 [-22, -2] | +8 [+1, +15] | -10 [-17, -4] | 148, -11, 7 / 18 | 2 / 10 / 2 | -9 (137) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 71 / 111 / 8 / 2 | +1 [-2, +5] | +0 [-3, +3] | +1 [+0, +3] | 181, +0, 1 / 1 | 1 / 1 / 0 | +0 (181) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 71 / 109 / 10 / 2 | +1 [-5, +8] | -2 [-6, +2] | +1 [+0, +3] | 179, +0, 5 / 5 | 3 / 1 / 2 | -2 (168) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 72 / 111 / 8 / 1 | +2 [-3, +8] | +1 [+0, +3] | +0 [+0, +0] | 182, +1, 4 / 3 | 0 / 1 / 0 | +1 (174) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 66 / 115 / 9 / 2 | -4 [-10, +1] | -1 [-5, +2] | +1 [+0, +3] | 180, -3, 2 / 5 | 2 / 1 / 1 | -3 (179) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 91 / 88 / 9 / 4 | +21 [+10, +32] | -3 [-10, +4] | +3 [-1, +8] | 175, +20, 24 / 4 | 7 / 4 / 5 | +12 (152) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 89 / 90 / 9 / 4 | +19 [+8, +30] | -3 [-10, +4] | +3 [-1, +8] | 175, +18, 22 / 4 | 7 / 4 / 5 | +11 (152) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 85 / 99 / 7 / 1 | -13 [-26, +0] | +0 [-6, +6] | -3 [-7, +0] | 180, -10, 15 / 25 | 4 / 4 / 4 | -11 (171) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 81 / 60 / 4 / 47 | +1 [-11, +13] | -19 [-29, -10] | +17 [+8, +27] | 138, +8, 16 / 8 | 22 / 3 / 2 | +4 (90) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 91 / 88 / 9 / 4 | 4.1 | +21 [+10, +32] | -3 [-10, +4] | +3 [-1, +8] | 175, +20 [+10, +30], 24 / 4 | 7 / 4 / 5 | 3 / 4 / 4 / 0 | +12 (152) |
| conf.md.p1 | 71 / 109 / 10 / 2 | 3.7 | +1 [-5, +8] | -2 [-6, +2] | +1 [+0, +3] | 179, +0 [-6, +6], 5 / 5 | 3 / 1 / 2 | 0 / 3 / 1 / 0 | -2 (168) |
| conf.rand.p1 | 66 / 115 / 9 / 2 | 3.3 | -4 [-10, +1] | -1 [-5, +2] | +1 [+0, +3] | 180, -3 [-8, +2], 2 / 5 | 2 / 1 / 1 | 1 / 1 / 0 / 1 | -3 (179) |
| conf.rand.m1 | 72 / 111 / 8 / 1 | 3.5 | +2 [-3, +8] | +1 [+0, +3] | +0 [+0, +0] | 182, +1 [-4, +6], 4 / 3 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (174) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +21 (10.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +25 [+15, +35]; vs conf.rand.m1 +19 [+8, +30] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-5.2, +2.1]; moved out / in 7 / 4 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +12 on 152 questions; mean words 3.4 -> 4.1
KEY family=E judge=pro layer=9 A=incomplete B=not_met dR=+21 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+19 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 12:58:14 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 81 / 64 / 2 / 45 | 4.1 | +3 [-8, +14] | -21 [-31, -12] | +22 [+13, +31] | 143, +8 [-1, +18], 16 / 8 | 23 / 2 / 0 | 6 / 17 / 1 / 1 | +5 (126) |
| x.w.m1 | 76 / 97 / 2 / 17 | 3.8 | -2 [-12, +7] | +7 [+0, +14] | -6 [-12, -1] | 163, -4 [-12, +4], 6 / 10 | 3 / 10 / 2 | 3 / 0 / 5 / 5 | -4 (152) |
| x.md.p1 | 87 / 64 / 1 / 40 | 4.3 | +9 [-1, +19] | -15 [-24, -7] | +17 [+10, +25] | 149, +11 [+2, +20], 17 / 6 | 17 / 2 / 0 | 4 / 13 / 2 / 0 | +6 (135) |
| x.rand.p1 | 77 / 88 / 2 / 25 | 4.1 | -1 [-8, +6] | -1 [-5, +2] | +2 [+0, +5] | 164, -1 [-8, +6], 6 / 7 | 2 / 1 / 0 | 0 / 2 / 0 / 1 | -1 (154) |
| x.rand.m1 | 83 / 84 / 2 / 23 | 4.1 | +5 [-2, +12] | +1 [+0, +3] | +0 [+0, +0] | 166, +4 [-3, +11], 8 / 4 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (156) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 93 / 56 / 0 / 43; dR -9 [-20, +2]; answered-rate change -6.2 pp [-10.4, -2.1]; out / in 15 / 3; mean words 9.3 -> 9.8
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +4 [-6, +15]; vs x.rand.m1 -2 [-13, +9] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -10.9 pp [-16.1, -6.2]; moved out / in 23 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 126 questions; mean words 4.0 -> 4.1
KEY family=E judge=allenai layer=15 A=not_met B=not_met dR=+3 dR_F=-9 dAns=-21 dAns_F=-12 rand_margin=-2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 15

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 87 / 62 / 0 / 43 | +5 [-6, +16] | -15 [-24, -7] | +16 [+8, +25] | 147, +7, 17 / 10 | 17 / 2 / 0 | +4 (129) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 96 / 71 / 1 / 24 | -6 [-17, +5] | +12 [+5, +20] | -13 [-21, -6] | 153, -8, 8 / 16 | 2 / 14 / 1 | -6 (141) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 87 / 102 / 1 / 2 | -1 [-5, +3] | +0 [+0, +0] | +0 [+0, +0] | 189, -1, 2 / 3 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 95 / 94 / 1 / 2 | +7 [-2, +16] | +0 [-4, +4] | +0 [-3, +3] | 187, +7, 13 / 6 | 2 / 2 / 1 | +3 (170) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 84 / 105 / 1 / 2 | -4 [-9, +0] | +0 [+0, +0] | +0 [+0, +0] | 189, -4, 1 / 5 | 0 / 0 / 0 | -4 (187) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 86 / 103 / 1 / 2 | -2 [-8, +3] | +0 [-3, +3] | +0 [+0, +0] | 188, -1, 3 / 4 | 1 / 1 / 1 | +2 (177) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 100 / 87 / 3 / 2 | +12 [+2, +22] | -2 [-7, +3] | +0 [-3, +3] | 185, +14, 19 / 5 | 4 / 2 / 3 | +9 (172) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 109 / 80 / 1 / 2 | +21 [+11, +32] | +0 [-4, +4] | +0 [-3, +3] | 187, +20, 24 / 4 | 2 / 2 / 1 | +11 (166) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 98 / 94 / 0 / 0 | -19 [-30, -8] | +1 [+0, +3] | +0 [+0, +0] | 191, -20, 7 / 27 | 0 / 1 / 0 | -16 (180) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 93 / 56 / 0 / 43 | -9 [-20, +2] | -12 [-20, -4] | +13 [+5, +21] | 146, -3, 9 / 12 | 15 / 3 / 0 | -3 (106) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 87 / 3 / 2 | 3.5 | +12 [+2, +22] | -2 [-7, +3] | +0 [-3, +3] | 185, +14 [+5, +23], 19 / 5 | 4 / 2 / 3 | 3 / 1 / 1 / 1 | +9 (172) |
| conf.md.p1 | 95 / 94 / 1 / 2 | 3.7 | +7 [-2, +16] | +0 [-4, +4] | +0 [-3, +3] | 187, +7 [-1, +15], 13 / 6 | 2 / 2 / 1 | 1 / 1 / 1 / 1 | +3 (170) |
| conf.rand.p1 | 86 / 103 / 1 / 2 | 3.6 | -2 [-8, +3] | +0 [-3, +3] | +0 [+0, +0] | 188, -1 [-6, +4], 3 / 4 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +2 (177) |
| conf.rand.m1 | 84 / 105 / 1 / 2 | 3.3 | -4 [-9, +0] | +0 [+0, +0] | +0 [+0, +0] | 189, -4 [-9, +0], 1 / 5 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -4 (187) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +12 (6.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +14 [+4, +24]; vs conf.rand.m1 +16 [+6, +27] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-3.6, +1.6]; moved out / in 4 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +9 on 172 questions; mean words 3.4 -> 3.5
KEY family=E judge=allenai layer=15 A=incomplete B=not_met dR=+12 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+14 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 68 / 74 / 4 / 46 | 4.1 | +4 [-6, +14] | -20 [-30, -10] | +22 [+13, +31] | 138, +6 [-2, +14], 12 / 6 | 24 / 4 / 2 | 5 / 19 / 3 / 1 | +4 (121) |
| x.w.m1 | 61 / 108 / 6 / 17 | 3.8 | -3 [-11, +5] | +7 [+2, +13] | -7 [-13, -2] | 162, -4 [-12, +4], 6 / 10 | 0 / 7 / 0 | 0 / 0 / 1 / 6 | -4 (152) |
| x.md.p1 | 68 / 79 / 4 / 41 | 4.3 | +4 [-4, +12] | -15 [-24, -6] | +17 [+10, +25] | 144, +5 [-2, +12], 9 / 4 | 18 / 3 / 2 | 3 / 15 / 2 / 1 | +2 (130) |
| x.rand.p1 | 63 / 99 / 4 / 26 | 4.1 | -1 [-8, +6] | +0 [-5, +5] | +2 [+0, +5] | 159, -3 [-9, +3], 3 / 6 | 3 / 3 / 1 | 1 / 2 / 3 / 0 | +0 (149) |
| x.rand.m1 | 61 / 100 / 7 / 24 | 4.1 | -3 [-8, +2] | -1 [-3, +0] | +0 [+0, +0] | 161, -2 [-7, +3], 2 / 4 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -1 (152) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 83 / 65 / 2 / 42; dR +3 [-7, +13]; answered-rate change -6.2 pp [-10.4, -2.1]; out / in 15 / 3; mean words 9.3 -> 9.8
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-5, +15]; vs x.rand.m1 +7 [-3, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -10.4 pp [-15.6, -5.2]; moved out / in 24 / 4 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +4 on 121 questions; mean words 4.0 -> 4.1
KEY family=E judge=pro layer=15 A=not_met B=not_met dR=+4 dR_F=+3 dAns=-20 dAns_F=-12 rand_margin=+5

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 15

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 70 / 73 / 6 / 43 | +6 [-4, +17] | -15 [-25, -6] | +15 [+7, +24] | 138, +5, 12 / 7 | 20 / 5 / 3 | +4 (124) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 79 / 84 / 4 / 25 | -2 [-12, +8] | +13 [+5, +21] | -15 [-23, -8] | 148, -4, 8 / 12 | 2 / 15 / 2 | -4 (136) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 70 / 112 / 8 / 2 | +0 [-4, +4] | +0 [-3, +3] | +1 [+0, +3] | 181, -1, 1 / 2 | 1 / 1 / 0 | -1 (178) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 77 / 101 / 12 / 2 | +7 [-1, +15] | -4 [-9, +0] | +1 [+0, +3] | 177, +7, 12 / 5 | 5 / 1 / 4 | +3 (162) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 71 / 111 / 9 / 1 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 182, +1, 2 / 1 | 0 / 0 / 0 | +1 (180) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 70 / 111 / 9 / 2 | +0 [-6, +6] | -1 [-5, +2] | +1 [+0, +3] | 180, +0, 3 / 3 | 2 / 1 / 1 | +1 (170) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 80 / 97 / 13 / 2 | +10 [+1, +19] | -5 [-10, -1] | +1 [+0, +3] | 177, +12, 17 / 5 | 5 / 0 / 4 | +8 (165) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 89 / 91 / 9 / 3 | +19 [+8, +30] | -2 [-7, +3] | +2 [+0, +5] | 178, +20, 26 / 6 | 4 / 2 / 2 | +9 (159) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 78 / 105 / 7 / 2 | -20 [-32, -8] | -1 [-6, +4] | -2 [-5, +0] | 180, -17, 9 / 26 | 4 / 3 / 4 | -16 (170) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 83 / 65 / 2 / 42 | +3 [-7, +13] | -12 [-20, -4] | +12 [+4, +20] | 145, +4, 12 / 8 | 15 / 3 / 0 | +8 (105) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 80 / 97 / 13 / 2 | 3.5 | +10 [+1, +19] | -5 [-10, -1] | +1 [+0, +3] | 177, +12 [+3, +21], 17 / 5 | 5 / 0 / 4 | 2 / 3 / 0 / 0 | +8 (165) |
| conf.md.p1 | 77 / 101 / 12 / 2 | 3.7 | +7 [-1, +15] | -4 [-9, +0] | +1 [+0, +3] | 177, +7 [-1, +15], 12 / 5 | 5 / 1 / 4 | 1 / 4 / 1 / 0 | +3 (162) |
| conf.rand.p1 | 70 / 111 / 9 / 2 | 3.6 | +0 [-6, +6] | -1 [-5, +2] | +1 [+0, +3] | 180, +0 [-5, +5], 3 / 3 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +1 (170) |
| conf.rand.m1 | 71 / 111 / 9 / 1 | 3.3 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 182, +1 [-2, +5], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +10 [+1, +19]; vs conf.rand.m1 +9 [-1, +19] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.6 pp [-5.2, -0.5]; moved out / in 5 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +8 on 165 questions; mean words 3.4 -> 3.5
KEY family=E judge=pro layer=15 A=incomplete B=not_met dR=+10 dR_F=NA dAns=-5 dAns_F=NA rand_margin=+9 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 13:27:58 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

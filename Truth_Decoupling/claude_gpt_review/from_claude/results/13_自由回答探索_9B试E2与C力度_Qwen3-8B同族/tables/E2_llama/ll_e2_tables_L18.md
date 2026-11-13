TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 84 / 78 / 1 / 29 | 4.3 | +6 [-4, +15] | -4 [-10, +2] | +6 [+1, +12] | 159, +7 [-1, +15], 13 / 6 | 7 / 3 / 0 | 3 / 4 / 2 / 1 | +4 (144) |
| x.w.m1 | 85 / 86 / 2 / 19 | 4.2 | +7 [+0, +15] | +5 [+1, +10] | -4 [-8, -1] | 166, +3 [-3, +10], 7 / 4 | 0 / 5 / 0 | 0 / 0 / 4 / 1 | +1 (156) |
| x.md.p1 | 83 / 79 / 2 / 28 | 4.2 | +5 [-4, +14] | -4 [-10, +1] | +5 [+0, +10] | 160, +6 [-2, +14], 12 / 6 | 6 / 2 / 0 | 2 / 4 / 1 / 1 | +3 (151) |
| x.rand.p1 | 81 / 86 / 1 / 24 | 4.0 | +3 [-2, +8] | +1 [-2, +5] | +1 [+0, +3] | 165, +3 [-1, +7], 4 / 1 | 1 / 2 / 0 | 1 / 0 / 1 / 1 | +4 (157) |
| x.rand.m1 | 76 / 88 / 3 / 25 | 4.0 | -2 [-8, +3] | -2 [-7, +3] | +2 [+0, +5] | 162, -1 [-6, +4], 3 / 4 | 4 / 2 / 2 | 1 / 3 / 0 / 2 | -1 (159) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 96 / 59 / 2 / 35; dR -6 [-16, +3]; answered-rate change -3.1 pp [-6.2, -0.5]; out / in 7 / 1; mean words 9.3 -> 9.3
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +3 [-7, +13]; vs x.rand.m1 +8 [-2, +19] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.1 pp [-5.2, +1.0]; moved out / in 7 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +4 on 144 questions; mean words 4.0 -> 4.3
KEY family=E judge=allenai layer=18 A=not_met B=not_met dR=+6 dR_F=-6 dAns=-4 dAns_F=-6 rand_margin=+3

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 18

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 88 / 71 / 2 / 31 | +6 [-3, +15] | -5 [-11, +0] | +4 [+0, +9] | 158, +5, 13 / 8 | 6 / 1 / 1 | +2 (142) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 99 / 60 / 0 / 33 | -3 [-11, +5] | +4 [-1, +10] | -4 [-10, +1] | 153, -2, 7 / 9 | 2 / 6 / 0 | -2 (146) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 88 / 101 / 1 / 2 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 189, +0, 2 / 2 | 0 / 0 / 0 | -1 (187) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 92 / 97 / 2 / 1 | +4 [-4, +12] | +0 [-4, +4] | -1 [-3, +0] | 187, +5, 10 / 5 | 2 / 2 / 2 | +0 (175) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 87 / 102 / 1 / 2 | -1 [-5, +3] | +0 [+0, +0] | +0 [+0, +0] | 189, -1, 2 / 3 | 0 / 0 / 0 | -1 (184) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 86 / 103 / 1 / 2 | -2 [-9, +5] | +0 [-3, +3] | +0 [+0, +0] | 188, -1, 5 / 6 | 1 / 1 / 1 | +0 (183) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 94 / 95 / 2 / 1 | +6 [-3, +15] | +0 [-4, +4] | -1 [-3, +0] | 187, +7, 12 / 5 | 2 / 2 / 2 | +5 (173) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 102 / 86 / 2 / 2 | +14 [+5, +24] | -1 [-5, +2] | +0 [+0, +0] | 187, +16, 20 / 4 | 2 / 1 / 2 | +11 (171) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 107 / 83 / 0 / 2 | -10 [-20, -1] | -1 [-5, +2] | +2 [+0, +5] | 189, -9, 6 / 15 | 2 / 1 / 0 | -7 (179) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 96 / 59 / 2 / 35 | -6 [-16, +3] | -6 [-12, -1] | +5 [+0, +10] | 154, -4, 8 / 12 | 7 / 1 / 1 | -5 (130) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 94 / 95 / 2 / 1 | 3.6 | +6 [-3, +15] | +0 [-4, +4] | -1 [-3, +0] | 187, +7 [-1, +15], 12 / 5 | 2 / 2 / 2 | 2 / 0 / 1 / 1 | +5 (173) |
| conf.md.p1 | 92 / 97 / 2 / 1 | 3.5 | +4 [-4, +12] | +0 [-4, +4] | -1 [-3, +0] | 187, +5 [-2, +13], 10 / 5 | 2 / 2 / 2 | 2 / 0 / 1 / 1 | +0 (175) |
| conf.rand.p1 | 86 / 103 / 1 / 2 | 3.4 | -2 [-9, +5] | +0 [-3, +3] | +0 [+0, +0] | 188, -1 [-7, +5], 5 / 6 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (183) |
| conf.rand.m1 | 87 / 102 / 1 / 2 | 3.4 | -1 [-5, +3] | +0 [+0, +0] | +0 [+0, +0] | 189, -1 [-5, +3], 2 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +8 [+0, +17]; vs conf.rand.m1 +7 [-3, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-2.1, +2.1]; moved out / in 2 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +5 on 173 questions; mean words 3.4 -> 3.6
KEY family=E judge=allenai layer=18 A=incomplete B=not_met dR=+6 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+7 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 70 / 87 / 5 / 30 | 4.3 | +6 [-2, +15] | -5 [-11, +1] | +6 [+1, +12] | 155, +5 [-2, +13], 10 / 5 | 7 / 2 / 1 | 1 / 6 / 2 / 0 | +4 (141) |
| x.w.m1 | 64 / 99 / 8 / 21 | 4.2 | +0 [-6, +6] | +1 [-2, +5] | -3 [-7, +0] | 161, +0 [-6, +6], 4 / 4 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +0 (151) |
| x.md.p1 | 65 / 91 / 7 / 29 | 4.2 | +1 [-7, +9] | -6 [-12, +0] | +5 [+0, +10] | 154, +1 [-6, +8], 7 / 6 | 8 / 2 / 2 | 2 / 6 / 2 / 0 | +0 (144) |
| x.rand.p1 | 67 / 96 / 5 / 24 | 4.0 | +3 [-1, +8] | +1 [+0, +3] | +0 [+0, +0] | 162, +2 [-2, +6], 3 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +3 (154) |
| x.rand.m1 | 64 / 97 / 5 / 26 | 4.0 | +0 [-5, +5] | -1 [-5, +2] | +2 [+0, +5] | 160, -1 [-6, +3], 2 / 3 | 2 / 1 / 0 | 0 / 2 / 1 / 0 | +0 (157) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 83 / 72 / 2 / 35; dR +3 [-6, +12]; answered-rate change -2.6 pp [-5.7, +0.5]; out / in 7 / 2; mean words 9.3 -> 9.3
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +3 [-5, +11]; vs x.rand.m1 +6 [-2, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.6 pp [-5.7, +0.5]; moved out / in 7 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +4 on 141 questions; mean words 4.0 -> 4.3
KEY family=E judge=pro layer=18 A=met B=not_met dR=+6 dR_F=+3 dAns=-5 dAns_F=-5 rand_margin=+3

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 18

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 69 / 83 / 7 / 33 | +5 [-4, +14] | -6 [-13, +0] | +5 [+0, +11] | 149, +3, 10 / 7 | 9 / 3 / 3 | +1 (136) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 74 / 77 / 7 / 34 | -7 [-16, +2] | +1 [-6, +8] | -6 [-12, -1] | 144, -4, 6 / 10 | 6 / 7 / 5 | -4 (137) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 70 / 111 / 10 / 1 | +0 [-3, +3] | -1 [-3, +0] | +0 [+0, +0] | 181, +0, 1 / 1 | 1 / 0 / 1 | +0 (179) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 75 / 102 / 13 / 2 | +5 [-3, +13] | -5 [-10, -1] | +1 [+0, +3] | 177, +7, 11 / 4 | 5 / 0 / 4 | +4 (168) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 68 / 113 / 10 / 1 | -2 [-7, +3] | -1 [-3, +0] | +0 [+0, +0] | 181, -2, 2 / 4 | 1 / 0 / 1 | -2 (176) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 69 / 112 / 9 / 2 | -1 [-7, +5] | -1 [-5, +2] | +1 [+0, +3] | 180, -1, 3 / 4 | 2 / 1 / 1 | -1 (176) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 75 / 101 / 13 / 3 | +5 [-2, +12] | -6 [-11, -2] | +2 [+0, +5] | 176, +7, 9 / 2 | 6 / 0 / 5 | +4 (165) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 80 / 96 / 14 / 2 | +10 [+1, +20] | -6 [-12, -1] | +1 [+0, +3] | 175, +11, 16 / 5 | 7 / 1 / 6 | +8 (162) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 81 / 101 / 7 / 3 | -17 [-27, -7] | -2 [-6, +2] | -1 [-3, +0] | 181, -14, 5 / 19 | 3 / 1 / 3 | -13 (173) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 83 / 72 / 2 / 35 | +3 [-6, +12] | -5 [-11, +1] | +5 [-1, +11] | 153, +4, 10 / 6 | 7 / 2 / 0 | +2 (129) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 75 / 101 / 13 / 3 | 3.6 | +5 [-2, +12] | -6 [-11, -2] | +2 [+0, +5] | 176, +7 [+1, +14], 9 / 2 | 6 / 0 / 5 | 2 / 4 / 0 / 0 | +4 (165) |
| conf.md.p1 | 75 / 102 / 13 / 2 | 3.5 | +5 [-3, +13] | -5 [-10, -1] | +1 [+0, +3] | 177, +7 [+0, +15], 11 / 4 | 5 / 0 / 4 | 2 / 3 / 0 / 0 | +4 (168) |
| conf.rand.p1 | 69 / 112 / 9 / 2 | 3.4 | -1 [-7, +5] | -1 [-5, +2] | +1 [+0, +3] | 180, -1 [-6, +4], 3 / 4 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -1 (176) |
| conf.rand.m1 | 68 / 113 / 10 / 1 | 3.4 | -2 [-7, +3] | -1 [-3, +0] | +0 [+0, +0] | 181, -2 [-7, +3], 2 / 4 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -2 (176) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +6 [-1, +13]; vs conf.rand.m1 +7 [-1, +15] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -3.1 pp [-5.7, -1.0]; moved out / in 6 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +4 on 165 questions; mean words 3.4 -> 3.6
KEY family=E judge=pro layer=18 A=not_met B=not_met dR=+5 dR_F=NA dAns=-6 dAns_F=NA rand_margin=+6 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 13:50:54 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

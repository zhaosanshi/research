TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 67 / 51 / 2 / 72 | 4.7 | +6 [+1, +12] | +1 [-2, +5] | -1 [-5, +2] | 116, +4 [-1, +9], 5 / 1 | 1 / 2 / 0 | 0 / 1 / 2 / 0 | +4 (107) |
| x.w.m1 | 61 / 55 / 2 / 74 | 4.8 | +0 [-5, +5] | -1 [-5, +2] | +1 [-2, +5] | 115, +1 [-3, +5], 3 / 2 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +0 (105) |
| x.md.p1 | 59 / 56 / 3 / 74 | 4.5 | -2 [-7, +3] | -2 [-6, +2] | +1 [-2, +4] | 114, -1 [-5, +3], 2 / 3 | 3 / 1 / 1 | 1 / 2 / 0 / 1 | +0 (105) |
| x.rand.p1 | 62 / 55 / 2 / 73 | 4.8 | +1 [+0, +3] | +0 [-3, +3] | +0 [-3, +3] | 116, +1 [+0, +3], 1 / 0 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | +0 (109) |
| x.rand.m1 | 64 / 52 / 2 / 74 | 4.7 | +3 [-1, +8] | -1 [-3, +0] | +1 [+0, +3] | 116, +3 [-1, +8], 4 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +2 (109) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 73 / 61 / 2 / 56; dR +2 [-2, +6]; answered-rate change -0.5 pp [-1.6, +0.0]; out / in 1 / 0; mean words 7.9 -> 7.8
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +5 [+0, +10]; vs x.rand.m1 +3 [-2, +8] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.5 pp [-1.0, +2.6]; moved out / in 1 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +4 on 107 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=27 A=met B=not_met dR=+6 dR_F=+2 dAns=+1 dAns_F=-1 rand_margin=+3

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 27

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 47 / 47 / 2 / 96 | +3 [-1, +7] | +1 [-2, +5] | -1 [-5, +2] | 92, +1, 2 / 1 | 1 / 2 / 0 | +1 (86) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 44 / 37 / 0 / 111 | -1 [-7, +5] | -3 [-8, +2] | +3 [-2, +8] | 79, +2, 3 / 1 | 5 / 2 / 0 | +1 (74) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 2 / 1 | -2 [-6, +2] | -1 [-3, +0] | +1 [+0, +3] | 189, -1, 1 / 2 | 1 / 0 / 0 | -1 (187) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 97 / 91 / 3 / 1 | -3 [-12, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, -2, 9 / 11 | 2 / 0 / 1 | -3 (179) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 95 / 92 / 4 / 1 | -5 [-11, +1] | -3 [-7, +0] | +1 [+0, +3] | 187, -2, 2 / 4 | 3 / 0 / 2 | -2 (184) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 96 / 93 / 3 / 0 | -4 [-11, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -4, 3 / 7 | 1 / 0 / 1 | -4 (181) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 106 / 82 / 3 / 1 | +6 [-1, +13] | -2 [-5, +0] | +1 [+0, +3] | 188, +7, 10 / 3 | 2 / 0 / 1 | +5 (172) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 104 / 84 / 3 / 1 | +4 [-3, +11] | -2 [-5, +0] | +1 [+0, +3] | 188, +5, 8 / 3 | 2 / 0 / 1 | +3 (174) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 106 / 81 / 4 / 1 | -7 [-15, +1] | +1 [+0, +3] | +0 [+0, +0] | 186, -8, 4 / 12 | 0 / 1 / 0 | -7 (165) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 73 / 61 / 2 / 56 | +2 [-2, +6] | -1 [-3, +0] | +0 [+0, +0] | 134, +3, 3 / 0 | 1 / 0 / 1 | +2 (126) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 106 / 82 / 3 / 1 | 4.6 | +6 [-1, +13] | -2 [-5, +0] | +1 [+0, +3] | 188, +7 [+0, +14], 10 / 3 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +5 (172) |
| conf.md.p1 | 97 / 91 / 3 / 1 | 4.3 | -3 [-12, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, -2 [-11, +7], 9 / 11 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -3 (179) |
| conf.rand.p1 | 96 / 93 / 3 / 0 | 4.8 | -4 [-11, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -4 [-11, +2], 3 / 7 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -4 (181) |
| conf.rand.m1 | 95 / 92 / 4 / 1 | 4.5 | -5 [-11, +1] | -3 [-7, +0] | +1 [+0, +3] | 187, -2 [-7, +3], 2 / 4 | 3 / 0 / 2 | 3 / 0 / 0 / 0 | -2 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +10 [+2, +18]; vs conf.rand.m1 +11 [+4, +19] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +5 on 172 questions; mean words 4.9 -> 4.6
KEY family=E judge=allenai layer=27 A=incomplete B=not_met dR=+6 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+10 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 58 / 4 / 72 | 4.7 | +4 [-1, +10] | +1 [-3, +5] | -2 [-6, +2] | 113, +3 [-1, +8], 4 / 1 | 2 / 3 / 1 | 1 / 1 / 2 / 1 | +2 (105) |
| x.w.m1 | 54 / 59 / 4 / 75 | 4.8 | +0 [-4, +4] | -2 [-6, +2] | +1 [-2, +5] | 112, +1 [-2, +4], 2 / 1 | 3 / 1 / 1 | 1 / 2 / 0 / 1 | +1 (103) |
| x.md.p1 | 53 / 61 / 3 / 75 | 4.5 | -1 [-6, +3] | -1 [-5, +3] | +1 [-2, +4] | 112, +1 [-2, +4], 2 / 1 | 3 / 2 / 1 | 2 / 1 / 0 / 2 | +1 (104) |
| x.rand.p1 | 52 / 61 / 5 / 74 | 4.8 | -2 [-5, +0] | -2 [-6, +2] | +0 [-3, +3] | 112, +0 [+0, +0], 0 / 0 | 3 / 1 / 2 | 2 / 1 / 0 / 1 | +0 (105) |
| x.rand.m1 | 54 / 60 / 3 / 75 | 4.7 | +0 [-3, +3] | -1 [-3, +0] | +1 [+0, +3] | 114, +0 [-3, +3], 1 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +0 (107) |

- F, complete-sentence background (base x_F): x_F 61 / 70 / 5 / 56 -> with the E write 64 / 67 / 5 / 56; dR +3 [+0, +7]; answered-rate change +0.0 pp [-1.5, +1.5]; out / in 0 / 0; mean words 7.9 -> 7.8
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +6 [+1, +12]; vs x.rand.m1 +4 [-1, +10] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.5 pp [-1.6, +2.6]; moved out / in 2 / 3 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +2 on 105 questions; mean words 4.8 -> 4.7
KEY family=E judge=pro layer=27 A=not_met B=not_met dR=+4 dR_F=+3 dAns=+1 dAns_F=+0 rand_margin=+4

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 27

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 42 / 52 / 1 / 97 | 43 / 50 / 3 / 96 | +1 [-3, +6] | -1 [-6, +3] | -1 [-5, +2] | 91, +1, 1 / 0 | 3 / 2 / 2 | +1 (85) |
| caut_E2.wq.m1 | 45 / 40 / 0 / 107 | 42 / 39 / 1 / 110 | -3 [-9, +3] | -4 [-10, +1] | +3 [-2, +8] | 79, +1, 2 / 1 | 6 / 2 / 1 | +1 (74) |
| conf.iso.p1 | 87 / 97 / 5 / 3 | 86 / 97 / 6 / 3 | -1 [-4, +2] | -1 [-3, +0] | +0 [+0, +0] | 183, +0, 1 / 1 | 1 / 0 / 1 | +0 (181) |
| conf.md.p1 | 87 / 97 / 5 / 3 | 84 / 98 / 7 / 3 | -3 [-12, +6] | -2 [-6, +2] | +0 [+0, +0] | 181, -2, 8 / 10 | 3 / 1 / 3 | -3 (173) |
| conf.rand.m1 | 87 / 97 / 5 / 3 | 88 / 96 / 5 / 3 | +1 [-4, +6] | +0 [+0, +0] | +0 [+0, +0] | 184, +1, 4 / 3 | 0 / 0 / 0 | +1 (180) |
| conf.rand.p1 | 87 / 97 / 5 / 3 | 87 / 96 / 6 / 3 | +0 [-6, +6] | -1 [-3, +0] | +0 [+0, +0] | 183, +1, 5 / 4 | 1 / 0 / 1 | +1 (176) |
| conf.w.p1 | 87 / 97 / 5 / 3 | 89 / 93 / 7 / 3 | +2 [-6, +10] | -2 [-5, +0] | +0 [+0, +0] | 182, +3, 9 / 6 | 2 / 0 / 2 | +2 (166) |
| conf.wq.p1 | 87 / 97 / 5 / 3 | 91 / 92 / 6 / 3 | +4 [-4, +12] | -1 [-5, +2] | +0 [+0, +0] | 182, +4, 9 / 5 | 2 / 1 / 2 | +3 (168) |
| conf_E2.wq.m1 | 103 / 80 / 6 / 3 | 90 / 91 / 7 / 4 | -13 [-21, -5] | -2 [-6, +2] | +1 [+0, +3] | 180, -10, 3 / 13 | 3 / 1 / 2 | -5 (161) |
| x_F.w.p1 | 61 / 70 / 5 / 56 | 64 / 67 / 5 / 56 | +3 [+0, +7] | +0 [+0, +0] | +0 [+0, +0] | 131, +3, 3 / 0 | 0 / 0 / 0 | +2 (123) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 87 / 97 / 5 / 3; rule abstentions 0; mean words 4.9

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 89 / 93 / 7 / 3 | 4.6 | +2 [-6, +10] | -2 [-5, +0] | +0 [+0, +0] | 182, +3 [-4, +11], 9 / 6 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | +2 (166) |
| conf.md.p1 | 84 / 98 / 7 / 3 | 4.3 | -3 [-12, +6] | -2 [-6, +2] | +0 [+0, +0] | 181, -2 [-10, +6], 8 / 10 | 3 / 1 / 3 | 2 / 1 / 1 / 0 | -3 (173) |
| conf.rand.p1 | 87 / 96 / 6 / 3 | 4.8 | +0 [-6, +6] | -1 [-3, +0] | +0 [+0, +0] | 183, +1 [-5, +7], 5 / 4 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +1 (176) |
| conf.rand.m1 | 88 / 96 / 5 / 3 | 4.5 | +1 [-4, +6] | +0 [+0, +0] | +0 [+0, +0] | 184, +1 [-4, +6], 4 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +2 (1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +2 [-5, +9]; vs conf.rand.m1 +1 [-6, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +2 on 166 questions; mean words 4.9 -> 4.6
KEY family=E judge=pro layer=27 A=not_met B=not_met dR=+2 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+1 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 15:48:50 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 60 / 42 / 1 / 89 | 4.6 | -1 [-9, +7] | -15 [-24, -7] | +16 [+8, +24] | 100, +5 [+0, +10], 6 / 1 | 17 / 2 / 0 | 8 / 9 / 2 / 0 | +2 (88) |
| x.w.m1 | 67 / 60 / 2 / 63 | 4.8 | +6 [-1, +14] | +10 [+4, +16] | -10 [-16, -4] | 117, +2 [-4, +8], 6 / 4 | 0 / 10 / 0 | 0 / 0 / 4 / 6 | +3 (106) |
| x.md.p1 | 61 / 60 / 1 / 70 | 4.8 | +0 [-6, +6] | +4 [+0, +9] | -3 [-8, +1] | 116, -1 [-6, +4], 3 / 4 | 1 / 5 / 0 | 1 / 0 / 2 / 3 | +1 (102) |
| x.rand.p1 | 60 / 59 / 2 / 71 | 4.6 | -1 [-5, +2] | +2 [+0, +5] | -2 [-5, +0] | 117, -1 [-5, +2], 1 / 2 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -2 (113) |
| x.rand.m1 | 60 / 52 / 2 / 78 | 4.5 | -1 [-6, +4] | -5 [-10, +0] | +5 [+0, +10] | 111, +1 [-3, +5], 3 / 2 | 6 / 1 / 0 | 2 / 4 / 0 / 1 | +2 (104) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 62 / 54 / 0 / 76; dR -9 [-18, -1]; answered-rate change -9.9 pp [-14.6, -5.2]; out / in 21 / 2; mean words 7.9 -> 7.0
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +0 [-8, +8]; vs x.rand.m1 +0 [-8, +8] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -7.8 pp [-12.5, -3.6]; moved out / in 17 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +2 on 88 questions; mean words 4.8 -> 4.6
KEY family=E judge=allenai layer=12 A=not_met B=not_met dR=-1 dR_F=-9 dAns=-15 dAns_F=-19 rand_margin=+0

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 46 / 35 / 0 / 111 | +2 [-6, +10] | -12 [-20, -5] | +14 [+7, +21] | 79, +4, 7 / 3 | 14 / 2 / 0 | +3 (72) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 56 / 49 / 0 / 87 | +11 [+3, +19] | +21 [+13, +30] | -21 [-30, -13] | 84, -1, 3 / 4 | 0 / 21 / 0 | +0 (79) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 97 / 92 / 2 / 1 | -3 [-7, +1] | -1 [-3, +0] | +1 [+0, +3] | 189, -2, 1 / 3 | 1 / 0 / 0 | -1 (185) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 95 / 94 / 3 / 0 | -5 [-11, +0] | -1 [-5, +2] | +0 [+0, +0] | 188, -6, 0 / 6 | 2 / 1 / 2 | -5 (173) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 97 / 92 / 3 / 0 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 1 / 3 | 1 / 0 / 1 | -2 (179) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 94 / 94 / 3 / 1 | -6 [-12, +0] | -2 [-5, +0] | +1 [+0, +3] | 188, -4, 2 / 6 | 2 / 0 / 1 | -4 (181) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 105 / 84 / 2 / 1 | +5 [-3, +14] | -1 [-5, +2] | +1 [+0, +3] | 188, +5, 11 / 6 | 2 / 1 / 1 | +1 (165) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 105 / 85 / 2 / 0 | +5 [-3, +13] | +0 [-3, +3] | +0 [+0, +0] | 189, +4, 10 / 6 | 1 / 1 / 1 | +0 (166) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 96 / 92 / 3 / 1 | -17 [-28, -6] | +2 [+0, +5] | +0 [+0, +0] | 186, -19, 5 / 24 | 0 / 2 / 0 | -17 (158) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 62 / 54 / 0 / 76 | -9 [-18, -1] | -19 [-28, -10] | +20 [+11, +29] | 114, +3, 5 / 2 | 21 / 2 / 0 | +2 (101) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 105 / 84 / 2 / 1 | 5.0 | +5 [-3, +14] | -1 [-5, +2] | +1 [+0, +3] | 188, +5 [-3, +13], 11 / 6 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +1 (165) |
| conf.md.p1 | 95 / 94 / 3 / 0 | 5.0 | -5 [-11, +0] | -1 [-5, +2] | +0 [+0, +0] | 188, -6 [-11, -2], 0 / 6 | 2 / 1 / 2 | 0 / 2 / 1 / 0 | -5 (173) |
| conf.rand.p1 | 94 / 94 / 3 / 1 | 4.6 | -6 [-12, +0] | -2 [-5, +0] | +1 [+0, +3] | 188, -4 [-10, +1], 2 / 6 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | -4 (181) |
| conf.rand.m1 | 97 / 92 / 3 / 0 | 4.7 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-6, +2], 1 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -2 (179) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +11 [+2, +20]; vs conf.rand.m1 +8 [-1, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 165 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=12 A=not_met B=not_met dR=+5 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+8 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:33:26 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

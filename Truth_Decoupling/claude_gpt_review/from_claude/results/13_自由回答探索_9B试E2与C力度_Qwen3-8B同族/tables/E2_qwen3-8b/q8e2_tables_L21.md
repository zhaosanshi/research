TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 21

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 39 / 2 / 93 | 4.4 | -3 [-12, +7] | -20 [-29, -11] | +20 [+12, +29] | 95, +4 [-3, +11], 8 / 4 | 22 / 2 / 1 | 9 / 13 / 2 / 0 | +1 (84) |
| x.w.m1 | 65 / 71 / 2 / 54 | 5.4 | +4 [-3, +11] | +19 [+11, +28] | -19 [-28, -11] | 117, +2 [-4, +8], 6 / 4 | 0 / 19 / 0 | 0 / 0 / 2 / 17 | +4 (100) |
| x.md.p1 | 60 / 64 / 2 / 66 | 4.4 | -1 [-8, +6] | +7 [+1, +14] | -7 [-14, -1] | 115, -4 [-10, +1], 2 / 6 | 2 / 9 / 0 | 1 / 1 / 4 / 5 | +0 (103) |
| x.rand.p1 | 58 / 59 / 3 / 72 | 4.8 | -3 [-8, +2] | +0 [-4, +4] | -1 [-5, +2] | 115, -3 [-8, +1], 1 / 4 | 2 / 2 / 1 | 1 / 1 / 1 / 1 | -1 (105) |
| x.rand.m1 | 62 / 52 / 3 / 75 | 4.5 | +1 [-2, +5] | -3 [-8, +1] | +2 [-2, +6] | 113, +2 [+0, +5], 2 / 0 | 4 / 1 / 1 | 1 / 3 / 0 / 1 | +1 (108) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 63 / 51 / 1 / 77; dR -8 [-16, +0]; answered-rate change -10.9 pp [-15.6, -6.8]; out / in 21 / 0; mean words 7.9 -> 7.0
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +0 [-9, +9]; vs x.rand.m1 -4 [-14, +6] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -10.4 pp [-15.1, -5.7]; moved out / in 22 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +1 on 84 questions; mean words 4.8 -> 4.4
KEY family=E judge=allenai layer=21 A=not_met B=not_met dR=-3 dR_F=-8 dAns=-20 dAns_F=-21 rand_margin=-4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 21

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 41 / 31 / 0 / 120 | -3 [-12, +6] | -21 [-30, -12] | +23 [+15, +32] | 71, +6, 9 / 3 | 22 / 1 / 0 | +2 (66) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 54 / 51 / 4 / 83 | +9 [+1, +18] | +21 [+12, +30] | -25 [-35, -16] | 83, +1, 5 / 4 | 1 / 22 / 0 | +0 (76) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -1, 0 / 1 | 0 / 0 / 0 | -1 (187) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 97 / 91 / 3 / 1 | -3 [-11, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, -1, 8 / 9 | 2 / 0 / 1 | -2 (168) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 95 / 91 / 6 / 0 | -5 [-12, +2] | -4 [-8, -1] | +0 [+0, +0] | 186, -3, 4 / 7 | 4 / 0 / 4 | -4 (172) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-6, +4] | +0 [+0, +0] | +0 [+0, +0] | 190, -1, 3 / 4 | 0 / 0 / 0 | +0 (180) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 107 / 81 / 3 / 1 | +7 [-4, +18] | -2 [-7, +3] | +1 [+0, +3] | 186, +7, 18 / 11 | 4 / 2 / 3 | +3 (159) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 108 / 80 / 3 / 1 | +8 [-3, +20] | -2 [-7, +3] | +1 [+0, +3] | 186, +8, 19 / 11 | 4 / 2 / 3 | +5 (161) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 102 / 87 / 3 / 0 | -11 [-22, +0] | +3 [+0, +7] | -1 [-3, +0] | 186, -14, 8 / 22 | 0 / 3 / 0 | -10 (161) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 63 / 51 / 1 / 77 | -8 [-16, +0] | -21 [-30, -13] | +21 [+13, +30] | 114, +1, 4 / 3 | 21 / 0 / 0 | +1 (99) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 21

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 107 / 81 / 3 / 1 | 4.7 | +7 [-4, +18] | -2 [-7, +3] | +1 [+0, +3] | 186, +7 [-3, +18], 18 / 11 | 4 / 2 / 3 | 2 / 2 / 2 / 0 | +3 (159) |
| conf.md.p1 | 97 / 91 / 3 / 1 | 4.4 | -3 [-11, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, -1 [-9, +7], 8 / 9 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | -2 (168) |
| conf.rand.p1 | 99 / 91 / 2 / 0 | 4.8 | -1 [-6, +4] | +0 [+0, +0] | +0 [+0, +0] | 190, -1 [-6, +4], 3 / 4 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (180) |
| conf.rand.m1 | 95 / 91 / 6 / 0 | 4.9 | -5 [-12, +2] | -4 [-8, -1] | +0 [+0, +0] | 186, -3 [-10, +3], 4 / 7 | 4 / 0 / 4 | 2 / 2 / 0 / 0 | -4 (172) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +8 [-3, +19]; vs conf.rand.m1 +12 [+2, +23] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-3.6, +1.6]; moved out / in 4 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 159 questions; mean words 4.9 -> 4.7
KEY family=E judge=allenai layer=21 A=incomplete B=not_met dR=+7 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+8 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 14:40:08 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

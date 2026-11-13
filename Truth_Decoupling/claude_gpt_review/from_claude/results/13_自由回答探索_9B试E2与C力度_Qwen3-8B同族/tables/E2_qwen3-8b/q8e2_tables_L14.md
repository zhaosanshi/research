TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 59 / 39 / 1 / 93 | 4.5 | -2 [-11, +7] | -19 [-28, -10] | +20 [+11, +29] | 95, +4 [-1, +10], 6 / 2 | 22 / 3 / 0 | 9 / 13 / 3 / 0 | +2 (82) |
| x.w.m1 | 73 / 63 / 2 / 54 | 5.1 | +12 [+4, +21] | +19 [+11, +28] | -19 [-28, -11] | 117, +4 [-2, +10], 7 / 3 | 0 / 19 / 0 | 0 / 0 / 8 / 11 | +5 (103) |
| x.md.p1 | 56 / 63 / 3 / 70 | 5.0 | -5 [-10, +0] | +2 [-5, +9] | -3 [-10, +3] | 112, -3 [-7, +0], 0 / 3 | 5 / 7 / 1 | 3 / 2 / 1 / 6 | -2 (101) |
| x.rand.p1 | 65 / 56 / 2 / 69 | 4.7 | +4 [+1, +8] | +4 [+0, +9] | -4 [-9, +0] | 116, +3 [+0, +7], 3 / 0 | 1 / 5 / 0 | 0 / 1 / 1 / 4 | +2 (106) |
| x.rand.m1 | 58 / 54 / 3 / 77 | 4.6 | -3 [-9, +3] | -5 [-10, -1] | +4 [+1, +8] | 112, +0 [-5, +5], 3 / 3 | 5 / 0 / 1 | 3 / 2 / 0 / 0 | -3 (105) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 66 / 50 / 1 / 75; dR -5 [-14, +4]; answered-rate change -9.9 pp [-14.6, -5.2]; out / in 21 / 2; mean words 7.9 -> 7.2
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -6 [-14, +2]; vs x.rand.m1 +1 [-8, +10] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -9.9 pp [-14.6, -5.2]; moved out / in 22 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +2 on 82 questions; mean words 4.8 -> 4.5
KEY family=E judge=allenai layer=14 A=not_met B=not_met dR=-2 dR_F=-5 dAns=-19 dAns_F=-19 rand_margin=-6

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 14

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 46 / 30 / 0 / 116 | +2 [-6, +10] | -17 [-26, -9] | +19 [+11, +27] | 74, +6, 8 / 2 | 19 / 2 / 0 | +3 (67) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 57 / 56 / 1 / 78 | +12 [+4, +21] | +29 [+20, +39] | -30 [-40, -21] | 84, -1, 3 / 4 | 0 / 29 / 0 | -1 (77) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 97 / 92 / 3 / 0 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -3, 1 / 4 | 1 / 0 / 1 | -2 (184) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 93 / 94 / 4 / 1 | -7 [-14, +0] | -3 [-7, +0] | +1 [+0, +3] | 187, -5, 3 / 8 | 3 / 0 / 2 | -5 (175) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-7, +3] | +0 [-3, +3] | +0 [+0, +0] | 189, -3, 1 / 4 | 1 / 1 / 1 | -2 (178) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 96 / 93 / 3 / 0 | -4 [-10, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -3, 3 / 6 | 1 / 0 / 1 | -4 (173) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 104 / 83 / 2 / 3 | +4 [-6, +14] | -3 [-8, +1] | +3 [+0, +7] | 186, +6, 13 / 7 | 4 / 1 / 1 | +2 (162) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 105 / 83 / 2 / 2 | +5 [-4, +14] | -2 [-6, +2] | +2 [+0, +5] | 187, +6, 13 / 7 | 3 / 1 / 1 | +2 (164) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 100 / 89 / 3 / 0 | -13 [-24, -3] | +3 [+0, +7] | -1 [-3, +0] | 186, -16, 5 / 21 | 0 / 3 / 0 | -13 (165) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 66 / 50 / 1 / 75 | -5 [-14, +4] | -19 [-28, -10] | +19 [+10, +28] | 114, +6, 7 / 1 | 21 / 2 / 0 | +3 (100) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 14

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 104 / 83 / 2 / 3 | 5.1 | +4 [-6, +14] | -3 [-8, +1] | +3 [+0, +7] | 186, +6 [-3, +15], 13 / 7 | 4 / 1 / 1 | 3 / 1 / 1 / 0 | +2 (162) |
| conf.md.p1 | 93 / 94 / 4 / 1 | 4.6 | -7 [-14, +0] | -3 [-7, +0] | +1 [+0, +3] | 187, -5 [-12, +1], 3 / 8 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | -5 (175) |
| conf.rand.p1 | 96 / 93 / 3 / 0 | 4.9 | -4 [-10, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -3 [-9, +3], 3 / 6 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -4 (173) |
| conf.rand.m1 | 98 / 92 / 2 / 0 | 4.8 | -2 [-7, +3] | +0 [-3, +3] | +0 [+0, +0] | 189, -3 [-8, +1], 1 / 4 | 1 / 1 / 1 | 0 / 1 / 1 / 0 | -2 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +8 [-2, +18]; vs conf.rand.m1 +6 [-3, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.2, +0.5]; moved out / in 4 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 162 questions; mean words 4.9 -> 5.1
KEY family=E judge=allenai layer=14 A=not_met B=not_met dR=+4 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+6 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:35:47 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

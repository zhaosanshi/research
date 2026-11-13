TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 19

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 49 / 34 / 2 / 107 | 4.1 | -12 [-21, -3] | -34 [-45, -24] | +34 [+24, +45] | 82, +2 [-3, +8], 5 / 3 | 35 / 1 / 1 | 15 / 20 / 1 / 0 | -1 (74) |
| x.w.m1 | 70 / 76 / 2 / 44 | 4.9 | +9 [+1, +18] | +29 [+20, +39] | -29 [-39, -20] | 117, +0 [-6, +6], 5 / 5 | 0 / 29 / 0 | 0 / 0 / 9 / 20 | +1 (101) |
| x.md.p1 | 54 / 49 / 1 / 88 | 4.4 | -7 [-14, +0] | -14 [-22, -7] | +15 [+8, +22] | 102, +1 [-3, +5], 3 / 2 | 15 / 1 / 0 | 9 / 6 / 1 / 0 | +1 (94) |
| x.rand.p1 | 65 / 53 / 2 / 72 | 4.6 | +4 [+0, +9] | +1 [-2, +5] | -1 [-5, +2] | 116, +5 [+1, +10], 5 / 0 | 1 / 2 / 0 | 1 / 0 / 0 / 2 | +4 (110) |
| x.rand.m1 | 59 / 56 / 2 / 75 | 4.6 | -2 [-7, +3] | -2 [-6, +2] | +2 [-2, +6] | 114, -1 [-4, +2], 1 / 2 | 3 / 1 / 0 | 2 / 1 / 1 / 0 | -2 (106) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 62 / 45 / 1 / 84; dR -9 [-17, -1]; answered-rate change -14.6 pp [-19.8, -9.9]; out / in 28 / 0; mean words 7.9 -> 6.9
- main effect: dR -12 (-6.2% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -16 [-26, -7]; vs x.rand.m1 -10 [-19, -1] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -17.7 pp [-23.4, -12.5]; moved out / in 35 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same -1 on 74 questions; mean words 4.8 -> 4.1
KEY family=E judge=allenai layer=19 A=not_met B=not_met dR=-12 dR_F=-9 dAns=-34 dAns_F=-28 rand_margin=-16

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 19

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 31 / 23 / 0 / 138 | -13 [-22, -4] | -39 [-50, -28] | +41 [+30, +52] | 54, +4, 4 / 0 | 39 / 0 / 0 | +1 (48) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 57 / 59 / 3 / 73 | +12 [+2, +22] | +32 [+22, +42] | -35 [-46, -25] | 84, -3, 4 / 7 | 0 / 32 / 0 | -1 (76) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -1, 1 / 2 | 0 / 0 / 0 | +0 (188) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 90 / 99 / 2 / 1 | -10 [-20, +0] | -1 [-5, +2] | +1 [+0, +3] | 188, -10, 6 / 16 | 2 / 1 / 1 | -6 (171) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 94 / 94 / 3 / 1 | -6 [-11, -2] | -2 [-5, +0] | +1 [+0, +3] | 188, -4, 0 / 4 | 2 / 0 / 1 | -4 (183) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 98 / 93 / 1 / 0 | -2 [-8, +4] | +1 [+0, +3] | +0 [+0, +0] | 190, -3, 3 / 6 | 0 / 1 / 0 | -2 (176) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 109 / 80 / 2 / 1 | +9 [-1, +20] | -1 [-5, +2] | +1 [+0, +3] | 188, +9, 18 / 9 | 2 / 1 / 1 | +4 (159) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 108 / 82 / 1 / 1 | +8 [-2, +19] | +0 [-4, +4] | +1 [+0, +3] | 188, +7, 17 / 10 | 2 / 2 / 1 | +3 (162) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 97 / 91 / 4 / 0 | -16 [-27, -5] | +2 [+0, +5] | -1 [-3, +0] | 186, -18, 7 / 25 | 0 / 2 / 0 | -11 (161) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 62 / 45 / 1 / 84 | -9 [-17, -1] | -28 [-38, -19] | +28 [+19, +38] | 107, +3, 4 / 1 | 28 / 0 / 0 | +2 (93) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 19

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 109 / 80 / 2 / 1 | 5.0 | +9 [-1, +20] | -1 [-5, +2] | +1 [+0, +3] | 188, +9 [-1, +19], 18 / 9 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +4 (159) |
| conf.md.p1 | 90 / 99 / 2 / 1 | 4.2 | -10 [-20, +0] | -1 [-5, +2] | +1 [+0, +3] | 188, -10 [-19, -1], 6 / 16 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -6 (171) |
| conf.rand.p1 | 98 / 93 / 1 / 0 | 5.0 | -2 [-8, +4] | +1 [+0, +3] | +0 [+0, +0] | 190, -3 [-9, +3], 3 / 6 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | -2 (176) |
| conf.rand.m1 | 94 / 94 / 3 / 1 | 4.6 | -6 [-11, -2] | -2 [-5, +0] | +1 [+0, +3] | 188, -4 [-8, -1], 0 / 4 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | -4 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +9 (4.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +11 [+2, +21]; vs conf.rand.m1 +15 [+5, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +4 on 159 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=19 A=incomplete B=not_met dR=+9 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+11 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 14:37:41 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

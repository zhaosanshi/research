TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 2

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 59 / 57 / 2 / 74 | 4.7 | -2 [-8, +3] | -1 [-3, +0] | +1 [+0, +3] | 116, -2 [-8, +3], 3 / 5 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -3 (107) |
| x.w.m1 | 66 / 53 / 2 / 71 | 4.6 | +5 [+0, +10] | +2 [-2, +6] | -2 [-6, +2] | 116, +5 [+0, +10], 6 / 1 | 1 / 3 / 0 | 0 / 1 / 0 / 3 | +4 (106) |
| x.md.p1 | 59 / 57 / 2 / 74 | 4.7 | -2 [-5, +0] | -1 [-3, +0] | +1 [+0, +3] | 116, -2 [-5, +0], 0 / 2 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -2 (110) |
| x.rand.p1 | 60 / 57 / 2 / 73 | 4.7 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 117, -1 [-5, +2], 1 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (116) |
| x.rand.m1 | 61 / 56 / 2 / 73 | 4.8 | +0 [-4, +4] | +0 [-3, +3] | +0 [-3, +3] | 116, +0 [-4, +4], 2 / 2 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | +0 (111) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 70 / 61 / 2 / 59; dR -1 [-7, +5]; answered-rate change -2.1 pp [-5.2, +1.0]; out / in 7 / 3; mean words 7.9 -> 7.6
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -1 [-7, +5]; vs x.rand.m1 -2 [-8, +4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -3 on 107 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=2 A=not_met B=not_met dR=-2 dR_F=-1 dAns=-1 dAns_F=-4 rand_margin=-2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 2

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 44 / 46 / 1 / 101 | +0 [-4, +4] | -3 [-9, +3] | +4 [-1, +10] | 87, +1, 1 / 0 | 6 / 3 / 0 | +0 (84) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 50 / 39 / 0 / 103 | +5 [+0, +10] | +5 [-1, +11] | -5 [-11, +1] | 82, +2, 2 / 0 | 2 / 7 / 0 | +1 (76) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 89 / 4 / 1 | -2 [-7, +3] | -3 [-7, +0] | +1 [+0, +3] | 187, +0, 2 / 2 | 3 / 0 / 2 | +0 (185) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 96 / 92 / 4 / 0 | -4 [-10, +1] | -2 [-5, +0] | +0 [+0, +0] | 188, -3, 2 / 5 | 2 / 0 / 2 | -2 (181) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 101 / 89 / 2 / 0 | +1 [-3, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, +1, 3 / 2 | 0 / 0 / 0 | +2 (182) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 92 / 95 / 5 / 0 | -8 [-14, -3] | -3 [-7, +0] | +0 [+0, +0] | 187, -6, 0 / 6 | 3 / 0 / 3 | -5 (181) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 95 / 89 / 6 / 2 | -5 [-13, +4] | -6 [-11, -2] | +2 [+0, +5] | 184, +0, 7 / 7 | 6 / 0 / 4 | +2 (172) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 96 / 90 / 4 / 2 | -4 [-12, +4] | -4 [-8, -1] | +2 [+0, +5] | 186, -1, 6 / 7 | 4 / 0 / 2 | +0 (171) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 102 / 85 / 5 / 0 | -11 [-19, -4] | +1 [-2, +4] | -1 [-3, +0] | 185, -11, 1 / 12 | 1 / 2 / 1 | -10 (169) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 70 / 61 / 2 / 59 | -1 [-7, +5] | -4 [-10, +2] | +3 [-3, +9] | 128, +4, 4 / 0 | 7 / 3 / 1 | +4 (122) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 2

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 95 / 89 / 6 / 2 | 4.8 | -5 [-13, +4] | -6 [-11, -2] | +2 [+0, +5] | 184, +0 [-7, +7], 7 / 7 | 6 / 0 / 4 | 5 / 1 / 0 / 0 | +2 (172) |
| conf.md.p1 | 96 / 92 / 4 / 0 | 4.8 | -4 [-10, +1] | -2 [-5, +0] | +0 [+0, +0] | 188, -3 [-8, +2], 2 / 5 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | -2 (181) |
| conf.rand.p1 | 92 / 95 / 5 / 0 | 4.9 | -8 [-14, -3] | -3 [-7, +0] | +0 [+0, +0] | 187, -6 [-11, -2], 0 / 6 | 3 / 0 / 3 | 2 / 1 / 0 / 0 | -5 (181) |
| conf.rand.m1 | 101 / 89 / 2 / 0 | 4.8 | +1 [-3, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, +1 [-3, +5], 3 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +2 (182) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -5 (-2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +3 [-5, +11]; vs conf.rand.m1 -6 [-14, +2] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -3.1 pp [-5.7, -1.0]; moved out / in 6 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +2 on 172 questions; mean words 4.9 -> 4.8
KEY family=E judge=allenai layer=2 A=not_met B=not_met dR=-5 dR_F=NA dAns=-6 dAns_F=NA rand_margin=-6 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:43:31 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

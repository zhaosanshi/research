TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 59 / 45 / 1 / 87 | 4.6 | -2 [-8, +4] | -13 [-21, -6] | +14 [+7, +21] | 103, +1 [-3, +5], 3 / 2 | 14 / 1 / 0 | 4 / 10 / 1 / 0 | +0 (93) |
| x.w.m1 | 59 / 62 / 3 / 68 | 4.5 | -2 [-7, +3] | +4 [+0, +9] | -5 [-10, -1] | 116, -2 [-7, +3], 2 / 4 | 1 / 5 / 1 | 0 / 1 / 0 / 5 | +0 (106) |
| x.md.p1 | 63 / 56 / 2 / 71 | 4.8 | +2 [-3, +7] | +2 [-2, +6] | -2 [-6, +2] | 116, +2 [-3, +7], 4 / 2 | 1 / 3 / 0 | 0 / 1 / 0 / 3 | +2 (103) |
| x.rand.p1 | 61 / 54 / 2 / 75 | 4.4 | +0 [-5, +5] | -2 [-5, +0] | +2 [+0, +5] | 115, +1 [-3, +5], 3 / 2 | 2 / 0 / 0 | 1 / 1 / 0 / 0 | +1 (109) |
| x.rand.m1 | 62 / 58 / 2 / 70 | 4.8 | +1 [-4, +6] | +3 [+0, +7] | -3 [-7, +0] | 117, +0 [-5, +5], 3 / 3 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +0 (107) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 64 / 61 / 1 / 66; dR -7 [-14, +0]; answered-rate change -5.2 pp [-9.4, -1.6]; out / in 12 / 2; mean words 7.9 -> 7.5
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -2 [-8, +4]; vs x.rand.m1 -3 [-10, +4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -6.8 pp [-10.9, -3.1]; moved out / in 14 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +0 on 93 questions; mean words 4.8 -> 4.6
KEY family=E judge=allenai layer=10 A=not_met B=not_met dR=-2 dR_F=-7 dAns=-13 dAns_F=-10 rand_margin=-3

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 10

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 43 / 43 / 0 / 106 | -1 [-7, +5] | -7 [-14, +0] | +9 [+3, +16] | 83, +1, 2 / 1 | 10 / 3 / 0 | +1 (78) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 49 / 47 / 0 / 96 | +4 [-2, +10] | +12 [+6, +19] | -12 [-19, -6] | 84, -2, 1 / 3 | 0 / 12 / 0 | -1 (77) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 101 / 88 / 2 / 1 | +1 [-3, +5] | -1 [-3, +0] | +1 [+0, +3] | 189, +2, 3 / 1 | 1 / 0 / 0 | +2 (187) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 93 / 94 / 4 / 1 | -7 [-13, -1] | -3 [-7, +0] | +1 [+0, +3] | 187, -4, 2 / 6 | 3 / 0 / 2 | -3 (172) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-6, +4] | +0 [-3, +3] | +0 [+0, +0] | 189, -2, 2 / 4 | 1 / 1 / 1 | -2 (177) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-8, +4] | +0 [+0, +0] | +0 [+0, +0] | 190, -2, 4 / 6 | 0 / 0 / 0 | -2 (183) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 101 / 88 / 2 / 1 | +1 [-7, +9] | -1 [-5, +2] | +1 [+0, +3] | 188, +1, 8 / 7 | 2 / 1 / 1 | +1 (170) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 102 / 86 / 3 / 1 | +2 [-6, +10] | -2 [-6, +2] | +1 [+0, +3] | 187, +3, 9 / 6 | 3 / 1 / 2 | +3 (170) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 96 / 92 / 3 / 1 | -17 [-27, -7] | +2 [+0, +5] | +0 [+0, +0] | 186, -19, 3 / 22 | 0 / 2 / 0 | -14 (165) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 64 / 61 / 1 / 66 | -7 [-14, +0] | -10 [-18, -3] | +10 [+3, +18] | 123, +1, 3 / 2 | 12 / 2 / 0 | +0 (112) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 101 / 88 / 2 / 1 | 5.1 | +1 [-7, +9] | -1 [-5, +2] | +1 [+0, +3] | 188, +1 [-7, +9], 8 / 7 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +1 (170) |
| conf.md.p1 | 93 / 94 / 4 / 1 | 4.8 | -7 [-13, -1] | -3 [-7, +0] | +1 [+0, +3] | 187, -4 [-10, +1], 2 / 6 | 3 / 0 / 2 | 3 / 0 / 0 / 0 | -3 (172) |
| conf.rand.p1 | 98 / 92 / 2 / 0 | 4.7 | -2 [-8, +4] | +0 [+0, +0] | +0 [+0, +0] | 190, -2 [-8, +4], 4 / 6 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -2 (183) |
| conf.rand.m1 | 99 / 91 / 2 / 0 | 4.8 | -1 [-6, +4] | +0 [-3, +3] | +0 [+0, +0] | 189, -2 [-7, +3], 2 / 4 | 1 / 1 / 1 | 0 / 1 / 1 / 0 | -2 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +3 [-6, +12]; vs conf.rand.m1 +2 [-6, +10] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 170 questions; mean words 4.9 -> 5.1
KEY family=E judge=allenai layer=10 A=not_met B=not_met dR=+1 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+2 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:31:09 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

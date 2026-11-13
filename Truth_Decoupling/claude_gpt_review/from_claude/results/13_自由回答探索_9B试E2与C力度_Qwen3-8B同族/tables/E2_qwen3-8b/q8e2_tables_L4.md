TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 4

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 58 / 1 / 75 | 4.5 | -3 [-9, +3] | -1 [-5, +3] | +2 [-2, +6] | 114, -4 [-9, +0], 1 / 5 | 3 / 2 / 0 | 1 / 2 / 2 / 0 | -3 (112) |
| x.w.m1 | 64 / 53 / 3 / 72 | 4.6 | +3 [-3, +9] | +0 [-5, +5] | -1 [-5, +3] | 114, +4 [-1, +10], 6 / 2 | 3 / 3 / 1 | 1 / 2 / 0 / 3 | +2 (101) |
| x.md.p1 | 56 / 53 / 2 / 81 | 4.5 | -5 [-12, +1] | -8 [-14, -3] | +8 [+3, +14] | 109, +0 [-5, +5], 3 / 3 | 8 / 0 / 0 | 5 / 3 / 0 / 0 | -1 (101) |
| x.rand.p1 | 62 / 58 / 2 / 70 | 4.7 | +1 [-3, +5] | +3 [+0, +7] | -3 [-7, +0] | 117, +1 [-3, +5], 3 / 2 | 0 / 3 / 0 | 0 / 0 / 0 / 3 | +0 (112) |
| x.rand.m1 | 62 / 55 / 2 / 73 | 4.7 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 117, +1 [-2, +5], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (111) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 71 / 57 / 3 / 61; dR +0 [-6, +6]; answered-rate change -3.6 pp [-6.8, -1.0]; out / in 8 / 1; mean words 7.9 -> 7.7
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -4 [-10, +1]; vs x.rand.m1 -4 [-11, +2] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-2.6, +1.6]; moved out / in 3 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -3 on 112 questions; mean words 4.8 -> 4.5
KEY family=E judge=allenai layer=4 A=not_met B=not_met dR=-3 dR_F=+0 dAns=-1 dAns_F=-7 rand_margin=-4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 4

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 46 / 42 / 0 / 104 | +2 [-3, +7] | -5 [-11, +1] | +7 [+2, +12] | 86, +2, 3 / 1 | 7 / 2 / 0 | +2 (83) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 48 / 43 / 0 / 101 | +3 [-2, +8] | +7 [+2, +12] | -7 [-12, -2] | 84, -1, 1 / 2 | 0 / 7 / 0 | -1 (74) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 90 / 3 / 1 | -2 [-6, +2] | -2 [-5, +0] | +1 [+0, +3] | 188, -1, 1 / 2 | 2 / 0 / 1 | -1 (187) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 97 / 89 / 4 / 2 | -3 [-10, +4] | -4 [-8, -1] | +2 [+0, +5] | 186, +0, 4 / 4 | 4 / 0 / 2 | +0 (170) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 99 / 90 / 3 / 0 | -1 [-4, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, +0, 1 / 1 | 1 / 0 / 1 | +0 (179) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 92 / 94 / 4 / 2 | -8 [-15, -1] | -4 [-8, -1] | +2 [+0, +5] | 186, -5, 2 / 7 | 4 / 0 / 2 | -3 (182) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-7, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 7 / 6 | 2 / 0 / 1 | +2 (171) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-7, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 6 / 5 | 2 / 0 / 1 | +2 (174) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 103 / 84 / 4 / 1 | -10 [-18, -2] | +1 [+0, +3] | +0 [+0, +0] | 186, -11, 2 / 13 | 0 / 1 / 0 | -11 (169) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 71 / 57 / 3 / 61 | +0 [-6, +6] | -7 [-13, -2] | +5 [+0, +10] | 127, +5, 5 / 0 | 8 / 1 / 2 | +3 (115) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 4

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 88 / 3 / 1 | 5.0 | +0 [-7, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-6, +8], 7 / 6 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +2 (171) |
| conf.md.p1 | 97 / 89 / 4 / 2 | 5.0 | -3 [-10, +4] | -4 [-8, -1] | +2 [+0, +5] | 186, +0 [-6, +6], 4 / 4 | 4 / 0 / 2 | 3 / 1 / 0 / 0 | +0 (170) |
| conf.rand.p1 | 92 / 94 / 4 / 2 | 4.6 | -8 [-15, -1] | -4 [-8, -1] | +2 [+0, +5] | 186, -5 [-11, +0], 2 / 7 | 4 / 0 / 2 | 3 / 1 / 0 / 0 | -3 (182) |
| conf.rand.m1 | 99 / 90 / 3 / 0 | 4.9 | -1 [-4, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, +0 [-3, +3], 1 / 1 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (179) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +8 [+2, +15]; vs conf.rand.m1 +1 [-7, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +2 on 171 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=4 A=not_met B=not_met dR=+0 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+1 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:45:40 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

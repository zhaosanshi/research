TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 7

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 53 / 2 / 79 | 4.6 | -3 [-9, +3] | -6 [-12, +0] | +6 [+0, +12] | 109, -2 [-7, +3], 2 / 4 | 8 / 2 / 0 | 3 / 5 / 2 / 0 | -2 (103) |
| x.w.m1 | 63 / 57 / 3 / 69 | 4.5 | +2 [-3, +8] | +3 [-2, +8] | -4 [-9, +0] | 115, +1 [-3, +5], 3 / 2 | 2 / 5 / 1 | 1 / 1 / 2 / 3 | +1 (106) |
| x.md.p1 | 62 / 57 / 2 / 71 | 4.8 | +1 [-4, +6] | +2 [-2, +6] | -2 [-6, +2] | 116, +1 [-4, +6], 4 / 3 | 1 / 3 / 0 | 0 / 1 / 0 / 3 | +2 (108) |
| x.rand.p1 | 61 / 56 / 2 / 73 | 4.6 | +0 [-4, +4] | +0 [-3, +3] | +0 [-3, +3] | 116, -1 [-5, +2], 1 / 2 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | -1 (115) |
| x.rand.m1 | 63 / 56 / 2 / 71 | 4.8 | +2 [-2, +6] | +2 [+0, +5] | -2 [-5, +0] | 117, +2 [-2, +6], 3 / 1 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | +1 (111) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 66 / 57 / 1 / 68; dR -5 [-11, +1]; answered-rate change -6.2 pp [-9.9, -2.6]; out / in 13 / 1; mean words 7.9 -> 7.4
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -3 [-9, +3]; vs x.rand.m1 -5 [-12, +1] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.1 pp [-6.2, +0.0]; moved out / in 8 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same -2 on 103 questions; mean words 4.8 -> 4.6
KEY family=E judge=allenai layer=7 A=not_met B=not_met dR=-3 dR_F=-5 dAns=-6 dAns_F=-12 rand_margin=-5

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 7

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 47 / 42 / 1 / 102 | +3 [-3, +9] | -4 [-10, +1] | +5 [+0, +10] | 87, +3, 4 / 1 | 6 / 2 / 0 | +3 (84) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 49 / 43 / 0 / 100 | +4 [-1, +10] | +8 [+2, +14] | -8 [-14, -2] | 83, +1, 2 / 1 | 1 / 9 / 0 | +0 (76) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-5, +3] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 2 / 2 | 2 / 0 / 1 | +0 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 95 / 92 / 5 / 0 | -5 [-10, +0] | -3 [-7, +0] | +0 [+0, +0] | 187, -4, 1 / 5 | 3 / 0 / 3 | -3 (169) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 96 / 93 / 3 / 0 | -4 [-9, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -4, 1 / 5 | 1 / 0 / 1 | -3 (181) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 95 / 91 / 5 / 1 | -5 [-10, +0] | -4 [-8, -1] | +1 [+0, +3] | 186, -3, 1 / 4 | 4 / 0 / 3 | -2 (181) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-7, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 7 / 6 | 2 / 0 / 1 | +0 (173) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-7, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 7 / 6 | 2 / 0 / 1 | +0 (173) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 104 / 84 / 3 / 1 | -9 [-18, +0] | +2 [+0, +5] | +0 [+0, +0] | 186, -11, 4 / 15 | 0 / 2 / 0 | -11 (170) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 66 / 57 / 1 / 68 | -5 [-11, +1] | -12 [-19, -5] | +12 [+5, +19] | 122, +2, 2 / 0 | 13 / 1 / 0 | +1 (114) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 7

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 88 / 3 / 1 | 4.9 | +0 [-7, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-6, +8], 7 / 6 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +0 (173) |
| conf.md.p1 | 95 / 92 / 5 / 0 | 5.2 | -5 [-10, +0] | -3 [-7, +0] | +0 [+0, +0] | 187, -4 [-9, +0], 1 / 5 | 3 / 0 / 3 | 1 / 2 / 0 / 0 | -3 (169) |
| conf.rand.p1 | 95 / 91 / 5 / 1 | 4.9 | -5 [-10, +0] | -4 [-8, -1] | +1 [+0, +3] | 186, -3 [-8, +1], 1 / 4 | 4 / 0 / 3 | 2 / 2 / 0 / 0 | -2 (181) |
| conf.rand.m1 | 96 / 93 / 3 / 0 | 4.7 | -4 [-9, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -4 [-9, +0], 1 / 5 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -3 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +5 [-2, +12]; vs conf.rand.m1 +4 [-2, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +0 on 173 questions; mean words 4.9 -> 4.9
KEY family=E judge=allenai layer=7 A=not_met B=not_met dR=+0 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+4 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:48:55 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

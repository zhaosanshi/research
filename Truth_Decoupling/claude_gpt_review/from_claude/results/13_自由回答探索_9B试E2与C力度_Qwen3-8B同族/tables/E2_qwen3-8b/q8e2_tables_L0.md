TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 0

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 59 / 57 / 2 / 74 | 4.5 | -2 [-7, +2] | -1 [-5, +2] | +1 [-2, +5] | 115, -1 [-6, +3], 2 / 3 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | -3 (111) |
| x.w.m1 | 63 / 54 / 2 / 73 | 4.6 | +2 [-3, +7] | +0 [-3, +3] | +0 [-3, +3] | 116, +2 [-3, +7], 4 / 2 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | +2 (109) |
| x.md.p1 | 62 / 55 / 2 / 73 | 4.5 | +1 [-2, +5] | +0 [-3, +3] | +0 [-3, +3] | 116, +1 [-2, +5], 2 / 1 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | +1 (112) |
| x.rand.p1 | 60 / 56 / 2 / 74 | 4.6 | -1 [-5, +2] | -1 [-3, +0] | +1 [+0, +3] | 116, +0 [-3, +3], 1 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (113) |
| x.rand.m1 | 61 / 56 / 2 / 73 | 4.7 | +0 [-4, +4] | +0 [-3, +3] | +0 [-3, +3] | 116, +1 [-2, +4], 2 / 1 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +0 (113) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 70 / 61 / 3 / 58; dR -1 [-6, +4]; answered-rate change -2.1 pp [-5.2, +0.5]; out / in 6 / 2; mean words 7.9 -> 7.8
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -1 [-5, +3]; vs x.rand.m1 -2 [-7, +3] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -3 on 111 questions; mean words 4.8 -> 4.5
KEY family=E judge=allenai layer=0 A=not_met B=not_met dR=-2 dR_F=-1 dAns=-1 dAns_F=-4 rand_margin=-2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 0

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 44 / 49 / 1 / 98 | +0 [-4, +4] | +0 [-6, +6] | +1 [-4, +6] | 89, +0, 1 / 1 | 4 / 4 / 0 | +0 (85) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 49 / 41 / 0 / 102 | +4 [-1, +10] | +6 [+1, +12] | -6 [-12, -1] | 83, +1, 2 / 1 | 1 / 7 / 0 | +0 (77) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 100 / 89 / 3 / 0 | +0 [-5, +5] | -1 [-3, +0] | +0 [+0, +0] | 189, +0, 3 / 3 | 1 / 0 / 1 | +1 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-4, +4] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 2 / 1 | 2 / 0 / 1 | +1 (185) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 90 / 3 / 1 | -2 [-6, +2] | -2 [-5, +0] | +1 [+0, +3] | 188, -1, 1 / 2 | 2 / 0 / 1 | -1 (187) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 2 / 1 | -2 [-7, +3] | -1 [-3, +0] | +1 [+0, +3] | 189, -1, 2 / 3 | 1 / 0 / 0 | +0 (185) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 97 / 88 / 5 / 2 | -3 [-10, +3] | -5 [-10, -1] | +2 [+0, +5] | 185, +0, 4 / 4 | 5 / 0 / 3 | +2 (176) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 97 / 90 / 3 / 2 | -3 [-10, +3] | -3 [-7, +0] | +2 [+0, +5] | 187, -1, 4 / 5 | 3 / 0 / 1 | +1 (176) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 105 / 83 / 4 / 0 | -8 [-15, -1] | +2 [+0, +5] | -1 [-3, +0] | 186, -9, 2 / 11 | 0 / 2 / 0 | -9 (175) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 70 / 61 / 3 / 58 | -1 [-6, +4] | -4 [-10, +1] | +2 [-3, +7] | 129, +3, 3 / 0 | 6 / 2 / 2 | +2 (120) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 0

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 97 / 88 / 5 / 2 | 4.7 | -3 [-10, +3] | -5 [-10, -1] | +2 [+0, +5] | 185, +0 [-6, +6], 4 / 4 | 5 / 0 / 3 | 3 / 2 / 0 / 0 | +2 (176) |
| conf.md.p1 | 100 / 88 / 3 / 1 | 4.7 | +0 [-4, +4] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-2, +5], 2 / 1 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +1 (185) |
| conf.rand.p1 | 98 / 91 / 2 / 1 | 4.8 | -2 [-7, +3] | -1 [-3, +0] | +1 [+0, +3] | 189, -1 [-5, +3], 2 / 3 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (185) |
| conf.rand.m1 | 98 / 90 / 3 / 1 | 4.8 | -2 [-6, +2] | -2 [-5, +0] | +1 [+0, +3] | 188, -1 [-5, +2], 1 / 2 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (187) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 -1 [-7, +5]; vs conf.rand.m1 -1 [-8, +5] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.6 pp [-5.2, -0.5]; moved out / in 5 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 176 questions; mean words 4.9 -> 4.7
KEY family=E judge=allenai layer=0 A=not_met B=not_met dR=-3 dR_F=NA dAns=-5 dAns_F=NA rand_margin=-1 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:41:24 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

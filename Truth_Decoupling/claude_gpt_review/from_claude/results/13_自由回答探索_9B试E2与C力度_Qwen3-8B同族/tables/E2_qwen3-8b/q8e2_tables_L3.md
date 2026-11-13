TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 3

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 58 / 2 / 74 | 4.8 | -3 [-9, +3] | -1 [-5, +2] | +1 [-2, +5] | 115, -3 [-9, +3], 3 / 6 | 2 / 1 / 0 | 0 / 2 / 0 / 1 | -3 (105) |
| x.w.m1 | 61 / 57 / 2 / 72 | 4.6 | +0 [-5, +5] | +1 [-2, +4] | -1 [-4, +2] | 116, +0 [-5, +5], 3 / 3 | 1 / 2 / 0 | 0 / 1 / 0 / 2 | -1 (106) |
| x.md.p1 | 59 / 57 / 2 / 74 | 4.8 | -2 [-5, +0] | -1 [-3, +0] | +1 [+0, +3] | 116, -2 [-5, +0], 0 / 2 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -2 (108) |
| x.rand.p1 | 61 / 56 / 2 / 73 | 4.7 | +0 [-5, +5] | +0 [-3, +3] | +0 [-3, +3] | 116, +1 [-3, +5], 3 / 2 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +1 (112) |
| x.rand.m1 | 61 / 57 / 2 / 72 | 4.6 | +0 [-3, +3] | +1 [+0, +3] | -1 [-3, +0] | 117, +0 [-3, +3], 1 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (114) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 71 / 60 / 3 / 58; dR +0 [-6, +6]; answered-rate change -2.1 pp [-5.2, +0.5]; out / in 6 / 2; mean words 7.9 -> 7.8
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -3 [-9, +3]; vs x.rand.m1 -3 [-9, +3] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -3 on 105 questions; mean words 4.8 -> 4.8
KEY family=E judge=allenai layer=3 A=not_met B=not_met dR=-3 dR_F=+0 dAns=-1 dAns_F=-4 rand_margin=-3

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 3

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 45 / 45 / 1 / 101 | +1 [-4, +6] | -3 [-8, +2] | +4 [+0, +9] | 88, +2, 3 / 1 | 5 / 2 / 0 | +2 (84) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 49 / 41 / 0 / 102 | +4 [-1, +10] | +6 [+1, +12] | -6 [-12, -1] | 83, +1, 2 / 1 | 1 / 7 / 0 | +0 (78) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 97 / 90 / 4 / 1 | -3 [-8, +2] | -3 [-7, +0] | +1 [+0, +3] | 187, -1, 2 / 3 | 3 / 0 / 2 | -1 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 95 / 93 / 3 / 1 | -5 [-12, +2] | -2 [-5, +0] | +1 [+0, +3] | 188, -3, 4 / 7 | 2 / 0 / 1 | -2 (176) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 96 / 92 / 3 / 1 | -4 [-8, -1] | -2 [-5, +0] | +1 [+0, +3] | 188, -3, 0 / 3 | 2 / 0 / 1 | -2 (183) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 95 / 92 / 4 / 1 | -5 [-11, +1] | -3 [-7, +0] | +1 [+0, +3] | 187, -3, 2 / 5 | 3 / 0 / 2 | -3 (184) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 99 / 88 / 3 / 2 | -1 [-8, +7] | -3 [-7, +0] | +2 [+0, +5] | 187, +1, 7 / 6 | 3 / 0 / 1 | +2 (174) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 98 / 89 / 3 / 2 | -2 [-9, +5] | -3 [-7, +0] | +2 [+0, +5] | 187, +0, 6 / 6 | 3 / 0 / 1 | +1 (172) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 105 / 83 / 4 / 0 | -8 [-15, -1] | +2 [+0, +5] | -1 [-3, +0] | 186, -9, 2 / 11 | 0 / 2 / 0 | -9 (171) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 71 / 60 / 3 / 58 | +0 [-6, +6] | -4 [-10, +1] | +2 [-3, +7] | 129, +4, 5 / 1 | 6 / 2 / 2 | +2 (117) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 3

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 99 / 88 / 3 / 2 | 5.0 | -1 [-8, +7] | -3 [-7, +0] | +2 [+0, +5] | 187, +1 [-6, +8], 7 / 6 | 3 / 0 / 1 | 2 / 1 / 0 / 0 | +2 (174) |
| conf.md.p1 | 95 / 93 / 3 / 1 | 4.8 | -5 [-12, +2] | -2 [-5, +0] | +1 [+0, +3] | 188, -3 [-10, +3], 4 / 7 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | -2 (176) |
| conf.rand.p1 | 95 / 92 / 4 / 1 | 4.8 | -5 [-11, +1] | -3 [-7, +0] | +1 [+0, +3] | 187, -3 [-8, +2], 2 / 5 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | -3 (184) |
| conf.rand.m1 | 96 / 92 / 3 / 1 | 4.7 | -4 [-8, -1] | -2 [-5, +0] | +1 [+0, +3] | 188, -3 [-7, +0], 0 / 3 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -2 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +4 [-3, +11]; vs conf.rand.m1 +3 [-4, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 174 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=3 A=not_met B=not_met dR=-1 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+3 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:44:35 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

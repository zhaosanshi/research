TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 1

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 58 / 2 / 74 | 4.5 | -3 [-8, +2] | -1 [-5, +2] | +1 [-2, +5] | 115, -2 [-7, +3], 2 / 4 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | -2 (108) |
| x.w.m1 | 65 / 54 / 2 / 71 | 4.6 | +4 [+0, +9] | +2 [-2, +6] | -2 [-6, +2] | 116, +4 [+0, +9], 5 / 1 | 1 / 3 / 0 | 0 / 1 / 0 / 3 | +3 (107) |
| x.md.p1 | 62 / 54 / 2 / 74 | 4.8 | +1 [-2, +5] | -1 [-3, +0] | +1 [+0, +3] | 116, +1 [-2, +5], 2 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +1 (108) |
| x.rand.p1 | 59 / 57 / 2 / 74 | 4.4 | -2 [-5, +0] | -1 [-4, +2] | +1 [-2, +4] | 115, +0 [+0, +0], 0 / 0 | 2 / 1 / 0 | 2 / 0 / 0 / 1 | +0 (113) |
| x.rand.m1 | 62 / 55 / 2 / 73 | 4.6 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 117, +1 [-2, +5], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (114) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 72 / 60 / 3 / 57; dR +1 [-4, +6]; answered-rate change -1.6 pp [-4.2, +1.0]; out / in 5 / 2; mean words 7.9 -> 7.8
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -1 [-6, +4]; vs x.rand.m1 -4 [-10, +2] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -2 on 108 questions; mean words 4.8 -> 4.5
KEY family=E judge=allenai layer=1 A=not_met B=not_met dR=-3 dR_F=+1 dAns=-1 dAns_F=-3 rand_margin=-4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 1

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 45 / 44 / 1 / 102 | +1 [-3, +5] | -4 [-10, +1] | +5 [+0, +10] | 87, +2, 2 / 0 | 6 / 2 / 0 | +2 (85) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 50 / 41 / 0 / 101 | +5 [+0, +10] | +7 [+2, +12] | -7 [-12, -2] | 84, +1, 2 / 1 | 0 / 7 / 0 | +0 (81) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 97 / 91 / 3 / 1 | -3 [-7, +1] | -2 [-5, +0] | +1 [+0, +3] | 188, -1, 1 / 2 | 2 / 0 / 1 | +0 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-5, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 3 / 2 | 2 / 0 / 1 | +2 (180) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-4, +2] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 1 / 1 | 2 / 0 / 1 | +0 (186) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 3 / 0 | -2 [-6, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 1 / 3 | 1 / 0 / 1 | -2 (183) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 98 / 89 / 3 / 2 | -2 [-9, +5] | -3 [-7, +0] | +2 [+0, +5] | 187, +0, 5 / 5 | 3 / 0 / 1 | +1 (170) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 96 / 90 / 4 / 2 | -4 [-12, +4] | -4 [-8, -1] | +2 [+0, +5] | 186, -1, 6 / 7 | 4 / 0 / 2 | +0 (172) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 105 / 83 / 4 / 0 | -8 [-15, -1] | +2 [+0, +5] | -1 [-3, +0] | 186, -9, 2 / 11 | 0 / 2 / 0 | -9 (172) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 72 / 60 / 3 / 57 | +1 [-4, +6] | -3 [-8, +2] | +1 [-3, +6] | 130, +4, 4 / 0 | 5 / 2 / 2 | +3 (118) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 1

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 98 / 89 / 3 / 2 | 5.0 | -2 [-9, +5] | -3 [-7, +0] | +2 [+0, +5] | 187, +0 [-6, +6], 5 / 5 | 3 / 0 / 1 | 2 / 1 / 0 / 0 | +1 (170) |
| conf.md.p1 | 100 / 88 / 3 / 1 | 4.8 | +0 [-5, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-3, +5], 3 / 2 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +2 (180) |
| conf.rand.p1 | 98 / 91 / 3 / 0 | 4.9 | -2 [-6, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-6, +2], 1 / 3 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -2 (183) |
| conf.rand.m1 | 99 / 89 / 3 / 1 | 4.7 | -1 [-4, +2] | -2 [-5, +0] | +1 [+0, +3] | 188, +0 [-3, +3], 1 / 1 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +0 (186) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +0 [-7, +7]; vs conf.rand.m1 -1 [-8, +5] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +1 on 170 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=1 A=not_met B=not_met dR=-2 dR_F=NA dAns=-3 dAns_F=NA rand_margin=-1 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:42:27 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

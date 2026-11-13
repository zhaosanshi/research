TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 5

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 55 / 2 / 77 | 4.7 | -3 [-9, +3] | -4 [-9, +0] | +4 [+0, +9] | 112, -3 [-8, +2], 2 / 5 | 5 / 1 / 0 | 1 / 4 / 1 / 0 | -1 (104) |
| x.w.m1 | 64 / 54 / 3 / 71 | 4.6 | +3 [-3, +10] | +1 [-3, +5] | -2 [-6, +2] | 115, +3 [-3, +9], 6 / 3 | 2 / 3 / 1 | 1 / 1 / 1 / 2 | +2 (102) |
| x.md.p1 | 61 / 57 / 2 / 72 | 4.8 | +0 [-5, +5] | +1 [-2, +4] | -1 [-4, +2] | 116, +0 [-5, +5], 3 / 3 | 1 / 2 / 0 | 0 / 1 / 0 / 2 | +1 (108) |
| x.rand.p1 | 62 / 56 / 2 / 72 | 4.8 | +1 [-3, +5] | +1 [-2, +5] | -1 [-5, +2] | 116, +0 [-4, +4], 2 / 2 | 1 / 2 / 0 | 0 / 1 / 1 / 1 | +0 (107) |
| x.rand.m1 | 61 / 54 / 3 / 74 | 4.5 | +0 [-4, +4] | -2 [-6, +2] | +1 [-2, +4] | 114, +1 [-2, +4], 2 / 1 | 3 / 1 / 1 | 1 / 2 / 0 / 1 | +0 (113) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 68 / 60 / 2 / 62; dR -3 [-9, +3]; answered-rate change -3.6 pp [-6.8, -1.0]; out / in 8 / 1; mean words 7.9 -> 7.8
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -4 [-10, +2]; vs x.rand.m1 -3 [-9, +3] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.1 pp [-4.7, +0.0]; moved out / in 5 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same -1 on 104 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=5 A=not_met B=not_met dR=-3 dR_F=-3 dAns=-4 dAns_F=-7 rand_margin=-4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 5

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 44 / 43 / 2 / 103 | +0 [-5, +5] | -6 [-12, +0] | +6 [+0, +12] | 85, +1, 2 / 1 | 8 / 2 / 0 | +1 (82) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 47 / 43 / 0 / 102 | +2 [-3, +8] | +6 [+1, +12] | -6 [-12, -1] | 83, -1, 1 / 2 | 1 / 7 / 0 | -2 (74) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 2 / 1 | -2 [-8, +3] | -1 [-3, +0] | +1 [+0, +3] | 189, -1, 3 / 4 | 1 / 0 / 0 | -1 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 92 / 95 / 4 / 1 | -8 [-14, -2] | -3 [-7, +0] | +1 [+0, +3] | 187, -5, 1 / 6 | 3 / 0 / 2 | -4 (175) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 94 / 94 / 3 / 1 | -6 [-11, -2] | -2 [-5, +0] | +1 [+0, +3] | 188, -5, 0 / 5 | 2 / 0 / 1 | -3 (180) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 100 / 89 / 3 / 0 | +0 [-4, +4] | -1 [-3, +0] | +0 [+0, +0] | 189, +1, 2 / 1 | 1 / 0 / 1 | +1 (184) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-6, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 5 / 4 | 2 / 0 / 1 | +2 (169) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 98 / 89 / 4 / 1 | -2 [-9, +5] | -3 [-7, +0] | +1 [+0, +3] | 187, +0, 5 / 5 | 3 / 0 / 2 | +2 (170) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 103 / 86 / 2 / 1 | -10 [-19, -1] | +3 [+0, +7] | +0 [+0, +0] | 186, -13, 2 / 15 | 0 / 3 / 0 | -13 (167) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 68 / 60 / 2 / 62 | -3 [-9, +3] | -7 [-13, -2] | +6 [+1, +12] | 127, +2, 3 / 1 | 8 / 1 / 1 | +1 (116) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 5

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 88 / 3 / 1 | 5.0 | +0 [-6, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-5, +7], 5 / 4 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +2 (169) |
| conf.md.p1 | 92 / 95 / 4 / 1 | 5.0 | -8 [-14, -2] | -3 [-7, +0] | +1 [+0, +3] | 187, -5 [-10, +0], 1 / 6 | 3 / 0 / 2 | 3 / 0 / 0 / 0 | -4 (175) |
| conf.rand.p1 | 100 / 89 / 3 / 0 | 4.7 | +0 [-4, +4] | -1 [-3, +0] | +0 [+0, +0] | 189, +1 [-2, +5], 2 / 1 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +1 (184) |
| conf.rand.m1 | 94 / 94 / 3 / 1 | 5.0 | -6 [-11, -2] | -2 [-5, +0] | +1 [+0, +3] | 188, -5 [-10, -1], 0 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -3 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +0 [-7, +7]; vs conf.rand.m1 +6 [+2, +11] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +2 on 169 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=5 A=not_met B=not_met dR=+0 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+0 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:46:45 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

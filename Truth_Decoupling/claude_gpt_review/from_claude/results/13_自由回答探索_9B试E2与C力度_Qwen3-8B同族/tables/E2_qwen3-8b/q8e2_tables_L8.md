TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 57 / 53 / 2 / 80 | 4.5 | -4 [-11, +3] | -7 [-13, -2] | +7 [+2, +13] | 109, -2 [-8, +3], 3 / 5 | 8 / 1 / 0 | 3 / 5 / 1 / 0 | -1 (99) |
| x.w.m1 | 62 / 59 / 3 / 68 | 4.5 | +1 [-4, +6] | +4 [-1, +10] | -5 [-10, +0] | 115, +1 [-3, +6], 3 / 2 | 2 / 6 / 1 | 1 / 1 / 1 / 5 | +1 (103) |
| x.md.p1 | 59 / 58 / 2 / 73 | 4.6 | -2 [-7, +3] | +0 [-4, +4] | +0 [-4, +4] | 115, -1 [-6, +3], 2 / 3 | 2 / 2 / 0 | 1 / 1 / 0 / 2 | +0 (109) |
| x.rand.p1 | 64 / 53 / 2 / 73 | 4.9 | +3 [-1, +8] | +0 [-3, +3] | +0 [-3, +3] | 116, +2 [-2, +6], 3 / 1 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | +2 (107) |
| x.rand.m1 | 59 / 57 / 2 / 74 | 4.5 | -2 [-7, +3] | -1 [-5, +3] | +1 [-3, +5] | 114, +0 [-4, +4], 2 / 2 | 3 / 2 / 0 | 2 / 1 / 0 / 2 | +0 (110) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 65 / 57 / 1 / 69; dR -6 [-13, +1]; answered-rate change -6.8 pp [-10.9, -3.1]; out / in 14 / 1; mean words 7.9 -> 7.4
- main effect: dR -4 (-2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -7 [-14, +0]; vs x.rand.m1 -2 [-8, +4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.6 pp [-6.8, -1.0]; moved out / in 8 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same -1 on 99 questions; mean words 4.8 -> 4.5
KEY family=E judge=allenai layer=8 A=not_met B=not_met dR=-4 dR_F=-6 dAns=-7 dAns_F=-13 rand_margin=-7

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 8

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 44 / 40 / 2 / 106 | +0 [-5, +5] | -9 [-16, -2] | +9 [+2, +16] | 82, +2, 3 / 1 | 11 / 2 / 0 | +2 (77) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 49 / 43 / 0 / 100 | +4 [-1, +10] | +8 [+2, +14] | -8 [-14, -2] | 83, +0, 1 / 1 | 1 / 9 / 0 | +0 (77) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 2 / 1 | -2 [-7, +3] | -1 [-3, +0] | +1 [+0, +3] | 189, -1, 2 / 3 | 1 / 0 / 0 | +0 (185) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 93 / 92 / 6 / 1 | -7 [-13, -2] | -5 [-10, -1] | +1 [+0, +3] | 185, -4, 1 / 5 | 5 / 0 / 4 | -3 (176) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 96 / 92 / 4 / 0 | -4 [-10, +1] | -2 [-5, +0] | +0 [+0, +0] | 188, -2, 2 / 4 | 2 / 0 / 2 | -2 (185) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 94 / 95 / 2 / 1 | -6 [-12, -1] | -1 [-3, +0] | +1 [+0, +3] | 189, -5, 1 / 6 | 1 / 0 / 0 | -4 (176) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 98 / 90 / 3 / 1 | -2 [-10, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, -1, 7 / 8 | 2 / 0 / 1 | -1 (170) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-9, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 8 / 8 | 2 / 0 / 1 | +0 (172) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 104 / 86 / 2 / 0 | -9 [-20, +1] | +4 [+1, +8] | -1 [-3, +0] | 186, -13, 6 / 19 | 0 / 4 / 0 | -13 (169) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 65 / 57 / 1 / 69 | -6 [-13, +1] | -13 [-21, -6] | +13 [+6, +21] | 121, +1, 3 / 2 | 14 / 1 / 0 | +0 (114) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 98 / 90 / 3 / 1 | 4.8 | -2 [-10, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, -1 [-9, +6], 7 / 8 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (170) |
| conf.md.p1 | 93 / 92 / 6 / 1 | 4.9 | -7 [-13, -2] | -5 [-10, -1] | +1 [+0, +3] | 185, -4 [-9, +0], 1 / 5 | 5 / 0 / 4 | 3 / 2 / 0 / 0 | -3 (176) |
| conf.rand.p1 | 94 / 95 / 2 / 1 | 4.9 | -6 [-12, -1] | -1 [-3, +0] | +1 [+0, +3] | 189, -5 [-10, +0], 1 / 6 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | -4 (176) |
| conf.rand.m1 | 96 / 92 / 4 / 0 | 4.5 | -4 [-10, +1] | -2 [-5, +0] | +0 [+0, +0] | 188, -2 [-7, +3], 2 / 4 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | -2 (185) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +4 [-3, +11]; vs conf.rand.m1 +2 [-6, +10] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -1 on 170 questions; mean words 4.9 -> 4.8
KEY family=E judge=allenai layer=8 A=not_met B=not_met dR=-2 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+2 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:50:01 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 6

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 55 / 2 / 77 | 4.7 | -3 [-9, +3] | -4 [-10, +1] | +4 [-1, +10] | 111, -3 [-8, +2], 2 / 5 | 6 / 2 / 0 | 2 / 4 / 2 / 0 | -1 (103) |
| x.w.m1 | 65 / 56 / 3 / 68 | 4.6 | +4 [-2, +10] | +4 [-1, +10] | -5 [-10, +0] | 115, +3 [-2, +8], 5 / 2 | 2 / 6 / 1 | 1 / 1 / 2 / 4 | +2 (105) |
| x.md.p1 | 59 / 57 / 2 / 74 | 4.8 | -2 [-7, +3] | -1 [-5, +2] | +1 [-2, +5] | 115, -1 [-5, +3], 2 / 3 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +0 (106) |
| x.rand.p1 | 60 / 58 / 2 / 72 | 4.7 | -1 [-5, +2] | +1 [-2, +4] | -1 [-4, +2] | 116, -1 [-5, +2], 1 / 2 | 1 / 2 / 0 | 0 / 1 / 0 / 2 | -1 (112) |
| x.rand.m1 | 58 / 56 / 2 / 76 | 4.6 | -3 [-8, +1] | -3 [-7, +0] | +3 [+0, +7] | 114, -1 [-5, +2], 1 / 2 | 3 / 0 / 0 | 2 / 1 / 0 / 0 | -1 (107) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 66 / 58 / 1 / 67; dR -5 [-12, +1]; answered-rate change -5.7 pp [-9.4, -2.1]; out / in 12 / 1; mean words 7.9 -> 7.5
- main effect: dR -3 (-1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -2 [-8, +4]; vs x.rand.m1 +0 [-7, +7] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.1 pp [-5.2, +0.5]; moved out / in 6 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same -1 on 103 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=6 A=not_met B=not_met dR=-3 dR_F=-5 dAns=-4 dAns_F=-11 rand_margin=-2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 6

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 47 / 43 / 0 / 102 | +3 [-2, +8] | -3 [-9, +3] | +5 [+0, +10] | 87, +2, 2 / 0 | 6 / 3 / 0 | +1 (85) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 47 / 42 / 0 / 103 | +2 [-3, +8] | +5 [-1, +11] | -5 [-11, +1] | 82, +0, 1 / 1 | 2 / 7 / 0 | -1 (73) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 4 / 0 | -1 [-5, +2] | -2 [-5, +0] | +0 [+0, +0] | 188, -1, 1 / 2 | 2 / 0 / 2 | -1 (183) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 94 / 95 / 3 / 0 | -6 [-12, -1] | -1 [-3, +0] | +0 [+0, +0] | 189, -5, 1 / 6 | 1 / 0 / 1 | -4 (175) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 91 / 3 / 0 | -2 [-6, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 1 / 3 | 1 / 0 / 1 | -1 (181) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 96 / 92 / 3 / 1 | -4 [-10, +1] | -2 [-5, +0] | +1 [+0, +3] | 188, -3, 2 / 5 | 2 / 0 / 1 | -2 (175) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 98 / 90 / 3 / 1 | -2 [-9, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, -1, 5 / 6 | 2 / 0 / 1 | +0 (169) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-8, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 6 / 6 | 2 / 0 / 1 | +0 (170) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 99 / 88 / 4 / 1 | -14 [-23, -6] | +1 [+0, +3] | +0 [+0, +0] | 186, -15, 2 / 17 | 0 / 1 / 0 | -14 (167) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 66 / 58 / 1 / 67 | -5 [-12, +1] | -11 [-18, -4] | +11 [+4, +18] | 123, +2, 3 / 1 | 12 / 1 / 0 | +2 (115) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 6

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 98 / 90 / 3 / 1 | 4.8 | -2 [-9, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, -1 [-8, +5], 5 / 6 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +0 (169) |
| conf.md.p1 | 94 / 95 / 3 / 0 | 5.1 | -6 [-12, -1] | -1 [-3, +0] | +0 [+0, +0] | 189, -5 [-10, +0], 1 / 6 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -4 (175) |
| conf.rand.p1 | 96 / 92 / 3 / 1 | 5.0 | -4 [-10, +1] | -2 [-5, +0] | +1 [+0, +3] | 188, -3 [-8, +2], 2 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -2 (175) |
| conf.rand.m1 | 98 / 91 / 3 / 0 | 4.7 | -2 [-6, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-6, +2], 1 / 3 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -1 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +2 [-5, +9]; vs conf.rand.m1 +0 [-7, +7] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +0 on 169 questions; mean words 4.9 -> 4.8
KEY family=E judge=allenai layer=6 A=not_met B=not_met dR=-2 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+0 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:47:50 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 24

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 74 / 47 / 2 / 69 | 4.8 | +13 [+6, +21] | +4 [+0, +9] | -4 [-9, +0] | 116, +8 [+2, +15], 10 / 2 | 1 / 5 / 0 | 0 / 1 / 5 / 0 | +4 (98) |
| x.w.m1 | 63 / 50 / 2 / 77 | 4.8 | +2 [-4, +8] | -4 [-9, +0] | +4 [+0, +9] | 112, +5 [+0, +10], 6 / 1 | 5 / 1 / 0 | 3 / 2 / 0 / 1 | +4 (101) |
| x.md.p1 | 66 / 54 / 3 / 69 | 4.5 | +5 [-1, +12] | +3 [-3, +9] | -4 [-10, +1] | 114, +1 [-3, +5], 3 / 2 | 3 / 6 / 1 | 1 / 2 / 5 / 1 | +1 (105) |
| x.rand.p1 | 64 / 55 / 2 / 71 | 4.9 | +3 [-2, +8] | +2 [+0, +5] | -2 [-5, +0] | 117, +2 [-3, +7], 4 / 2 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +0 (109) |
| x.rand.m1 | 59 / 56 / 2 / 75 | 4.7 | -2 [-6, +2] | -2 [-5, +0] | +2 [+0, +5] | 115, +0 [-3, +3], 1 / 1 | 2 / 0 / 0 | 2 / 0 / 0 / 0 | +0 (109) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 74 / 63 / 2 / 53; dR +3 [-4, +10]; answered-rate change +1.0 pp [-1.0, +3.1]; out / in 1 / 3; mean words 7.9 -> 7.9
- main effect: dR +13 (6.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +10 [+4, +17]; vs x.rand.m1 +15 [+7, +24] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +2.1 pp [+0.0, +4.7]; moved out / in 1 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +4 on 98 questions; mean words 4.8 -> 4.8
KEY family=E judge=allenai layer=24 A=met B=not_met dR=+13 dR_F=+3 dAns=+4 dAns_F=+2 rand_margin=+10

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 24

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 50 / 49 / 1 / 92 | +6 [+0, +13] | +6 [+0, +12] | -5 [-11, +0] | 91, +1, 3 / 2 | 2 / 8 / 0 | -1 (82) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 41 / 37 / 0 / 114 | -4 [-11, +2] | -6 [-14, +1] | +6 [-1, +14] | 74, +1, 3 / 2 | 10 / 4 / 0 | +0 (68) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 2 / 1 | -2 [-6, +2] | -1 [-3, +0] | +1 [+0, +3] | 189, -1, 1 / 2 | 1 / 0 / 0 | -1 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-7, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 5 / 5 | 2 / 0 / 1 | -1 (168) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-9, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, -2, 5 / 7 | 0 / 0 / 0 | -3 (183) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 99 / 90 / 3 / 0 | -1 [-5, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -1, 1 / 2 | 1 / 0 / 1 | +0 (179) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 107 / 81 / 3 / 1 | +7 [-1, +15] | -2 [-5, +0] | +1 [+0, +3] | 188, +8, 12 / 4 | 2 / 0 / 1 | +5 (163) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 104 / 84 / 3 / 1 | +4 [-4, +12] | -2 [-5, +0] | +1 [+0, +3] | 188, +5, 10 / 5 | 2 / 0 / 1 | +4 (165) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 99 / 89 / 3 / 1 | -14 [-26, -2] | +2 [-3, +7] | +0 [-3, +3] | 184, -16, 7 / 23 | 2 / 4 / 1 | -13 (164) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 74 / 63 / 2 / 53 | +3 [-4, +10] | +2 [-2, +6] | -3 [-7, +0] | 134, +3, 7 / 4 | 1 / 3 / 1 | -1 (121) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 24

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 107 / 81 / 3 / 1 | 4.6 | +7 [-1, +15] | -2 [-5, +0] | +1 [+0, +3] | 188, +8 [+0, +16], 12 / 4 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +5 (163) |
| conf.md.p1 | 99 / 89 / 3 / 1 | 4.5 | -1 [-7, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, +0 [-6, +6], 5 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (168) |
| conf.rand.p1 | 99 / 90 / 3 / 0 | 4.9 | -1 [-5, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -1 [-5, +2], 1 / 2 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +0 (179) |
| conf.rand.m1 | 98 / 92 / 2 / 0 | 4.9 | -2 [-9, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, -2 [-9, +5], 5 / 7 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -3 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +8 [+0, +17]; vs conf.rand.m1 +9 [+0, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +5 on 163 questions; mean words 4.9 -> 4.6
KEY family=E judge=allenai layer=24 A=incomplete B=not_met dR=+7 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+8 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 14:43:40 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

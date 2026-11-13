TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 59 / 40 / 1 / 92 | 4.8 | -2 [-9, +5] | -18 [-26, -10] | +19 [+11, +27] | 98, +4 [-1, +10], 6 / 2 | 19 / 1 / 0 | 6 / 13 / 0 / 1 | +1 (82) |
| x.w.m1 | 70 / 64 / 2 / 56 | 5.0 | +9 [+1, +18] | +17 [+10, +25] | -17 [-25, -10] | 117, +2 [-5, +9], 7 / 5 | 0 / 17 / 0 | 0 / 0 / 7 / 10 | +3 (104) |
| x.md.p1 | 61 / 54 / 3 / 74 | 4.8 | +0 [-6, +6] | -2 [-8, +3] | +1 [-4, +6] | 112, +3 [-1, +8], 4 / 1 | 5 / 3 / 1 | 3 / 2 / 0 / 3 | +2 (98) |
| x.rand.p1 | 62 / 54 / 2 / 74 | 4.8 | +1 [-2, +4] | -1 [-5, +2] | +1 [-2, +5] | 115, +0 [-3, +3], 1 / 1 | 2 / 1 / 0 | 0 / 2 / 1 / 0 | +0 (104) |
| x.rand.m1 | 64 / 53 / 2 / 73 | 4.4 | +3 [-2, +8] | +0 [-4, +4] | +0 [-4, +4] | 115, +4 [+0, +9], 5 / 1 | 2 / 2 / 0 | 1 / 1 / 0 / 2 | +5 (113) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 62 / 49 / 0 / 81; dR -9 [-17, -1]; answered-rate change -12.5 pp [-17.2, -8.3]; out / in 24 / 0; mean words 7.9 -> 6.8
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -3 [-11, +5]; vs x.rand.m1 -5 [-12, +2] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -9.4 pp [-13.5, -5.2]; moved out / in 19 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +1 on 82 questions; mean words 4.8 -> 4.8
KEY family=E judge=allenai layer=13 A=not_met B=not_met dR=-2 dR_F=-9 dAns=-18 dAns_F=-24 rand_margin=-5

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 13

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 46 / 32 / 0 / 114 | +2 [-6, +10] | -15 [-23, -8] | +17 [+10, +25] | 77, +5, 8 / 3 | 16 / 1 / 0 | +4 (71) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 56 / 57 / 1 / 78 | +11 [+3, +20] | +29 [+20, +39] | -30 [-40, -21] | 84, -2, 2 / 4 | 0 / 29 / 0 | -1 (75) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 99 / 90 / 3 / 0 | -1 [-6, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, -1, 2 / 3 | 1 / 0 / 1 | +0 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 97 / 91 / 4 / 0 | -3 [-8, +2] | -2 [-5, +0] | +0 [+0, +0] | 188, -3, 2 / 5 | 2 / 0 / 2 | -4 (168) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 97 / 91 / 4 / 0 | -3 [-8, +2] | -2 [-5, +0] | +0 [+0, +0] | 188, -1, 2 / 3 | 2 / 0 / 2 | -1 (184) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 97 / 93 / 2 / 0 | -3 [-8, +1] | +0 [+0, +0] | +0 [+0, +0] | 190, -3, 1 / 4 | 0 / 0 / 0 | -4 (179) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 107 / 82 / 2 / 1 | +7 [-2, +16] | -1 [-5, +2] | +1 [+0, +3] | 188, +7, 14 / 7 | 2 / 1 / 1 | +3 (166) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 108 / 81 / 2 / 1 | +8 [-1, +17] | -1 [-5, +2] | +1 [+0, +3] | 188, +8, 14 / 6 | 2 / 1 / 1 | +5 (166) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 100 / 89 / 3 / 0 | -13 [-23, -3] | +3 [+0, +7] | -1 [-3, +0] | 186, -16, 5 / 21 | 0 / 3 / 0 | -12 (162) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 62 / 49 / 0 / 81 | -9 [-17, -1] | -24 [-33, -16] | +25 [+16, +34] | 111, +2, 4 / 2 | 24 / 0 / 0 | +2 (99) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 107 / 82 / 2 / 1 | 4.9 | +7 [-2, +16] | -1 [-5, +2] | +1 [+0, +3] | 188, +7 [-2, +16], 14 / 7 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +3 (166) |
| conf.md.p1 | 97 / 91 / 4 / 0 | 5.4 | -3 [-8, +2] | -2 [-5, +0] | +0 [+0, +0] | 188, -3 [-8, +2], 2 / 5 | 2 / 0 / 2 | 0 / 2 / 0 / 0 | -4 (168) |
| conf.rand.p1 | 97 / 93 / 2 / 0 | 5.0 | -3 [-8, +1] | +0 [+0, +0] | +0 [+0, +0] | 190, -3 [-8, +1], 1 / 4 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -4 (179) |
| conf.rand.m1 | 97 / 91 / 4 / 0 | 4.7 | -3 [-8, +2] | -2 [-5, +0] | +0 [+0, +0] | 188, -1 [-5, +3], 2 / 3 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | -1 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +10 [+1, +19]; vs conf.rand.m1 +10 [+1, +20] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +3 on 166 questions; mean words 4.9 -> 4.9
KEY family=E judge=allenai layer=13 A=incomplete B=not_met dR=+7 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+10 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 14:34:36 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

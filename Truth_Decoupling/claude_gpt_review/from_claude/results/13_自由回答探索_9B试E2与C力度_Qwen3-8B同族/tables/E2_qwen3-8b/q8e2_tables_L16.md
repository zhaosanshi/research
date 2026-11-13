TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 50 / 38 / 2 / 102 | 4.2 | -11 [-19, -3] | -29 [-39, -19] | +29 [+20, +39] | 87, +1 [-4, +6], 4 / 3 | 30 / 1 / 1 | 12 / 18 / 0 / 1 | -1 (79) |
| x.w.m1 | 68 / 67 / 2 / 55 | 4.8 | +7 [-1, +15] | +18 [+10, +26] | -18 [-26, -10] | 117, -1 [-7, +5], 4 / 5 | 0 / 18 / 0 | 0 / 0 / 8 / 10 | -1 (101) |
| x.md.p1 | 59 / 54 / 2 / 77 | 5.0 | -2 [-8, +4] | -4 [-10, +2] | +4 [-1, +10] | 110, +0 [-5, +5], 3 / 3 | 7 / 3 / 1 | 3 / 4 / 1 / 2 | +0 (95) |
| x.rand.p1 | 61 / 59 / 2 / 70 | 4.9 | +0 [-3, +3] | +3 [-1, +8] | -3 [-8, +1] | 116, +0 [-3, +3], 1 / 1 | 1 / 4 / 0 | 0 / 1 / 0 / 4 | +1 (105) |
| x.rand.m1 | 62 / 51 / 2 / 77 | 4.6 | +1 [-5, +7] | -4 [-8, -1] | +4 [+1, +8] | 113, +3 [-2, +8], 5 / 2 | 4 / 0 / 0 | 2 / 2 / 0 / 0 | +2 (106) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 63 / 50 / 1 / 78; dR -8 [-17, +1]; answered-rate change -11.5 pp [-16.1, -6.8]; out / in 23 / 1; mean words 7.9 -> 7.1
- main effect: dR -11 (-5.7% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -11 [-19, -3]; vs x.rand.m1 -12 [-20, -4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -15.1 pp [-20.3, -9.9]; moved out / in 30 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same -1 on 79 questions; mean words 4.8 -> 4.2
KEY family=E judge=allenai layer=16 A=not_met B=not_met dR=-11 dR_F=-8 dAns=-29 dAns_F=-22 rand_margin=-12

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 40 / 27 / 0 / 125 | -4 [-12, +4] | -26 [-36, -17] | +28 [+19, +38] | 66, +5, 6 / 1 | 27 / 1 / 0 | +3 (58) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 55 / 59 / 2 / 76 | +10 [+1, +19] | +30 [+21, +40] | -32 [-42, -22] | 84, -4, 3 / 7 | 0 / 30 / 0 | -1 (75) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 96 / 93 / 2 / 1 | -4 [-8, -1] | -1 [-3, +0] | +1 [+0, +3] | 189, -3, 0 / 3 | 1 / 0 / 0 | -3 (188) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 92 / 98 / 1 / 1 | -8 [-16, -1] | +0 [-3, +3] | +1 [+0, +3] | 189, -8, 2 / 10 | 1 / 1 / 0 | -6 (167) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 100 / 89 / 2 / 1 | +0 [-6, +6] | -1 [-3, +0] | +1 [+0, +3] | 189, +1, 5 / 4 | 1 / 0 / 0 | +0 (179) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 92 / 96 / 3 / 1 | -8 [-14, -2] | -2 [-5, +0] | +1 [+0, +3] | 188, -6, 1 / 7 | 2 / 0 / 1 | -5 (174) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 107 / 81 / 3 / 1 | +7 [-3, +18] | -2 [-6, +2] | +1 [+0, +3] | 187, +8, 17 / 9 | 3 / 1 / 2 | +5 (158) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 105 / 83 / 3 / 1 | +5 [-5, +15] | -2 [-6, +2] | +1 [+0, +3] | 187, +6, 16 / 10 | 3 / 1 / 2 | +4 (159) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 93 / 96 / 3 / 0 | -20 [-31, -9] | +3 [+0, +7] | -1 [-3, +0] | 186, -23, 3 / 26 | 0 / 3 / 0 | -19 (166) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 63 / 50 / 1 / 78 | -8 [-17, +1] | -22 [-31, -13] | +22 [+13, +31] | 112, +4, 7 / 3 | 23 / 1 / 0 | +2 (98) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 107 / 81 / 3 / 1 | 4.7 | +7 [-3, +18] | -2 [-6, +2] | +1 [+0, +3] | 187, +8 [-2, +18], 17 / 9 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +5 (158) |
| conf.md.p1 | 92 / 98 / 1 / 1 | 4.9 | -8 [-16, -1] | +0 [-3, +3] | +1 [+0, +3] | 189, -8 [-15, -2], 2 / 10 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | -6 (167) |
| conf.rand.p1 | 92 / 96 / 3 / 1 | 4.8 | -8 [-14, -2] | -2 [-5, +0] | +1 [+0, +3] | 188, -6 [-12, -1], 1 / 7 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | -5 (174) |
| conf.rand.m1 | 100 / 89 / 2 / 1 | 4.9 | +0 [-6, +6] | -1 [-3, +0] | +1 [+0, +3] | 189, +1 [-5, +7], 5 / 4 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (179) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +15 [+4, +26]; vs conf.rand.m1 +7 [-3, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-3.1, +1.0]; moved out / in 3 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 158 questions; mean words 4.9 -> 4.7
KEY family=E judge=allenai layer=16 A=incomplete B=not_met dR=+7 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+7 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 14:38:10 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

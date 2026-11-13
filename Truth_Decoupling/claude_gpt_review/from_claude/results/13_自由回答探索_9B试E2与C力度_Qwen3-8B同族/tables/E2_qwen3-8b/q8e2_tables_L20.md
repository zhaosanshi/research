TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 51 / 37 / 2 / 102 | 4.3 | -10 [-19, -1] | -29 [-40, -19] | +29 [+19, +39] | 86, +0 [-6, +6], 5 / 5 | 31 / 2 / 1 | 11 / 20 / 1 / 1 | -2 (74) |
| x.w.m1 | 71 / 72 / 2 / 47 | 5.4 | +10 [+2, +18] | +26 [+17, +36] | -26 [-36, -17] | 117, +3 [-3, +9], 6 / 3 | 0 / 26 / 0 | 0 / 0 / 7 / 19 | +3 (98) |
| x.md.p1 | 54 / 53 / 2 / 83 | 4.3 | -7 [-13, -2] | -10 [-16, -4] | +10 [+4, +16] | 107, -1 [-5, +2], 1 / 2 | 10 / 0 / 0 | 6 / 4 / 0 / 0 | +0 (96) |
| x.rand.p1 | 63 / 53 / 2 / 74 | 4.9 | +2 [-3, +8] | -1 [-4, +2] | +1 [-2, +4] | 115, +3 [-2, +8], 5 / 2 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +1 (99) |
| x.rand.m1 | 57 / 59 / 2 / 74 | 4.3 | -4 [-8, -1] | -1 [-5, +3] | +1 [-3, +5] | 114, -2 [-5, +0], 0 / 2 | 3 / 2 / 0 | 2 / 1 / 0 / 2 | -1 (108) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 62 / 46 / 1 / 83; dR -9 [-17, -1]; answered-rate change -14.1 pp [-19.3, -9.4]; out / in 27 / 0; mean words 7.9 -> 6.9
- main effect: dR -10 (-5.2% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -12 [-22, -3]; vs x.rand.m1 -6 [-15, +2] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -15.1 pp [-20.8, -9.9]; moved out / in 31 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same -2 on 74 questions; mean words 4.8 -> 4.3
KEY family=E judge=allenai layer=20 A=not_met B=not_met dR=-10 dR_F=-9 dAns=-29 dAns_F=-27 rand_margin=-12

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 20

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 34 / 28 / 1 / 129 | -10 [-19, -2] | -31 [-41, -21] | +32 [+22, +42] | 62, +2, 5 / 3 | 31 / 0 / 0 | -1 (56) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 55 / 57 / 4 / 76 | +10 [+2, +19] | +28 [+18, +38] | -32 [-42, -22] | 83, +1, 5 / 4 | 1 / 29 / 1 | +0 (75) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-6, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -2, 1 / 3 | 0 / 0 / 0 | -2 (188) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 92 / 97 / 2 / 1 | -8 [-17, +1] | -1 [-5, +2] | +1 [+0, +3] | 188, -8, 6 / 14 | 2 / 1 / 1 | -3 (171) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 93 / 96 / 3 / 0 | -7 [-14, -1] | -1 [-3, +0] | +0 [+0, +0] | 189, -6, 2 / 8 | 1 / 0 / 1 | -4 (180) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 102 / 87 / 3 / 0 | +2 [-4, +8] | -1 [-5, +2] | +0 [+0, +0] | 188, +1, 4 / 3 | 2 / 1 / 2 | +0 (170) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 105 / 81 / 4 / 2 | +5 [-5, +15] | -4 [-8, -1] | +2 [+0, +5] | 186, +7, 16 / 9 | 4 / 0 / 2 | +3 (159) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 106 / 81 / 3 / 2 | +6 [-4, +17] | -3 [-8, +1] | +2 [+0, +5] | 186, +7, 17 / 10 | 4 / 1 / 2 | +3 (158) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 101 / 88 / 3 / 0 | -12 [-24, +0] | +3 [+0, +7] | -1 [-3, +0] | 186, -15, 9 / 24 | 0 / 3 / 0 | -11 (162) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 62 / 46 / 1 / 83 | -9 [-17, -1] | -27 [-37, -18] | +27 [+18, +37] | 108, +2, 4 / 2 | 27 / 0 / 0 | +1 (94) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 105 / 81 / 4 / 2 | 4.8 | +5 [-5, +15] | -4 [-8, -1] | +2 [+0, +5] | 186, +7 [-2, +17], 16 / 9 | 4 / 0 / 2 | 2 / 2 / 0 / 0 | +3 (159) |
| conf.md.p1 | 92 / 97 / 2 / 1 | 4.3 | -8 [-17, +1] | -1 [-5, +2] | +1 [+0, +3] | 188, -8 [-17, +1], 6 / 14 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | -3 (171) |
| conf.rand.p1 | 102 / 87 / 3 / 0 | 5.3 | +2 [-4, +8] | -1 [-5, +2] | +0 [+0, +0] | 188, +1 [-4, +6], 4 / 3 | 2 / 1 / 2 | 0 / 2 / 1 / 0 | +0 (170) |
| conf.rand.m1 | 93 / 96 / 3 / 0 | 4.6 | -7 [-14, -1] | -1 [-3, +0] | +0 [+0, +0] | 189, -6 [-12, +0], 2 / 8 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -4 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +3 [-7, +13]; vs conf.rand.m1 +12 [+2, +23] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.1 pp [-4.2, -0.5]; moved out / in 4 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 159 questions; mean words 4.9 -> 4.8
KEY family=E judge=allenai layer=20 A=not_met B=not_met dR=+5 dR_F=NA dAns=-4 dAns_F=NA rand_margin=+3 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:38:55 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

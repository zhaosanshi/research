TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 47 / 35 / 2 / 108 | 4.1 | -14 [-23, -5] | -35 [-46, -25] | +35 [+25, +46] | 81, +1 [-4, +6], 4 / 3 | 36 / 1 / 1 | 15 / 21 / 0 / 1 | +0 (73) |
| x.w.m1 | 73 / 73 / 2 / 44 | 5.1 | +12 [+3, +21] | +29 [+20, +39] | -29 [-39, -20] | 117, +0 [-6, +6], 5 / 5 | 0 / 29 / 0 | 0 / 0 / 12 / 17 | +1 (102) |
| x.md.p1 | 55 / 55 / 3 / 79 | 3.8 | -6 [-13, +1] | -7 [-14, +0] | +6 [+0, +12] | 107, -2 [-7, +3], 2 / 4 | 10 / 3 / 2 | 6 / 4 / 2 / 1 | -1 (101) |
| x.rand.p1 | 62 / 52 / 2 / 76 | 4.5 | +1 [-4, +6] | -3 [-7, +0] | +3 [+0, +7] | 114, +2 [-3, +7], 4 / 2 | 3 / 0 / 0 | 1 / 2 / 0 / 0 | +3 (108) |
| x.rand.m1 | 59 / 60 / 2 / 71 | 4.6 | -2 [-7, +3] | +2 [-2, +6] | -2 [-6, +2] | 116, -2 [-6, +2], 1 / 3 | 1 / 3 / 0 | 1 / 0 / 1 / 2 | -1 (111) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 60 / 43 / 1 / 88; dR -11 [-20, -2]; answered-rate change -16.7 pp [-22.4, -11.5]; out / in 32 / 0; mean words 7.9 -> 6.8
- main effect: dR -14 (-7.3% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -15 [-24, -7]; vs x.rand.m1 -12 [-21, -4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -18.2 pp [-24.0, -13.0]; moved out / in 36 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +0 on 73 questions; mean words 4.8 -> 4.1
KEY family=E judge=allenai layer=18 A=not_met B=not_met dR=-14 dR_F=-11 dAns=-35 dAns_F=-32 rand_margin=-15

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 18

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 34 / 21 / 0 / 137 | -10 [-19, -1] | -38 [-49, -28] | +40 [+29, +51] | 55, +5, 6 / 1 | 38 / 0 / 0 | +2 (50) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 58 / 63 / 3 / 68 | +13 [+4, +22] | +37 [+27, +48] | -40 [-51, -29] | 84, -1, 5 / 6 | 0 / 37 / 0 | +0 (74) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 100 / 90 / 2 / 0 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 190, +0, 2 / 2 | 0 / 0 / 0 | +1 (185) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 96 / 94 / 2 / 0 | -4 [-13, +4] | +0 [-3, +3] | +0 [+0, +0] | 189, -5, 7 / 12 | 1 / 1 / 1 | -4 (179) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 91 / 95 / 5 / 1 | -9 [-16, -3] | -4 [-8, -1] | +1 [+0, +3] | 186, -7, 1 / 8 | 4 / 0 / 3 | -7 (174) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 4 / 0 | +0 [-6, +6] | -2 [-5, +0] | +0 [+0, +0] | 188, +1, 4 / 3 | 2 / 0 / 2 | +1 (176) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 103 / 84 / 3 / 2 | +3 [-8, +14] | -3 [-8, +1] | +2 [+0, +5] | 186, +5, 16 / 11 | 4 / 1 / 2 | +2 (159) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 102 / 85 / 4 / 1 | +2 [-9, +13] | -3 [-8, +1] | +1 [+0, +3] | 186, +4, 15 / 11 | 4 / 1 / 3 | +2 (158) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 94 / 94 / 4 / 0 | -19 [-30, -8] | +2 [-2, +6] | -1 [-3, +0] | 185, -20, 6 / 26 | 1 / 3 / 1 | -15 (162) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 60 / 43 / 1 / 88 | -11 [-20, -2] | -32 [-43, -22] | +32 [+22, +43] | 103, +3, 5 / 2 | 32 / 0 / 0 | +2 (88) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 18

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 103 / 84 / 3 / 2 | 4.7 | +3 [-8, +14] | -3 [-8, +1] | +2 [+0, +5] | 186, +5 [-5, +15], 16 / 11 | 4 / 1 / 2 | 3 / 1 / 1 / 0 | +2 (159) |
| conf.md.p1 | 96 / 94 / 2 / 0 | 4.1 | -4 [-13, +4] | +0 [-3, +3] | +0 [+0, +0] | 189, -5 [-14, +3], 7 / 12 | 1 / 1 / 1 | 0 / 1 / 1 / 0 | -4 (179) |
| conf.rand.p1 | 100 / 88 / 4 / 0 | 4.8 | +0 [-6, +6] | -2 [-5, +0] | +0 [+0, +0] | 188, +1 [-4, +6], 4 / 3 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | +1 (176) |
| conf.rand.m1 | 91 / 95 / 5 / 1 | 4.9 | -9 [-16, -3] | -4 [-8, -1] | +1 [+0, +3] | 186, -7 [-13, -2], 1 / 8 | 4 / 0 / 3 | 2 / 2 / 0 / 0 | -7 (174) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +3 [-8, +14]; vs conf.rand.m1 +12 [+2, +23] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.2, +0.5]; moved out / in 4 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 159 questions; mean words 4.9 -> 4.7
KEY family=E judge=allenai layer=18 A=not_met B=not_met dR=+3 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+3 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:36:28 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 50 / 36 / 2 / 104 | 4.2 | -11 [-19, -3] | -31 [-42, -21] | +31 [+21, +41] | 85, +2 [-3, +7], 4 / 2 | 32 / 1 / 1 | 13 / 19 / 0 / 1 | +0 (75) |
| x.w.m1 | 71 / 69 / 2 / 50 | 5.1 | +10 [+1, +19] | +23 [+15, +32] | -23 [-32, -15] | 117, -1 [-7, +5], 5 / 6 | 0 / 23 / 0 | 0 / 0 / 11 / 12 | -1 (100) |
| x.md.p1 | 61 / 55 / 3 / 73 | 4.3 | +0 [-7, +7] | -1 [-9, +8] | +0 [-8, +8] | 107, +1 [-2, +5], 2 / 1 | 10 / 9 / 1 | 5 / 5 / 4 / 5 | +0 (98) |
| x.rand.p1 | 60 / 57 / 3 / 72 | 4.6 | -1 [-6, +3] | +0 [-4, +4] | -1 [-5, +2] | 115, +0 [-4, +4], 2 / 2 | 2 / 2 / 1 | 1 / 1 / 0 / 2 | +1 (108) |
| x.rand.m1 | 62 / 52 / 2 / 76 | 4.9 | +1 [-3, +5] | -3 [-8, +1] | +3 [-1, +8] | 113, +2 [-2, +6], 3 / 1 | 4 / 1 / 0 | 1 / 3 / 0 / 1 | +1 (102) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 60 / 47 / 1 / 84; dR -11 [-20, -3]; answered-rate change -14.6 pp [-19.8, -9.9]; out / in 28 / 0; mean words 7.9 -> 6.8
- main effect: dR -11 (-5.7% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -10 [-19, -2]; vs x.rand.m1 -12 [-21, -4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -16.1 pp [-21.9, -10.9]; moved out / in 32 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +0 on 75 questions; mean words 4.8 -> 4.2
KEY family=E judge=allenai layer=17 A=not_met B=not_met dR=-11 dR_F=-11 dAns=-31 dAns_F=-28 rand_margin=-12

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 34 / 27 / 0 / 131 | -10 [-19, -2] | -32 [-43, -22] | +34 [+24, +45] | 60, +2, 4 / 2 | 33 / 1 / 0 | -1 (54) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 55 / 61 / 2 / 74 | +10 [+1, +20] | +32 [+22, +42] | -34 [-44, -24] | 84, -3, 5 / 8 | 0 / 32 / 0 | -2 (72) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 97 / 93 / 2 / 0 | -3 [-7, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -3, 0 / 3 | 0 / 0 / 0 | -3 (188) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 96 / 92 / 4 / 0 | -4 [-12, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, -4, 5 / 9 | 2 / 0 / 2 | -5 (175) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 97 / 91 / 4 / 0 | -3 [-10, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, -2, 4 / 6 | 2 / 0 / 2 | -3 (171) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 93 / 94 / 4 / 1 | -7 [-13, -2] | -3 [-7, +0] | +1 [+0, +3] | 187, -5, 1 / 6 | 3 / 0 / 2 | -2 (172) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 105 / 83 / 3 / 1 | +5 [-5, +15] | -2 [-6, +2] | +1 [+0, +3] | 187, +6, 15 / 9 | 3 / 1 / 2 | +0 (165) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 104 / 84 / 3 / 1 | +4 [-6, +14] | -2 [-6, +2] | +1 [+0, +3] | 187, +5, 15 / 10 | 3 / 1 / 2 | +0 (165) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 95 / 93 / 4 / 0 | -18 [-29, -7] | +2 [+0, +5] | -1 [-3, +0] | 186, -20, 5 / 25 | 0 / 2 / 0 | -15 (161) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 60 / 47 / 1 / 84 | -11 [-20, -3] | -28 [-38, -19] | +28 [+19, +38] | 107, +2, 4 / 2 | 28 / 0 / 0 | +1 (93) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 105 / 83 / 3 / 1 | 4.7 | +5 [-5, +15] | -2 [-6, +2] | +1 [+0, +3] | 187, +6 [-3, +16], 15 / 9 | 3 / 1 / 2 | 2 / 1 / 1 / 0 | +0 (165) |
| conf.md.p1 | 96 / 92 / 4 / 0 | 4.0 | -4 [-12, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, -4 [-12, +3], 5 / 9 | 2 / 0 / 2 | 0 / 2 / 0 / 0 | -5 (175) |
| conf.rand.p1 | 93 / 94 / 4 / 1 | 4.6 | -7 [-13, -2] | -3 [-7, +0] | +1 [+0, +3] | 187, -5 [-10, +0], 1 / 6 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | -2 (172) |
| conf.rand.m1 | 97 / 91 / 4 / 0 | 5.3 | -3 [-10, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, -2 [-8, +4], 4 / 6 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | -3 (171) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +12 [+2, +23]; vs conf.rand.m1 +8 [-1, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-3.1, +1.0]; moved out / in 3 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +0 on 165 questions; mean words 4.9 -> 4.7
KEY family=E judge=allenai layer=17 A=not_met B=not_met dR=+5 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+8 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:39:22 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

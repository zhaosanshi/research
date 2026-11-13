TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 23

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 65 / 46 / 3 / 78 | 4.6 | +4 [-4, +12] | -6 [-13, +0] | +5 [-1, +12] | 108, +5 [-1, +11], 7 / 2 | 9 / 3 / 1 | 4 / 5 / 3 / 0 | +1 (93) |
| x.w.m1 | 66 / 56 / 2 / 68 | 5.1 | +5 [-1, +12] | +5 [+0, +10] | -5 [-10, +0] | 116, +6 [+0, +12], 8 / 2 | 1 / 6 / 0 | 1 / 0 / 0 / 6 | +5 (103) |
| x.md.p1 | 62 / 51 / 3 / 76 | 4.4 | +1 [-5, +7] | -4 [-10, +1] | +3 [-2, +8] | 111, +2 [-3, +7], 4 / 2 | 6 / 2 / 1 | 3 / 3 / 2 / 0 | +3 (100) |
| x.rand.p1 | 61 / 55 / 2 / 74 | 4.8 | +0 [-5, +5] | -1 [-5, +2] | +1 [-2, +5] | 115, +1 [-3, +6], 3 / 2 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +1 (103) |
| x.rand.m1 | 59 / 55 / 3 / 75 | 4.5 | -2 [-6, +2] | -3 [-7, +0] | +2 [+0, +5] | 114, -1 [-5, +2], 1 / 2 | 3 / 0 / 1 | 1 / 2 / 0 / 0 | -1 (110) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 67 / 60 / 1 / 64; dR -4 [-11, +3]; answered-rate change -4.2 pp [-7.3, -1.0]; out / in 9 / 1; mean words 7.9 -> 7.7
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +4 [-4, +12]; vs x.rand.m1 +6 [-2, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.1 pp [-6.8, +0.0]; moved out / in 9 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +1 on 93 questions; mean words 4.8 -> 4.6
KEY family=E judge=allenai layer=23 A=not_met B=not_met dR=+4 dR_F=-4 dAns=-6 dAns_F=-8 rand_margin=+4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 23

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 47 / 43 / 1 / 101 | +3 [-3, +10] | -3 [-9, +3] | +4 [-2, +10] | 87, +1, 4 / 3 | 6 / 3 / 0 | -1 (78) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 47 / 44 / 0 / 101 | +2 [-5, +9] | +7 [+0, +15] | -7 [-15, +0] | 80, +0, 3 / 3 | 4 / 11 / 0 | +0 (72) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 100 / 90 / 2 / 0 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 190, +0, 1 / 1 | 0 / 0 / 0 | +0 (187) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-8, +8] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 9 / 8 | 2 / 0 / 1 | +2 (174) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 91 / 3 / 0 | -2 [-7, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 2 / 4 | 1 / 0 / 1 | -2 (181) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 94 / 95 / 2 / 1 | -6 [-12, -1] | -1 [-3, +0] | +1 [+0, +3] | 189, -5, 1 / 6 | 1 / 0 / 0 | -6 (177) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 105 / 83 / 3 / 1 | +5 [-3, +14] | -2 [-5, +0] | +1 [+0, +3] | 188, +6, 12 / 6 | 2 / 0 / 1 | +4 (162) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 102 / 86 / 3 / 1 | +2 [-6, +10] | -2 [-5, +0] | +1 [+0, +3] | 188, +3, 9 / 6 | 2 / 0 / 1 | +2 (163) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 105 / 84 / 3 / 0 | -8 [-19, +3] | +3 [-1, +8] | -1 [-3, +0] | 185, -10, 8 / 18 | 1 / 4 / 1 | -6 (159) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 67 / 60 / 1 / 64 | -4 [-11, +3] | -8 [-14, -2] | +8 [+2, +14] | 126, +0, 5 / 5 | 9 / 1 / 0 | -2 (109) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 23

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 105 / 83 / 3 / 1 | 4.8 | +5 [-3, +14] | -2 [-5, +0] | +1 [+0, +3] | 188, +6 [-2, +14], 12 / 6 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +4 (162) |
| conf.md.p1 | 100 / 88 / 3 / 1 | 4.3 | +0 [-8, +8] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-7, +9], 9 / 8 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +2 (174) |
| conf.rand.p1 | 94 / 95 / 2 / 1 | 5.0 | -6 [-12, -1] | -1 [-3, +0] | +1 [+0, +3] | 189, -5 [-11, +0], 1 / 6 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | -6 (177) |
| conf.rand.m1 | 98 / 91 / 3 / 0 | 4.7 | -2 [-7, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-7, +3], 2 / 4 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -2 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +11 [+2, +20]; vs conf.rand.m1 +7 [-3, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +4 on 162 questions; mean words 4.9 -> 4.8
KEY family=E judge=allenai layer=23 A=not_met B=not_met dR=+5 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+7 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:42:31 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 55 / 40 / 1 / 96 | 4.4 | -6 [-14, +2] | -22 [-31, -13] | +23 [+15, +32] | 94, +4 [-1, +10], 6 / 2 | 23 / 1 / 0 | 10 / 13 / 0 / 1 | +2 (83) |
| x.w.m1 | 70 / 63 / 2 / 57 | 5.0 | +9 [+1, +17] | +16 [+9, +24] | -16 [-24, -9] | 117, +2 [-4, +8], 6 / 4 | 0 / 16 / 0 | 0 / 0 / 7 / 9 | +3 (101) |
| x.md.p1 | 60 / 57 / 3 / 72 | 5.1 | -1 [-6, +4] | +0 [-6, +6] | -1 [-6, +4] | 113, +0 [-4, +4], 2 / 2 | 4 / 4 / 1 | 2 / 2 / 1 / 3 | +0 (101) |
| x.rand.p1 | 59 / 54 / 2 / 77 | 4.8 | -2 [-8, +4] | -4 [-10, +1] | +4 [-1, +10] | 111, -1 [-5, +3], 2 / 3 | 6 / 2 / 0 | 2 / 4 / 1 / 1 | -2 (102) |
| x.rand.m1 | 64 / 53 / 3 / 72 | 4.6 | +3 [-2, +8] | +0 [-4, +4] | -1 [-4, +2] | 115, +4 [+0, +9], 5 / 1 | 2 / 2 / 1 | 1 / 1 / 0 / 2 | +3 (106) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 66 / 50 / 0 / 76; dR -5 [-14, +4]; answered-rate change -9.9 pp [-14.6, -5.7]; out / in 20 / 1; mean words 7.9 -> 7.2
- main effect: dR -6 (-3.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -4 [-12, +4]; vs x.rand.m1 -9 [-18, +0] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -11.5 pp [-16.1, -6.8]; moved out / in 23 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +2 on 83 questions; mean words 4.8 -> 4.4
KEY family=E judge=allenai layer=15 A=not_met B=not_met dR=-6 dR_F=-5 dAns=-22 dAns_F=-19 rand_margin=-9

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 15

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 42 / 30 / 0 / 120 | -2 [-10, +6] | -21 [-30, -12] | +23 [+14, +32] | 71, +6, 7 / 1 | 22 / 1 / 0 | +3 (66) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 55 / 59 / 1 / 77 | +10 [+1, +19] | +30 [+20, +40] | -31 [-41, -21] | 84, -3, 3 / 6 | 0 / 30 / 0 | -1 (75) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-7, +3] | +0 [+0, +0] | +0 [+0, +0] | 190, -2, 2 / 4 | 0 / 0 / 0 | -1 (187) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 88 / 99 / 4 / 1 | -12 [-19, -5] | -3 [-8, +1] | +1 [+0, +3] | 186, -11, 0 / 11 | 4 / 1 / 3 | -8 (170) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 96 / 92 / 4 / 0 | -4 [-9, +0] | -2 [-5, +0] | +0 [+0, +0] | 188, -2, 1 / 3 | 2 / 0 / 2 | -2 (177) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 92 / 96 / 3 / 1 | -8 [-16, -1] | -2 [-5, +0] | +1 [+0, +3] | 188, -6, 3 / 9 | 2 / 0 / 1 | -5 (176) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 104 / 84 / 2 / 2 | +4 [-7, +15] | -2 [-7, +3] | +2 [+0, +5] | 186, +5, 15 / 10 | 4 / 2 / 2 | +1 (160) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 106 / 82 / 2 / 2 | +6 [-4, +16] | -2 [-7, +3] | +2 [+0, +5] | 186, +7, 14 / 7 | 4 / 2 / 2 | +3 (161) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 97 / 92 / 3 / 0 | -16 [-27, -5] | +3 [+0, +7] | -1 [-3, +0] | 186, -19, 5 / 24 | 0 / 3 / 0 | -17 (162) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 66 / 50 / 0 / 76 | -5 [-14, +4] | -19 [-28, -11] | +20 [+12, +29] | 115, +5, 8 / 3 | 20 / 1 / 0 | +1 (96) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 15

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 104 / 84 / 2 / 2 | 5.2 | +4 [-7, +15] | -2 [-7, +3] | +2 [+0, +5] | 186, +5 [-5, +15], 15 / 10 | 4 / 2 / 2 | 3 / 1 / 2 / 0 | +1 (160) |
| conf.md.p1 | 88 / 99 / 4 / 1 | 4.9 | -12 [-19, -5] | -3 [-8, +1] | +1 [+0, +3] | 186, -11 [-18, -5], 0 / 11 | 4 / 1 / 3 | 2 / 2 / 1 / 0 | -8 (170) |
| conf.rand.p1 | 92 / 96 / 3 / 1 | 5.2 | -8 [-16, -1] | -2 [-5, +0] | +1 [+0, +3] | 188, -6 [-13, +1], 3 / 9 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | -5 (176) |
| conf.rand.m1 | 96 / 92 / 4 / 0 | 4.8 | -4 [-9, +0] | -2 [-5, +0] | +0 [+0, +0] | 188, -2 [-6, +2], 1 / 3 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | -2 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +12 [+2, +22]; vs conf.rand.m1 +8 [-2, +19] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-3.6, +1.6]; moved out / in 4 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +1 on 160 questions; mean words 4.9 -> 5.2
KEY family=E judge=allenai layer=15 A=not_met B=not_met dR=+4 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+8 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:36:58 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

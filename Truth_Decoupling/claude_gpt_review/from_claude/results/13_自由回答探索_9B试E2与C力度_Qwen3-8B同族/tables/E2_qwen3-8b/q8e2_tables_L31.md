TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 31

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 67 / 49 / 2 / 74 | 4.7 | +6 [+1, +12] | -1 [-5, +2] | +1 [-2, +5] | 115, +6 [+2, +11], 6 / 0 | 2 / 1 / 0 | 1 / 1 / 1 / 0 | +4 (108) |
| x.w.m1 | 60 / 57 / 2 / 73 | 4.6 | -1 [-5, +3] | +0 [-3, +3] | +0 [-3, +3] | 116, +0 [-4, +4], 2 / 2 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | -1 (109) |
| x.md.p1 | 62 / 55 / 2 / 73 | 4.7 | +1 [-4, +6] | +0 [-4, +4] | +0 [-4, +4] | 115, +1 [-3, +5], 3 / 2 | 2 / 2 / 0 | 1 / 1 / 1 / 1 | +1 (105) |
| x.rand.p1 | 63 / 55 / 2 / 72 | 4.9 | +2 [+0, +5] | +1 [+0, +3] | -1 [-3, +0] | 117, +2 [+0, +5], 2 / 0 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (113) |
| x.rand.m1 | 61 / 55 / 2 / 74 | 4.6 | +0 [-3, +3] | -1 [-3, +0] | +1 [+0, +3] | 116, +1 [+0, +3], 1 / 0 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +1 (115) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 72 / 61 / 2 / 57; dR +1 [-3, +5]; answered-rate change -1.0 pp [-2.6, +0.0]; out / in 2 / 0; mean words 7.9 -> 7.7
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +4 [-1, +10]; vs x.rand.m1 +6 [+1, +12] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +4 on 108 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=31 A=met B=not_met dR=+6 dR_F=+1 dAns=-1 dAns_F=-2 rand_margin=+4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 31

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 43 / 47 / 2 / 100 | -1 [-3, +0] | -3 [-7, +0] | +3 [+0, +7] | 90, -1, 0 / 1 | 3 / 0 / 0 | +0 (86) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 48 / 39 / 0 / 105 | +3 [-3, +9] | +3 [-3, +9] | -3 [-9, +3] | 81, +4, 4 / 0 | 3 / 6 / 0 | +2 (77) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-6, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -2, 1 / 3 | 0 / 0 / 0 | -2 (190) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 101 / 87 / 4 / 0 | +1 [-6, +8] | -2 [-5, +0] | +0 [+0, +0] | 188, +1, 7 / 6 | 2 / 0 / 2 | +1 (179) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -2, 0 / 2 | 0 / 0 / 0 | -1 (184) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 97 / 92 / 3 / 0 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -3, 1 / 4 | 1 / 0 / 1 | -3 (186) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 100 / 87 / 5 / 0 | +0 [-5, +5] | -3 [-7, +0] | +0 [+0, +0] | 187, +1, 3 / 2 | 3 / 0 / 3 | +2 (178) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 103 / 85 / 4 / 0 | +3 [-2, +8] | -2 [-5, +0] | +0 [+0, +0] | 188, +4, 5 / 1 | 2 / 0 / 2 | +4 (181) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 107 / 80 / 4 / 1 | -6 [-12, +0] | +1 [+0, +3] | +0 [+0, +0] | 186, -7, 1 / 8 | 0 / 1 / 0 | -5 (172) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 72 / 61 / 2 / 57 | +1 [-3, +5] | -2 [-5, +0] | +1 [+0, +3] | 133, +3, 3 / 0 | 2 / 0 / 1 | +3 (131) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 31

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 87 / 5 / 0 | 4.7 | +0 [-5, +5] | -3 [-7, +0] | +0 [+0, +0] | 187, +1 [-3, +5], 3 / 2 | 3 / 0 / 3 | 1 / 2 / 0 / 0 | +2 (178) |
| conf.md.p1 | 101 / 87 / 4 / 0 | 4.8 | +1 [-6, +8] | -2 [-5, +0] | +0 [+0, +0] | 188, +1 [-6, +8], 7 / 6 | 2 / 0 / 2 | 0 / 2 / 0 / 0 | +1 (179) |
| conf.rand.p1 | 97 / 92 / 3 / 0 | 4.8 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -3 [-8, +1], 1 / 4 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -3 (186) |
| conf.rand.m1 | 98 / 92 / 2 / 0 | 4.9 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -2 [-5, +0], 0 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +3 [-2, +8]; vs conf.rand.m1 +2 [-3, +8] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 178 questions; mean words 4.9 -> 4.7
KEY family=E judge=allenai layer=31 A=not_met B=not_met dR=+0 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+2 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:56:13 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

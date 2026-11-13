TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 35

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 71 / 49 / 2 / 70 | 4.5 | +10 [+4, +16] | +3 [-2, +8] | -3 [-8, +2] | 115, +6 [+2, +11], 6 / 0 | 2 / 5 / 0 | 0 / 2 / 4 / 1 | +3 (110) |
| x.w.m1 | 59 / 56 / 2 / 75 | 4.8 | -2 [-6, +2] | -2 [-5, +0] | +2 [+0, +5] | 115, -1 [-5, +2], 1 / 2 | 2 / 0 / 0 | 1 / 1 / 0 / 0 | +0 (104) |
| x.md.p1 | 71 / 55 / 2 / 64 | 4.7 | +10 [+4, +16] | +9 [+4, +15] | -9 [-15, -4] | 117, +3 [+0, +7], 3 / 0 | 0 / 9 / 0 | 0 / 0 / 7 / 2 | +1 (110) |
| x.rand.p1 | 62 / 56 / 2 / 72 | 4.7 | +1 [-2, +5] | +1 [+0, +3] | -1 [-3, +0] | 117, +1 [-2, +5], 2 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +2 (116) |
| x.rand.m1 | 62 / 55 / 2 / 73 | 4.7 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 117, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (115) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 73 / 61 / 2 / 56; dR +2 [-2, +6]; answered-rate change -0.5 pp [-1.6, +0.0]; out / in 1 / 0; mean words 7.9 -> 7.5
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +9 [+2, +16]; vs x.rand.m1 +9 [+4, +15] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +1.6 pp [-1.0, +4.2]; moved out / in 2 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 110 questions; mean words 4.8 -> 4.5
KEY family=E judge=allenai layer=35 A=met B=not_met dR=+10 dR_F=+2 dAns=+3 dAns_F=-1 rand_margin=+9

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 35

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 54 / 43 / 1 / 94 | +10 [+4, +16] | +4 [-1, +10] | -3 [-8, +2] | 91, +5, 5 / 0 | 2 / 6 / 0 | +3 (87) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 44 / 36 / 0 / 112 | -1 [-7, +5] | -4 [-9, +0] | +4 [+0, +9] | 79, +2, 4 / 2 | 5 / 1 / 0 | +1 (72) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 3 / 0 | -2 [-6, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 1 / 3 | 1 / 0 / 1 | -2 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-8, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 5 / 5 | 2 / 0 / 1 | +0 (177) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-6, +3] | +0 [+0, +0] | +0 [+0, +0] | 190, -1, 2 / 3 | 0 / 0 / 0 | +0 (184) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 3 / 0 | -2 [-5, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 0 / 2 | 1 / 0 / 1 | -2 (185) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 103 / 85 / 4 / 0 | +3 [-3, +9] | -2 [-5, +0] | +0 [+0, +0] | 188, +4, 6 / 2 | 2 / 0 / 2 | +3 (182) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 105 / 83 / 4 / 0 | +5 [-1, +12] | -2 [-5, +0] | +0 [+0, +0] | 188, +6, 8 / 2 | 2 / 0 / 2 | +5 (181) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 106 / 81 / 4 / 1 | -7 [-16, +2] | +1 [+0, +3] | +0 [+0, +0] | 186, -8, 6 / 14 | 0 / 1 / 0 | -5 (167) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 73 / 61 / 2 / 56 | +2 [-2, +6] | -1 [-3, +0] | +0 [+0, +0] | 134, +3, 3 / 0 | 1 / 0 / 1 | +3 (131) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 35

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 103 / 85 / 4 / 0 | 4.4 | +3 [-3, +9] | -2 [-5, +0] | +0 [+0, +0] | 188, +4 [-1, +10], 6 / 2 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | +3 (182) |
| conf.md.p1 | 99 / 89 / 3 / 1 | 4.8 | -1 [-8, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, +0 [-6, +6], 5 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +0 (177) |
| conf.rand.p1 | 98 / 91 / 3 / 0 | 5.0 | -2 [-5, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-5, +0], 0 / 2 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -2 (185) |
| conf.rand.m1 | 99 / 91 / 2 / 0 | 4.8 | -1 [-6, +3] | +0 [+0, +0] | +0 [+0, +0] | 190, -1 [-6, +3], 2 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (184) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +5 [-1, +12]; vs conf.rand.m1 +4 [-1, +10] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +3 on 182 questions; mean words 4.9 -> 4.4
KEY family=E judge=allenai layer=35 A=not_met B=not_met dR=+3 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+4 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 15:00:21 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 26

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 65 / 52 / 2 / 73 | 4.6 | +4 [-2, +10] | +0 [-4, +4] | +0 [-4, +4] | 115, +3 [-2, +8], 5 / 2 | 2 / 2 / 0 | 1 / 1 / 2 / 0 | +3 (104) |
| x.w.m1 | 63 / 52 / 3 / 74 | 4.8 | +2 [-3, +7] | -2 [-6, +2] | +1 [-2, +5] | 114, +3 [-1, +8], 4 / 1 | 3 / 1 / 1 | 1 / 2 / 0 / 1 | +2 (105) |
| x.md.p1 | 60 / 55 / 3 / 74 | 4.5 | -1 [-6, +4] | -2 [-7, +3] | +1 [-3, +5] | 113, +0 [-4, +4], 2 / 2 | 4 / 2 / 1 | 2 / 2 / 1 / 1 | +1 (103) |
| x.rand.p1 | 60 / 56 / 2 / 74 | 5.0 | -1 [-5, +2] | -1 [-3, +0] | +1 [+0, +3] | 116, +0 [-3, +3], 1 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | -1 (106) |
| x.rand.m1 | 62 / 56 / 2 / 72 | 4.8 | +1 [-2, +5] | +1 [+0, +3] | -1 [-3, +0] | 117, +1 [-2, +5], 2 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (112) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 74 / 62 / 2 / 54; dR +3 [-2, +9]; answered-rate change +0.5 pp [-1.0, +2.6]; out / in 1 / 2; mean words 7.9 -> 7.8
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-1, +11]; vs x.rand.m1 +3 [-4, +10] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-2.1, +2.1]; moved out / in 2 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +3 on 104 questions; mean words 4.8 -> 4.6
KEY family=E judge=allenai layer=26 A=not_met B=not_met dR=+4 dR_F=+3 dAns=+0 dAns_F=+1 rand_margin=+3

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 26

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 48 / 46 / 1 / 97 | +4 [-1, +10] | +1 [-3, +5] | +0 [-4, +4] | 91, +2, 3 / 1 | 2 / 3 / 0 | +2 (85) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 44 / 40 / 0 / 108 | -1 [-7, +5] | +0 [-6, +6] | +0 [-6, +6] | 79, +2, 3 / 1 | 5 / 5 / 0 | +1 (74) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-6, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -2, 1 / 3 | 0 / 0 / 0 | -1 (188) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 98 / 89 / 4 / 1 | -2 [-11, +7] | -3 [-7, +0] | +1 [+0, +3] | 187, +0, 9 / 9 | 3 / 0 / 2 | -2 (175) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 90 / 3 / 1 | -2 [-8, +3] | -2 [-5, +0] | +1 [+0, +3] | 188, -1, 3 / 4 | 2 / 0 / 1 | -1 (179) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 97 / 93 / 2 / 0 | -3 [-8, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -3, 2 / 5 | 0 / 0 / 0 | -3 (182) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 102 / 85 / 4 / 1 | +2 [-5, +10] | -3 [-7, +0] | +1 [+0, +3] | 187, +4, 8 / 4 | 3 / 0 / 2 | +4 (171) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 99 / 88 / 4 / 1 | -1 [-8, +6] | -3 [-7, +0] | +1 [+0, +3] | 187, +1, 6 / 5 | 3 / 0 / 2 | +1 (169) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 104 / 82 / 4 / 2 | -9 [-18, +0] | +0 [-3, +3] | +1 [+0, +3] | 185, -9, 6 / 15 | 1 / 1 / 0 | -8 (163) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 74 / 62 / 2 / 54 | +3 [-2, +9] | +1 [-2, +5] | -2 [-5, +0] | 134, +3, 5 / 2 | 1 / 2 / 1 | +2 (124) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 26

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 102 / 85 / 4 / 1 | 4.5 | +2 [-5, +10] | -3 [-7, +0] | +1 [+0, +3] | 187, +4 [-3, +11], 8 / 4 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +4 (171) |
| conf.md.p1 | 98 / 89 / 4 / 1 | 4.3 | -2 [-11, +7] | -3 [-7, +0] | +1 [+0, +3] | 187, +0 [-8, +8], 9 / 9 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | -2 (175) |
| conf.rand.p1 | 97 / 93 / 2 / 0 | 5.0 | -3 [-8, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -3 [-8, +2], 2 / 5 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -3 (182) |
| conf.rand.m1 | 98 / 90 / 3 / 1 | 4.8 | -2 [-8, +3] | -2 [-5, +0] | +1 [+0, +3] | 188, -1 [-6, +4], 3 / 4 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (179) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +2 (1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +5 [-3, +13]; vs conf.rand.m1 +4 [-3, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +4 on 171 questions; mean words 4.9 -> 4.5
KEY family=E judge=allenai layer=26 A=not_met B=not_met dR=+2 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+4 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:45:55 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 34

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 63 / 53 / 2 / 74 | 4.6 | +2 [+0, +5] | -1 [-3, +0] | +1 [+0, +3] | 116, +2 [+0, +5], 2 / 0 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +2 (115) |
| x.w.m1 | 60 / 57 / 2 / 73 | 4.7 | -1 [-5, +2] | +0 [-3, +3] | +0 [-3, +3] | 116, +0 [-3, +3], 1 / 1 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +0 (114) |
| x.md.p1 | 69 / 55 / 2 / 66 | 4.9 | +8 [+3, +14] | +7 [+2, +12] | -7 [-12, -2] | 117, +3 [+0, +7], 3 / 0 | 0 / 7 / 0 | 0 / 0 / 5 / 2 | +1 (111) |
| x.rand.p1 | 61 / 56 / 2 / 73 | 4.8 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (117) |
| x.rand.m1 | 61 / 56 / 2 / 73 | 4.8 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (116) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 71 / 63 / 2 / 56; dR +0 [-3, +3]; answered-rate change -0.5 pp [-1.6, +0.0]; out / in 1 / 0; mean words 7.9 -> 7.7
- main effect: dR +2 (1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +2 [+0, +5]; vs x.rand.m1 +2 [+0, +5] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +2 on 115 questions; mean words 4.8 -> 4.6
KEY family=E judge=allenai layer=34 A=not_met B=not_met dR=+2 dR_F=+0 dAns=-1 dAns_F=-1 rand_margin=+2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 34

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 47 / 44 / 2 / 99 | +3 [+0, +7] | -2 [-6, +2] | +2 [-2, +6] | 90, +2, 2 / 0 | 3 / 1 / 0 | +1 (88) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 44 / 41 / 0 / 107 | -1 [-4, +2] | +1 [-2, +4] | -1 [-4, +2] | 83, +0, 1 / 1 | 1 / 2 / 0 | +0 (81) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 102 / 87 / 3 / 0 | +2 [-2, +6] | -1 [-3, +0] | +0 [+0, +0] | 189, +2, 3 / 1 | 1 / 0 / 1 | +1 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 97 / 92 / 3 / 0 | -3 [-10, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, -3, 4 / 7 | 1 / 0 / 1 | -3 (177) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 100 / 90 / 2 / 0 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, +0, 0 / 0 | 0 / 0 / 0 | +0 (188) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 102 / 88 / 2 / 0 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, +2, 2 / 0 | 0 / 0 / 0 | +2 (188) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 103 / 86 / 3 / 0 | +3 [-1, +8] | -1 [-3, +0] | +0 [+0, +0] | 189, +3, 4 / 1 | 1 / 0 / 1 | +2 (185) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 102 / 87 / 3 / 0 | +2 [-3, +7] | -1 [-3, +0] | +0 [+0, +0] | 189, +2, 4 / 2 | 1 / 0 / 1 | +1 (183) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 110 / 76 / 5 / 1 | -3 [-8, +2] | +0 [+0, +0] | +0 [+0, +0] | 186, -3, 2 / 5 | 0 / 0 / 0 | -2 (181) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 71 / 63 / 2 / 56 | +0 [-3, +3] | -1 [-3, +0] | +0 [+0, +0] | 134, +1, 1 / 0 | 1 / 0 / 1 | +1 (134) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 34

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 103 / 86 / 3 / 0 | 4.8 | +3 [-1, +8] | -1 [-3, +0] | +0 [+0, +0] | 189, +3 [-1, +8], 4 / 1 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +2 (185) |
| conf.md.p1 | 97 / 92 / 3 / 0 | 4.9 | -3 [-10, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, -3 [-10, +3], 4 / 7 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -3 (177) |
| conf.rand.p1 | 102 / 88 / 2 / 0 | 4.7 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, +2 [+0, +5], 2 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +2 (188) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 4.9 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (188) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +1 [-2, +4]; vs conf.rand.m1 +3 [-1, +8] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +2 on 185 questions; mean words 4.9 -> 4.8
KEY family=E judge=allenai layer=34 A=not_met B=not_met dR=+3 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+1 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:59:17 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

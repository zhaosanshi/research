TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 29

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 62 / 54 / 2 / 74 | 4.8 | +1 [-3, +5] | -1 [-3, +0] | +1 [+0, +3] | 116, +2 [-2, +6], 3 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +1 (113) |
| x.w.m1 | 61 / 56 / 2 / 73 | 4.7 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (117) |
| x.md.p1 | 60 / 58 / 2 / 72 | 4.7 | -1 [-7, +5] | +1 [-3, +5] | -1 [-5, +3] | 115, -1 [-6, +4], 3 / 4 | 2 / 3 / 0 | 1 / 1 / 1 / 2 | +0 (103) |
| x.rand.p1 | 61 / 56 / 2 / 73 | 4.8 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (117) |
| x.rand.m1 | 62 / 55 / 2 / 73 | 4.7 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 117, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (117) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 70 / 65 / 2 / 55; dR -1 [-5, +2]; answered-rate change +0.0 pp [-1.6, +1.6]; out / in 1 / 1; mean words 7.9 -> 7.8
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +1 [-3, +5]; vs x.rand.m1 +0 [-5, +5] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 113 questions; mean words 4.8 -> 4.8
KEY family=E judge=allenai layer=29 A=not_met B=not_met dR=+1 dR_F=-1 dAns=-1 dAns_F=+0 rand_margin=+0

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 29

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 44 / 48 / 1 / 99 | +0 [-3, +3] | -1 [-5, +2] | +2 [+0, +5] | 91, -1, 0 / 1 | 2 / 1 / 0 | +0 (86) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 45 / 37 / 0 / 110 | +0 [-4, +4] | -2 [-5, +0] | +2 [+0, +5] | 82, +2, 2 / 0 | 2 / 0 / 0 | +1 (79) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 100 / 90 / 2 / 0 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 190, +0, 1 / 1 | 0 / 0 / 0 | +0 (187) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 98 / 89 / 4 / 1 | -2 [-10, +5] | -3 [-7, +0] | +1 [+0, +3] | 187, -1, 6 / 7 | 3 / 0 / 2 | -1 (176) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 92 / 2 / 0 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -2, 0 / 2 | 0 / 0 / 0 | -2 (187) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 101 / 88 / 3 / 0 | +1 [+0, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, +1, 1 / 0 | 1 / 0 / 1 | +1 (186) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 102 / 87 / 2 / 1 | +2 [-2, +6] | -1 [-3, +0] | +1 [+0, +3] | 189, +3, 3 / 0 | 1 / 0 / 0 | +3 (183) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 100 / 89 / 3 / 0 | +0 [-5, +5] | -1 [-3, +0] | +0 [+0, +0] | 189, +0, 3 / 3 | 1 / 0 / 1 | +0 (185) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 113 / 73 / 5 / 1 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 186, +0, 1 / 1 | 0 / 0 / 0 | -1 (181) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 70 / 65 / 2 / 55 | -1 [-5, +2] | +0 [-3, +3] | -1 [-3, +0] | 134, +0, 1 / 1 | 1 / 1 / 1 | +0 (134) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 29

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 102 / 87 / 2 / 1 | 4.9 | +2 [-2, +6] | -1 [-3, +0] | +1 [+0, +3] | 189, +3 [+0, +7], 3 / 0 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +3 (183) |
| conf.md.p1 | 98 / 89 / 4 / 1 | 4.5 | -2 [-10, +5] | -3 [-7, +0] | +1 [+0, +3] | 187, -1 [-8, +6], 6 / 7 | 3 / 0 / 2 | 1 / 2 / 0 / 0 | -1 (176) |
| conf.rand.p1 | 101 / 88 / 3 / 0 | 4.9 | +1 [+0, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, +1 [+0, +3], 1 / 0 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +1 (186) |
| conf.rand.m1 | 98 / 92 / 2 / 0 | 4.9 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -2 [-5, +0], 0 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -2 (187) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +2 (1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +1 [-2, +5]; vs conf.rand.m1 +4 [+0, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +3 on 183 questions; mean words 4.9 -> 4.9
KEY family=E judge=allenai layer=29 A=not_met B=not_met dR=+2 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+1 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:54:10 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

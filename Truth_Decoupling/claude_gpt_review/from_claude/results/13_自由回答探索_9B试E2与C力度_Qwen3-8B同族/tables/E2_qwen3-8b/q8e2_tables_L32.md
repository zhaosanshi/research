TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 32

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 67 / 50 / 2 / 73 | 4.7 | +6 [+0, +12] | +0 [-4, +4] | +0 [-4, +4] | 115, +5 [+0, +10], 6 / 1 | 2 / 2 / 0 | 1 / 1 / 2 / 0 | +4 (107) |
| x.w.m1 | 61 / 57 / 2 / 72 | 4.7 | +0 [-4, +4] | +1 [-2, +4] | -1 [-4, +2] | 116, +1 [-2, +5], 2 / 1 | 1 / 2 / 0 | 1 / 0 / 0 / 2 | +0 (108) |
| x.md.p1 | 61 / 56 / 2 / 73 | 4.7 | +0 [-6, +6] | +0 [-4, +4] | +0 [-4, +4] | 115, -1 [-6, +3], 2 / 3 | 2 / 2 / 0 | 1 / 1 / 2 / 0 | +0 (106) |
| x.rand.p1 | 63 / 54 / 2 / 73 | 4.7 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 117, +2 [+0, +5], 2 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (115) |
| x.rand.m1 | 60 / 56 / 2 / 74 | 4.6 | -1 [-5, +2] | -1 [-3, +0] | +1 [+0, +3] | 116, +0 [-3, +3], 1 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (115) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 71 / 62 / 2 / 57; dR +0 [-4, +4]; answered-rate change -1.0 pp [-2.6, +0.0]; out / in 2 / 0; mean words 7.9 -> 7.7
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +4 [-1, +10]; vs x.rand.m1 +7 [+0, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-2.1, +2.1]; moved out / in 2 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +4 on 107 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=32 A=not_met B=not_met dR=+6 dR_F=+0 dAns=+0 dAns_F=-2 rand_margin=+4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 32

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 44 / 46 / 2 / 100 | +0 [-3, +3] | -3 [-7, +0] | +3 [+0, +7] | 90, +0, 1 / 1 | 3 / 0 / 0 | +1 (86) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 48 / 37 / 0 / 107 | +3 [-3, +9] | +1 [-4, +6] | -1 [-6, +4] | 81, +4, 4 / 0 | 3 / 4 / 0 | +2 (75) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 99 / 91 / 2 / 0 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 190, -1, 1 / 2 | 0 / 0 / 0 | -1 (190) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 3 / 0 | -2 [-8, +4] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 4 / 6 | 1 / 0 / 1 | -3 (184) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 97 / 92 / 3 / 0 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 1 / 3 | 1 / 0 / 1 | -2 (186) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 101 / 88 / 3 / 0 | +1 [+0, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, +1, 1 / 0 | 1 / 0 / 1 | +1 (184) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-7, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 4 / 4 | 2 / 0 / 1 | +1 (181) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-6, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 5 / 4 | 2 / 0 / 1 | +2 (182) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 107 / 79 / 5 / 1 | -6 [-12, +0] | +0 [+0, +0] | +0 [+0, +0] | 186, -6, 2 / 8 | 0 / 0 / 0 | -5 (172) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 71 / 62 / 2 / 57 | +0 [-4, +4] | -2 [-5, +0] | +1 [+0, +3] | 133, +2, 2 / 0 | 2 / 0 / 1 | +2 (132) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 32

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 99 / 89 / 3 / 1 | 4.7 | -1 [-7, +5] | -2 [-5, +0] | +1 [+0, +3] | 188, +0 [-6, +5], 4 / 4 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +1 (181) |
| conf.md.p1 | 98 / 91 / 3 / 0 | 4.6 | -2 [-8, +4] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-8, +4], 4 / 6 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -3 (184) |
| conf.rand.p1 | 101 / 88 / 3 / 0 | 4.8 | +1 [+0, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, +1 [+0, +3], 1 / 0 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +1 (184) |
| conf.rand.m1 | 97 / 92 / 3 / 0 | 4.9 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-6, +2], 1 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -2 (186) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 -2 [-8, +3]; vs conf.rand.m1 +2 [-3, +8] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 181 questions; mean words 4.9 -> 4.7
KEY family=E judge=allenai layer=32 A=not_met B=not_met dR=-1 dR_F=NA dAns=-2 dAns_F=NA rand_margin=-2 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:57:17 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

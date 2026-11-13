TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 28

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 66 / 51 / 2 / 73 | 4.6 | +5 [-1, +11] | +0 [-4, +4] | +0 [-4, +4] | 115, +4 [-1, +9], 5 / 1 | 2 / 2 / 0 | 1 / 1 / 2 / 0 | +4 (109) |
| x.w.m1 | 61 / 55 / 2 / 74 | 4.8 | +0 [-5, +5] | -1 [-5, +2] | +1 [-2, +5] | 115, +1 [-3, +5], 3 / 2 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +1 (106) |
| x.md.p1 | 60 / 53 / 3 / 76 | 4.6 | -1 [-6, +4] | -4 [-9, +0] | +3 [-1, +8] | 112, +0 [-5, +5], 3 / 3 | 5 / 1 / 1 | 1 / 4 / 0 / 1 | +0 (102) |
| x.rand.p1 | 59 / 58 / 2 / 73 | 5.0 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 117, -2 [-5, +0], 0 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (110) |
| x.rand.m1 | 61 / 56 / 2 / 73 | 4.5 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (117) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 73 / 61 / 2 / 56; dR +2 [-2, +6]; answered-rate change -0.5 pp [-1.6, +0.0]; out / in 1 / 0; mean words 7.9 -> 7.8
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +7 [+1, +14]; vs x.rand.m1 +5 [-1, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-2.1, +2.1]; moved out / in 2 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +4 on 109 questions; mean words 4.8 -> 4.6
KEY family=E judge=allenai layer=28 A=not_met B=not_met dR=+5 dR_F=+2 dAns=+0 dAns_F=-1 rand_margin=+5

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 28

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 46 / 45 / 2 / 99 | +2 [-2, +6] | -2 [-6, +2] | +2 [-2, +6] | 90, +1, 2 / 1 | 3 / 1 / 0 | +1 (85) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 43 / 38 / 0 / 111 | -2 [-8, +3] | -3 [-9, +2] | +3 [-2, +9] | 79, +1, 2 / 1 | 5 / 2 / 0 | +0 (73) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 96 / 92 / 3 / 1 | -4 [-8, -1] | -2 [-5, +0] | +1 [+0, +3] | 188, -3, 0 / 3 | 2 / 0 / 1 | -3 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 101 / 87 / 3 / 1 | +1 [-7, +9] | -2 [-5, +0] | +1 [+0, +3] | 188, +2, 8 / 6 | 2 / 0 / 1 | +1 (176) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 99 / 90 / 3 / 0 | -1 [-4, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -1, 1 / 2 | 1 / 0 / 1 | +1 (183) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 97 / 93 / 2 / 0 | -3 [-8, +1] | +0 [+0, +0] | +0 [+0, +0] | 190, -3, 1 / 4 | 0 / 0 / 0 | -3 (182) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 101 / 85 / 5 / 1 | +1 [-5, +8] | -4 [-8, -1] | +1 [+0, +3] | 186, +3, 6 / 3 | 4 / 0 / 3 | +2 (173) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 100 / 87 / 4 / 1 | +0 [-7, +7] | -3 [-7, +0] | +1 [+0, +3] | 187, +2, 6 / 4 | 3 / 0 / 2 | +1 (174) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 108 / 79 / 4 / 1 | -5 [-13, +2] | +1 [+0, +3] | +0 [+0, +0] | 186, -6, 4 / 10 | 0 / 1 / 0 | -6 (167) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 73 / 61 / 2 / 56 | +2 [-2, +6] | -1 [-3, +0] | +0 [+0, +0] | 134, +3, 3 / 0 | 1 / 0 / 1 | +2 (128) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 28

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 101 / 85 / 5 / 1 | 4.6 | +1 [-5, +8] | -4 [-8, -1] | +1 [+0, +3] | 186, +3 [-3, +9], 6 / 3 | 4 / 0 / 3 | 2 / 2 / 0 / 0 | +2 (173) |
| conf.md.p1 | 101 / 87 / 3 / 1 | 4.3 | +1 [-7, +9] | -2 [-5, +0] | +1 [+0, +3] | 188, +2 [-6, +9], 8 / 6 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +1 (176) |
| conf.rand.p1 | 97 / 93 / 2 / 0 | 5.0 | -3 [-8, +1] | +0 [+0, +0] | +0 [+0, +0] | 190, -3 [-8, +1], 1 / 4 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -3 (182) |
| conf.rand.m1 | 99 / 90 / 3 / 0 | 4.7 | -1 [-4, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -1 [-4, +2], 1 / 2 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +1 (183) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +4 [-3, +11]; vs conf.rand.m1 +2 [-5, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -2.1 pp [-4.2, -0.5]; moved out / in 4 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 173 questions; mean words 4.9 -> 4.6
KEY family=E judge=allenai layer=28 A=not_met B=not_met dR=+1 dR_F=NA dAns=-4 dAns_F=NA rand_margin=+2 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:53:09 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

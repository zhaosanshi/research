TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 54 / 55 / 2 / 81 | 4.7 | -7 [-14, +0] | -8 [-15, -2] | +8 [+2, +15] | 107, -5 [-10, +0], 1 / 6 | 10 / 2 / 0 | 4 / 6 / 2 / 0 | -4 (95) |
| x.w.m1 | 61 / 59 / 3 / 69 | 4.6 | +0 [-6, +6] | +3 [-2, +8] | -4 [-9, +0] | 115, +0 [-5, +5], 3 / 3 | 2 / 5 / 1 | 1 / 1 / 1 / 4 | +1 (106) |
| x.md.p1 | 61 / 58 / 2 / 71 | 5.1 | +0 [-6, +5] | +2 [-3, +8] | -2 [-8, +3] | 114, +2 [-3, +7], 4 / 2 | 3 / 5 / 0 | 2 / 1 / 0 / 5 | +1 (103) |
| x.rand.p1 | 59 / 56 / 3 / 74 | 4.4 | -2 [-7, +3] | -2 [-6, +2] | +1 [-2, +4] | 114, +0 [-4, +4], 2 / 2 | 3 / 1 / 1 | 2 / 1 / 0 / 1 | +0 (112) |
| x.rand.m1 | 65 / 51 / 2 / 74 | 4.6 | +4 [+0, +9] | -1 [-4, +2] | +1 [-2, +4] | 115, +3 [-1, +8], 4 / 1 | 2 / 1 / 0 | 0 / 2 / 1 / 0 | +2 (108) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 68 / 54 / 1 / 69; dR -3 [-10, +4]; answered-rate change -6.8 pp [-10.9, -3.1]; out / in 14 / 1; mean words 7.9 -> 7.4
- main effect: dR -7 (-3.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -5 [-12, +2]; vs x.rand.m1 -11 [-19, -4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -4.2 pp [-7.8, -1.0]; moved out / in 10 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same -4 on 95 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=9 A=not_met B=not_met dR=-7 dR_F=-3 dAns=-8 dAns_F=-13 rand_margin=-11

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 9

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 42 / 44 / 0 / 106 | -2 [-7, +3] | -7 [-14, -1] | +9 [+4, +15] | 84, +1, 2 / 1 | 9 / 2 / 0 | +0 (79) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 50 / 43 / 0 / 99 | +5 [-1, +11] | +9 [+4, +15] | -9 [-15, -4] | 84, +0, 2 / 2 | 0 / 9 / 0 | +1 (79) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 99 / 90 / 3 / 0 | -1 [-5, +3] | -1 [-3, +0] | +0 [+0, +0] | 189, -1, 2 / 3 | 1 / 0 / 1 | -1 (189) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 97 / 91 / 4 / 0 | -3 [-10, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, -2, 4 / 6 | 2 / 0 / 2 | +0 (172) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-5, +6] | -2 [-6, +2] | +1 [+0, +3] | 187, +0, 3 / 3 | 3 / 1 / 2 | +0 (177) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 96 / 92 / 4 / 0 | -4 [-11, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, -3, 4 / 7 | 2 / 0 / 2 | -2 (182) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 101 / 88 / 2 / 1 | +1 [-8, +9] | -1 [-5, +2] | +1 [+0, +3] | 188, +1, 9 / 8 | 2 / 1 / 1 | +0 (162) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 104 / 86 / 2 / 0 | +4 [-4, +12] | +0 [-3, +3] | +0 [+0, +0] | 189, +3, 10 / 7 | 1 / 1 / 1 | +2 (164) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 101 / 89 / 2 / 0 | -12 [-22, -2] | +4 [+1, +8] | -1 [-3, +0] | 186, -16, 4 / 20 | 0 / 4 / 0 | -13 (164) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 68 / 54 / 1 / 69 | -3 [-10, +4] | -13 [-21, -6] | +13 [+6, +21] | 121, +4, 5 / 1 | 14 / 1 / 0 | +2 (110) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 101 / 88 / 2 / 1 | 5.2 | +1 [-8, +9] | -1 [-5, +2] | +1 [+0, +3] | 188, +1 [-7, +9], 9 / 8 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +0 (162) |
| conf.md.p1 | 97 / 91 / 4 / 0 | 5.1 | -3 [-10, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, -2 [-8, +4], 4 / 6 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | +0 (172) |
| conf.rand.p1 | 96 / 92 / 4 / 0 | 4.6 | -4 [-11, +3] | -2 [-5, +0] | +0 [+0, +0] | 188, -3 [-10, +3], 4 / 7 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | -2 (182) |
| conf.rand.m1 | 100 / 88 / 3 / 1 | 4.8 | +0 [-5, +6] | -2 [-6, +2] | +1 [+0, +3] | 187, +0 [-5, +5], 3 / 3 | 3 / 1 / 2 | 1 / 2 / 1 / 0 | +0 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +5 [-4, +14]; vs conf.rand.m1 +1 [-6, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +0 on 162 questions; mean words 4.9 -> 5.2
KEY family=E judge=allenai layer=9 A=not_met B=not_met dR=+1 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+1 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:30:02 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

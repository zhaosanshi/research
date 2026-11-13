TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 111 / 48 / 0 / 33 | 3.0 | +8 [+1, +15] | -7 [-12, -2] | +8 [+3, +14] | 159, +9 [+2, +16], 11 / 2 | 7 / 0 / 0 | 1 / 6 / 0 / 0 | +8 (155) |
| x.w.m1 | 94 / 74 / 2 / 22 | 3.4 | -9 [-17, -2] | +2 [-3, +8] | -3 [-8, +2] | 163, -8 [-15, -2], 2 / 10 | 3 / 5 / 1 | 1 / 2 / 0 / 5 | -4 (145) |
| x.md.p1 | 99 / 61 / 1 / 31 | 3.3 | -4 [-11, +3] | -6 [-11, -2] | +6 [+2, +11] | 160, +0 [-6, +6], 4 / 4 | 6 / 0 / 0 | 4 / 2 / 0 / 0 | +1 (155) |
| x.rand.p1 | 103 / 58 / 2 / 29 | 3.2 | +0 [-4, +4] | -5 [-10, -1] | +4 [+1, +8] | 161, +2 [+0, +5], 2 / 0 | 5 / 0 / 1 | 2 / 3 / 0 / 0 | +2 (160) |
| x.rand.m1 | 101 / 66 / 1 / 24 | 3.4 | -2 [-6, +2] | +1 [+0, +3] | -1 [-3, +0] | 166, -2 [-6, +2], 1 / 3 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (157) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 140 / 33 / 1 / 18; dR +6 [-2, +14]; answered-rate change -1.0 pp [-3.6, +1.6]; out / in 4 / 2; mean words 10.1 -> 10.3
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +8 [+0, +16]; vs x.rand.m1 +10 [+2, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.6 pp [-6.2, -1.0]; moved out / in 7 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 155 questions; mean words 3.2 -> 3.0
KEY family=E judge=allenai layer=8 A=not_met B=not_met dR=+8 dR_F=+6 dAns=-7 dAns_F=-2 rand_margin=+8

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 8

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 105 / 47 / 0 / 40 | +4 [-3, +11] | -8 [-16, -1] | +10 [+3, +17] | 149, +3, 6 / 3 | 11 / 3 / 0 | +3 (143) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 105 / 53 / 2 / 32 | -12 [-19, -5] | +6 [+1, +12] | -8 [-14, -3] | 151, -11, 1 / 12 | 1 / 7 / 1 | -10 (142) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 110 / 77 / 3 / 2 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 109 / 77 / 3 / 3 | -2 [-8, +4] | -1 [-3, +0] | +1 [+0, +3] | 186, -1, 4 / 5 | 1 / 0 / 0 | +1 (175) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 112 / 75 / 3 / 2 | +1 [-3, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +1, 3 / 2 | 0 / 0 / 0 | +2 (180) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 115 / 73 / 2 / 2 | +4 [+0, +9] | +1 [+0, +3] | +0 [+0, +0] | 187, +3, 4 / 1 | 0 / 1 / 0 | +1 (182) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 128 / 60 / 2 / 2 | +17 [+10, +25] | +1 [+0, +3] | +0 [+0, +0] | 187, +16, 16 / 0 | 0 / 1 / 0 | +14 (180) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 126 / 62 / 2 / 2 | +15 [+8, +23] | +1 [+0, +3] | +0 [+0, +0] | 187, +14, 14 / 0 | 0 / 1 / 0 | +12 (179) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 121 / 67 / 2 / 2 | -14 [-23, -5] | -1 [-3, +0] | +0 [+0, +0] | 188, -13, 4 / 17 | 1 / 0 / 1 | -9 (169) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 137 / 39 / 1 / 15 | +3 [-3, +10] | +1 [-2, +5] | +0 [-3, +3] | 174, +2, 5 / 3 | 1 / 2 / 0 | +1 (161) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 140 / 33 / 1 / 18 | +6 [-2, +14] | -2 [-7, +3] | +3 [-1, +8] | 171, +6, 9 / 3 | 4 / 2 / 0 | +5 (147) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 138 / 27 / 0 / 27 | +4 [-5, +13] | -10 [-18, -2] | +12 [+5, +20] | 161, +6, 9 / 3 | 14 / 4 / 0 | +6 (122) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 128 / 60 / 2 / 2 | 2.9 | +17 [+10, +25] | +1 [+0, +3] | +0 [+0, +0] | 187, +16 [+9, +24], 16 / 0 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +14 (180) |
| conf.md.p1 | 109 / 77 / 3 / 3 | 3.2 | -2 [-8, +4] | -1 [-3, +0] | +1 [+0, +3] | 186, -1 [-7, +5], 4 / 5 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +1 (175) |
| conf.rand.p1 | 115 / 73 / 2 / 2 | 3.1 | +4 [+0, +9] | +1 [+0, +3] | +0 [+0, +0] | 187, +3 [-1, +8], 4 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (182) |
| conf.rand.m1 | 112 / 75 / 3 / 2 | 3.2 | +1 [-3, +5] | +0 [+0, +0] | +0 [+0, +0] | 187, +1 [-3, +5], 3 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +2 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +13 [+6, +21]; vs conf.rand.m1 +16 [+9, +24] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [+0.0, +1.6]; moved out / in 0 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +14 on 180 questions; mean words 3.1 -> 2.9
KEY family=E judge=allenai layer=8 A=incomplete B=incomplete dR=+17 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+13 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 10:31:12 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 55 / 41 / 1 / 95 | 4.5 | -6 [-14, +2] | -21 [-30, -12] | +22 [+14, +31] | 95, +1 [-4, +6], 4 / 3 | 22 / 1 / 0 | 8 / 14 / 1 / 0 | -1 (85) |
| x.w.m1 | 64 / 61 / 3 / 64 | 4.8 | +3 [-3, +10] | +8 [+2, +14] | -9 [-15, -4] | 116, +0 [-6, +6], 4 / 4 | 1 / 9 / 1 | 0 / 1 / 3 / 6 | +1 (104) |
| x.md.p1 | 61 / 60 / 1 / 70 | 4.7 | +0 [-4, +4] | +4 [+1, +8] | -3 [-7, +0] | 117, -1 [-5, +2], 1 / 2 | 0 / 4 / 0 | 0 / 0 / 1 / 3 | +0 (110) |
| x.rand.p1 | 65 / 52 / 2 / 73 | 4.5 | +4 [+0, +9] | +0 [-4, +4] | +0 [-4, +4] | 115, +3 [-1, +8], 4 / 1 | 2 / 2 / 0 | 0 / 2 / 1 / 1 | +3 (110) |
| x.rand.m1 | 60 / 58 / 1 / 73 | 4.5 | -1 [-6, +4] | +1 [-2, +5] | +0 [-3, +3] | 116, -1 [-6, +3], 2 / 3 | 1 / 2 / 0 | 1 / 0 / 1 / 1 | -1 (111) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 61 / 51 / 1 / 79; dR -10 [-19, -1]; answered-rate change -12.0 pp [-16.7, -7.8]; out / in 23 / 0; mean words 7.9 -> 6.7
- main effect: dR -6 (-3.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -10 [-19, -2]; vs x.rand.m1 -5 [-12, +2] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -10.9 pp [-15.6, -6.2]; moved out / in 22 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same -1 on 85 questions; mean words 4.8 -> 4.5
KEY family=E judge=allenai layer=11 A=not_met B=not_met dR=-6 dR_F=-10 dAns=-21 dAns_F=-23 rand_margin=-10

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 43 / 36 / 0 / 113 | -1 [-7, +5] | -14 [-22, -6] | +16 [+9, +24] | 77, +1, 2 / 1 | 16 / 2 / 0 | +1 (71) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 52 / 52 / 0 / 88 | +7 [+0, +15] | +20 [+12, +29] | -20 [-29, -12] | 84, -1, 3 / 4 | 0 / 20 / 0 | -1 (79) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 3 / 0 | -2 [-6, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 1 / 3 | 1 / 0 / 1 | -2 (189) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 95 / 94 / 3 / 0 | -5 [-10, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -5, 1 / 6 | 1 / 0 / 1 | -4 (178) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 95 / 95 / 2 / 0 | -5 [-10, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -5, 1 / 6 | 0 / 0 / 0 | -4 (177) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 95 / 91 / 6 / 0 | -5 [-11, +1] | -4 [-8, -1] | +0 [+0, +0] | 186, -3, 2 / 5 | 4 / 0 / 4 | -3 (180) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 104 / 85 / 2 / 1 | +4 [-4, +12] | -1 [-5, +2] | +1 [+0, +3] | 188, +4, 10 / 6 | 2 / 1 / 1 | +2 (161) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 104 / 85 / 2 / 1 | +4 [-4, +12] | -1 [-5, +2] | +1 [+0, +3] | 188, +4, 10 / 6 | 2 / 1 / 1 | +2 (160) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 98 / 90 / 3 / 1 | -15 [-25, -5] | +2 [+0, +5] | +0 [+0, +0] | 186, -17, 4 / 21 | 0 / 2 / 0 | -15 (166) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 61 / 51 / 1 / 79 | -10 [-19, -1] | -23 [-32, -15] | +23 [+15, +32] | 112, +1, 5 / 4 | 23 / 0 / 0 | +1 (105) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 104 / 85 / 2 / 1 | 5.3 | +4 [-4, +12] | -1 [-5, +2] | +1 [+0, +3] | 188, +4 [-4, +12], 10 / 6 | 2 / 1 / 1 | 1 / 1 / 1 / 0 | +2 (161) |
| conf.md.p1 | 95 / 94 / 3 / 0 | 4.7 | -5 [-10, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -5 [-10, +0], 1 / 6 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -4 (178) |
| conf.rand.p1 | 95 / 91 / 6 / 0 | 4.9 | -5 [-11, +1] | -4 [-8, -1] | +0 [+0, +0] | 186, -3 [-8, +2], 2 / 5 | 4 / 0 / 4 | 2 / 2 / 0 / 0 | -3 (180) |
| conf.rand.m1 | 95 / 95 / 2 / 0 | 4.7 | -5 [-10, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, -5 [-10, +0], 1 / 6 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -4 (177) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +9 [+0, +18]; vs conf.rand.m1 +9 [+1, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +2 on 161 questions; mean words 4.9 -> 5.3
KEY family=E judge=allenai layer=11 A=not_met B=not_met dR=+4 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+9 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:32:16 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

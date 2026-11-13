TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 109 / 48 / 0 / 35 | 3.1 | +6 [-1, +13] | -9 [-15, -4] | +10 [+5, +16] | 157, +8 [+2, +15], 10 / 2 | 9 / 0 / 0 | 2 / 7 / 0 / 0 | +8 (152) |
| x.w.m1 | 99 / 76 / 1 / 16 | 3.5 | -4 [-12, +4] | +9 [+4, +15] | -9 [-15, -4] | 166, -5 [-13, +2], 5 / 10 | 0 / 9 / 0 | 0 / 0 / 1 / 8 | -4 (143) |
| x.md.p1 | 106 / 53 / 2 / 31 | 3.1 | +3 [-5, +10] | -7 [-13, -2] | +6 [+2, +11] | 159, +5 [-2, +12], 9 / 4 | 7 / 0 / 1 | 2 / 5 / 0 / 0 | +6 (149) |
| x.rand.p1 | 101 / 61 / 1 / 29 | 3.1 | -2 [-6, +2] | -4 [-8, -1] | +4 [+1, +8] | 162, -1 [-5, +2], 1 / 2 | 4 / 0 / 0 | 1 / 3 / 0 / 0 | +0 (160) |
| x.rand.m1 | 105 / 62 / 2 / 23 | 3.5 | +2 [-3, +7] | +1 [-2, +5] | -2 [-5, +0] | 165, +3 [-1, +8], 4 / 1 | 1 / 2 / 1 | 1 / 0 / 0 / 2 | +2 (157) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 135 / 36 / 2 / 19; dR +1 [-5, +7]; answered-rate change -2.1 pp [-4.2, -0.5]; out / in 4 / 0; mean words 10.1 -> 10.2
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +8 [+2, +14]; vs x.rand.m1 +4 [-5, +12] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -4.7 pp [-7.8, -2.1]; moved out / in 9 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 152 questions; mean words 3.2 -> 3.1
KEY family=E judge=allenai layer=16 A=not_met B=not_met dR=+6 dR_F=+1 dAns=-9 dAns_F=-4 rand_margin=+4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 106 / 45 / 0 / 41 | +5 [-2, +13] | -9 [-17, -2] | +11 [+4, +19] | 148, +4, 7 / 3 | 12 / 3 / 0 | +4 (143) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 113 / 51 / 2 / 26 | -4 [-12, +4] | +12 [+5, +20] | -14 [-21, -7] | 151, -5, 4 / 9 | 1 / 13 / 1 | -6 (136) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 76 / 3 / 2 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 1 / 1 | 0 / 0 / 0 | +0 (185) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 119 / 68 / 2 / 3 | +8 [+0, +16] | +0 [+0, +0] | +1 [+0, +3] | 187, +8, 13 / 5 | 0 / 0 / 0 | +7 (176) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 109 / 78 / 2 / 3 | -2 [-7, +3] | +0 [+0, +0] | +1 [+0, +3] | 187, -2, 2 / 4 | 0 / 0 / 0 | +0 (176) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 113 / 75 / 2 / 2 | +2 [-2, +6] | +1 [+0, +3] | +0 [+0, +0] | 187, +1, 2 / 1 | 0 / 1 / 0 | +1 (184) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 126 / 61 / 2 / 3 | +15 [+8, +23] | +0 [+0, +0] | +1 [+0, +3] | 187, +15, 16 / 1 | 0 / 0 / 0 | +15 (179) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 127 / 60 / 2 / 3 | +16 [+8, +24] | +0 [+0, +0] | +1 [+0, +3] | 187, +16, 17 / 1 | 0 / 0 / 0 | +15 (177) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 121 / 67 / 2 / 2 | -14 [-24, -4] | -1 [-3, +0] | +0 [+0, +0] | 188, -13, 6 / 19 | 1 / 0 / 1 | -10 (163) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 134 / 40 / 2 / 16 | +0 [-5, +5] | -1 [-3, +0] | +1 [+0, +3] | 174, +1, 4 / 3 | 1 / 0 / 0 | +1 (159) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 135 / 36 / 2 / 19 | +1 [-5, +7] | -4 [-8, -1] | +4 [+1, +8] | 171, +3, 5 / 2 | 4 / 0 / 0 | +2 (146) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 134 / 29 / 0 / 29 | +0 [-9, +9] | -12 [-20, -5] | +14 [+7, +21] | 161, +3, 9 / 6 | 14 / 2 / 0 | +1 (122) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 111 / 76 / 3 / 2; rule abstentions 2; mean words 3.1

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 126 / 61 / 2 / 3 | 2.8 | +15 [+8, +23] | +0 [+0, +0] | +1 [+0, +3] | 187, +15 [+8, +23], 16 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +15 (179) |
| conf.md.p1 | 119 / 68 / 2 / 3 | 3.1 | +8 [+0, +16] | +0 [+0, +0] | +1 [+0, +3] | 187, +8 [+0, +16], 13 / 5 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +7 (176) |
| conf.rand.p1 | 113 / 75 / 2 / 2 | 3.1 | +2 [-2, +6] | +1 [+0, +3] | +0 [+0, +0] | 187, +1 [-2, +5], 2 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (184) |
| conf.rand.m1 | 109 / 78 / 2 / 3 | 3.3 | -2 [-7, +3] | +0 [+0, +0] | +1 [+0, +3] | 187, -2 [-7, +3], 2 / 4 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (176) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +15 (7.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p1 +13 [+6, +21]; vs conf.rand.m1 +17 [+8, +26] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.0 pp [-1.5, +1.5]; moved out / in 0 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +15 on 179 questions; mean words 3.1 -> 2.8
KEY family=E judge=allenai layer=16 A=incomplete B=incomplete dR=+15 dR_F=NA dAns=+0 dAns_F=NA rand_margin=+13 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 10:18:12 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 25

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 67 / 48 / 2 / 75 | 4.7 | +6 [-1, +13] | -2 [-7, +3] | +2 [-3, +7] | 113, +5 [-1, +12], 8 / 3 | 4 / 2 / 0 | 1 / 3 / 2 / 0 | +3 (97) |
| x.w.m1 | 63 / 55 / 2 / 72 | 5.0 | +2 [-3, +7] | +1 [-2, +4] | -1 [-4, +2] | 116, +3 [-1, +8], 4 / 1 | 1 / 2 / 0 | 1 / 0 / 0 / 2 | +2 (106) |
| x.md.p1 | 61 / 55 / 3 / 73 | 4.5 | +0 [-7, +7] | -1 [-6, +4] | +0 [-5, +5] | 113, -1 [-6, +4], 3 / 4 | 4 / 3 / 1 | 2 / 2 / 3 / 0 | -1 (105) |
| x.rand.p1 | 61 / 53 / 2 / 76 | 4.5 | +0 [-4, +4] | -3 [-7, +0] | +3 [+0, +7] | 114, +1 [-2, +4], 2 / 1 | 3 / 0 / 0 | 1 / 2 / 0 / 0 | +2 (108) |
| x.rand.m1 | 60 / 58 / 2 / 72 | 4.6 | -1 [-6, +4] | +1 [-2, +4] | -1 [-4, +2] | 116, -1 [-5, +3], 2 / 3 | 1 / 2 / 0 | 1 / 0 / 1 / 1 | -2 (110) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 71 / 62 / 1 / 58; dR +0 [-6, +6]; answered-rate change -1.0 pp [-3.1, +1.0]; out / in 3 / 1; mean words 7.9 -> 7.9
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +6 [-2, +14]; vs x.rand.m1 +7 [+0, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.0 pp [-3.6, +1.6]; moved out / in 4 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 97 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=25 A=not_met B=not_met dR=+6 dR_F=+0 dAns=-2 dAns_F=-2 rand_margin=+6

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 25

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 54 / 40 / 1 / 97 | +10 [+4, +17] | +1 [-4, +6] | +0 [-5, +5] | 90, +7, 7 / 0 | 3 / 4 / 0 | +3 (81) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 43 / 42 / 0 / 107 | -2 [-9, +5] | +1 [-7, +9] | -1 [-9, +7] | 77, +2, 4 / 2 | 7 / 8 / 0 | +1 (69) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 101 / 89 / 2 / 0 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 190, +1, 1 / 0 | 0 / 0 / 0 | +1 (188) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-9, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 8 / 8 | 2 / 0 / 1 | -2 (176) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 99 / 89 / 4 / 0 | -1 [-7, +5] | -2 [-5, +0] | +0 [+0, +0] | 188, +1, 4 / 3 | 2 / 0 / 2 | +2 (180) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 97 / 91 / 3 / 1 | -3 [-8, +1] | -2 [-5, +0] | +1 [+0, +3] | 188, -2, 1 / 3 | 2 / 0 / 1 | -2 (178) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 107 / 81 / 3 / 1 | +7 [-1, +16] | -2 [-5, +0] | +1 [+0, +3] | 188, +8, 13 / 5 | 2 / 0 / 1 | +4 (163) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 106 / 82 / 3 / 1 | +6 [-2, +14] | -2 [-5, +0] | +1 [+0, +3] | 188, +7, 12 / 5 | 2 / 0 / 1 | +5 (166) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 101 / 85 / 4 / 2 | -12 [-22, -2] | +0 [-3, +3] | +1 [+0, +3] | 185, -12, 6 / 18 | 1 / 1 / 0 | -10 (166) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 71 / 62 / 1 / 58 | +0 [-6, +6] | -2 [-6, +2] | +2 [-2, +6] | 132, +3, 5 / 2 | 3 / 1 / 0 | +2 (120) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 25

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 107 / 81 / 3 / 1 | 5.0 | +7 [-1, +16] | -2 [-5, +0] | +1 [+0, +3] | 188, +8 [+0, +17], 13 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +4 (163) |
| conf.md.p1 | 99 / 89 / 3 / 1 | 4.3 | -1 [-9, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +0 [-8, +8], 8 / 8 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -2 (176) |
| conf.rand.p1 | 97 / 91 / 3 / 1 | 4.8 | -3 [-8, +1] | -2 [-5, +0] | +1 [+0, +3] | 188, -2 [-6, +2], 1 / 3 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -2 (178) |
| conf.rand.m1 | 99 / 89 / 4 / 0 | 4.8 | -1 [-7, +5] | -2 [-5, +0] | +0 [+0, +0] | 188, +1 [-4, +6], 4 / 3 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | +2 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +10 [+3, +18]; vs conf.rand.m1 +8 [-1, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +4 on 163 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=25 A=incomplete B=not_met dR=+7 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+8 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 14:44:48 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

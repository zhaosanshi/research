TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 22

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 61 / 36 / 1 / 94 | 4.5 | +0 [-9, +9] | -20 [-29, -11] | +21 [+12, +30] | 95, +5 [-1, +12], 8 / 3 | 22 / 2 / 0 | 7 / 15 / 2 / 0 | +1 (80) |
| x.w.m1 | 67 / 67 / 4 / 54 | 5.3 | +6 [-1, +13] | +17 [+9, +26] | -19 [-28, -11] | 115, +3 [-3, +9], 6 / 3 | 2 / 19 / 2 | 0 / 2 / 3 / 16 | +3 (101) |
| x.md.p1 | 63 / 56 / 3 / 70 | 4.4 | +2 [-6, +10] | +2 [-4, +8] | -3 [-9, +3] | 113, +1 [-5, +7], 5 / 4 | 4 / 6 / 1 | 3 / 1 / 4 / 2 | +1 (99) |
| x.rand.p1 | 65 / 52 / 2 / 73 | 4.7 | +4 [+1, +8] | +0 [-3, +3] | +0 [-3, +3] | 116, +3 [+0, +7], 3 / 0 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | +3 (107) |
| x.rand.m1 | 57 / 61 / 2 / 72 | 4.7 | -4 [-10, +1] | +1 [-4, +6] | -1 [-6, +4] | 114, -2 [-7, +3], 2 / 4 | 3 / 4 / 0 | 2 / 1 / 0 / 4 | -2 (107) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 68 / 50 / 1 / 73; dR -3 [-10, +3]; answered-rate change -8.9 pp [-13.0, -5.2]; out / in 17 / 0; mean words 7.9 -> 7.4
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -4 [-13, +5]; vs x.rand.m1 +4 [-4, +12] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -10.4 pp [-15.1, -5.7]; moved out / in 22 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +1 on 80 questions; mean words 4.8 -> 4.5
KEY family=E judge=allenai layer=22 A=not_met B=not_met dR=+0 dR_F=-3 dAns=-20 dAns_F=-17 rand_margin=-4

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 22

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 44 / 31 / 0 / 117 | +0 [-8, +8] | -18 [-27, -10] | +20 [+12, +29] | 74, +5, 7 / 2 | 19 / 1 / 0 | +0 (66) |
| caut_E2.wq.m1 | 45 / 39 / 0 / 108 | 52 / 54 / 2 / 84 | +7 [-1, +15] | +22 [+13, +32] | -24 [-34, -15] | 82, +1, 5 / 4 | 2 / 24 / 0 | +1 (73) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 102 / 88 / 2 / 0 | +2 [-2, +6] | +0 [+0, +0] | +0 [+0, +0] | 190, +2, 3 / 1 | 0 / 0 / 0 | +2 (187) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 100 / 89 / 3 / 0 | +0 [-9, +9] | -1 [-3, +0] | +0 [+0, +0] | 189, +0, 10 / 10 | 1 / 0 / 1 | +2 (174) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 95 / 94 / 3 / 0 | -5 [-10, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -5, 1 / 6 | 1 / 0 / 1 | -4 (180) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 95 / 92 / 4 / 1 | -5 [-12, +2] | -3 [-7, +0] | +1 [+0, +3] | 187, -3, 4 / 7 | 3 / 0 / 2 | -3 (178) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 105 / 82 / 3 / 2 | +5 [-5, +15] | -3 [-8, +1] | +2 [+0, +5] | 186, +6, 15 / 9 | 4 / 1 / 2 | +2 (159) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 105 / 82 / 3 / 2 | +5 [-5, +15] | -3 [-8, +1] | +2 [+0, +5] | 186, +6, 15 / 9 | 4 / 1 / 2 | +2 (158) |
| conf_E2.wq.m1 | 113 / 73 / 5 / 1 | 102 / 86 / 4 / 0 | -11 [-23, +1] | +2 [-2, +6] | -1 [-3, +0] | 185, -13, 9 / 22 | 1 / 3 / 1 | -10 (167) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 68 / 50 / 1 / 73 | -3 [-10, +3] | -17 [-25, -10] | +17 [+10, +25] | 118, +2, 4 / 2 | 17 / 0 / 0 | +1 (101) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 22

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 105 / 82 / 3 / 2 | 5.0 | +5 [-5, +15] | -3 [-8, +1] | +2 [+0, +5] | 186, +6 [-3, +16], 15 / 9 | 4 / 1 / 2 | 2 / 2 / 1 / 0 | +2 (159) |
| conf.md.p1 | 100 / 89 / 3 / 0 | 4.2 | +0 [-9, +9] | -1 [-3, +0] | +0 [+0, +0] | 189, +0 [-9, +9], 10 / 10 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +2 (174) |
| conf.rand.p1 | 95 / 92 / 4 / 1 | 4.8 | -5 [-12, +2] | -3 [-7, +0] | +1 [+0, +3] | 187, -3 [-10, +3], 4 / 7 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | -3 (178) |
| conf.rand.m1 | 95 / 94 / 3 / 0 | 4.9 | -5 [-10, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -5 [-10, +0], 1 / 6 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -4 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +10 [+0, +20]; vs conf.rand.m1 +10 [+1, +20] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-4.2, +0.5]; moved out / in 4 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +2 on 159 questions; mean words 4.9 -> 5.0
KEY family=E judge=allenai layer=22 A=not_met B=not_met dR=+5 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+10 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

=== 2026-11-13 14:41:21 [qwen3-8b_E2 E tables gpunone e_on=E2] tables done ===

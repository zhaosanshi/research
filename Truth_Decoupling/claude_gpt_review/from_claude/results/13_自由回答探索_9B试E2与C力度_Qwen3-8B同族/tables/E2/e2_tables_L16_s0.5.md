TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=0.5 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 106 / 54 / 0 / 32 | 3.2 | +3 [-3, +10] | -6 [-11, -2] | +7 [+3, +12] | 160, +5 [-1, +11], 7 / 2 | 6 / 0 / 0 | 2 / 4 / 0 / 0 | +6 (156) |
| x.rand.p0.5 | 102 / 63 / 1 / 26 | 3.2 | -1 [-3, +0] | -1 [-3, +0] | +1 [+0, +3] | 165, -1 [-3, +0], 0 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -1 (165) |
| x.rand.m0.5 | 104 / 61 / 2 / 25 | 3.4 | +1 [-2, +5] | -1 [-3, +0] | +0 [+0, +0] | 165, +2 [+0, +5], 2 / 0 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +1 (162) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 134 / 40 / 2 / 16; dR +0 [-5, +5]; answered-rate change -0.5 pp [-1.6, +0.0]; out / in 1 / 0; mean words 10.1 -> 10.1
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p0.5 +4 [-2, +10]; vs x.rand.m0.5 +2 [-5, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.1 pp [-5.7, -1.0]; moved out / in 6 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +6 on 156 questions; mean words 3.2 -> 3.2
KEY family=E judge=allenai layer=16 A=not_met B=not_met dR=+3 dR_F=+0 dAns=-6 dAns_F=-1 rand_margin=+2 EXPLORATORY base=x main=x.w.p0.5 strength=0.5

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

=== 2026-11-13 10:18:15 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=0.5 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 110 / 51 / 0 / 31 | 3.1 | +7 [+1, +14] | -5 [-10, -1] | +6 [+2, +11] | 161, +8 [+2, +14], 9 / 1 | 5 / 0 / 0 | 1 / 4 / 0 / 0 | +9 (158) |
| x.rand.p0.5 | 104 / 57 / 2 / 29 | 3.2 | +1 [-3, +5] | -5 [-10, -1] | +4 [+1, +8] | 161, +3 [+0, +7], 3 / 0 | 5 / 0 / 1 | 2 / 3 / 0 / 0 | +3 (160) |
| x.rand.m0.5 | 103 / 64 / 1 / 24 | 3.2 | +0 [-3, +3] | +1 [+0, +3] | -1 [-3, +0] | 166, +0 [-3, +3], 1 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (165) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 138 / 37 / 1 / 16; dR +4 [-2, +10]; answered-rate change +0.0 pp [-1.6, +1.6]; out / in 1 / 1; mean words 10.1 -> 9.9
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p0.5 +6 [+0, +12]; vs x.rand.m0.5 +7 [+1, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.6 pp [-5.2, -0.5]; moved out / in 5 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +9 on 158 questions; mean words 3.2 -> 3.1
KEY family=E judge=allenai layer=10 A=met B=not_met dR=+7 dR_F=+4 dAns=-5 dAns_F=+0 rand_margin=+6 EXPLORATORY base=x main=x.w.p0.5 strength=0.5

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 10

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 101 / 48 / 0 / 43 | +0 [-8, +8] | -11 [-19, -4] | +13 [+6, +21] | 147, +2, 6 / 4 | 13 / 2 / 0 | +2 (140) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 109 / 50 / 2 / 31 | -8 [-16, +0] | +7 [+2, +13] | -9 [-15, -4] | 151, -8, 3 / 11 | 1 / 8 / 1 | -9 (140) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 77 / 2 / 2 | +0 [-3, +3] | +1 [+0, +3] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 1 / 0 | +0 (186) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 116 / 71 / 3 / 2 | +5 [-1, +12] | +0 [-3, +3] | +0 [+0, +0] | 186, +5, 7 / 2 | 1 / 1 / 1 | +2 (170) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 113 / 75 / 2 / 2 | +2 [-2, +6] | +1 [+0, +3] | +0 [+0, +0] | 187, +1, 2 / 1 | 0 / 1 / 0 | +1 (181) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 113 / 73 / 3 / 3 | +2 [-3, +7] | -1 [-3, +0] | +1 [+0, +3] | 186, +3, 4 / 1 | 1 / 0 / 1 | +2 (179) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 127 / 59 / 3 / 3 | +16 [+8, +25] | -1 [-3, +0] | +1 [+0, +3] | 186, +17, 18 / 1 | 1 / 0 / 1 | +15 (175) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 129 / 58 / 2 / 3 | +18 [+10, +27] | +0 [+0, +0] | +1 [+0, +3] | 187, +18, 19 / 1 | 0 / 0 / 0 | +15 (176) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 118 / 70 / 2 / 2 | -17 [-26, -8] | -1 [-3, +0] | +0 [+0, +0] | 188, -16, 3 / 19 | 1 / 0 / 1 | -10 (166) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 138 / 37 / 1 / 16 | +4 [-2, +10] | +0 [-3, +3] | +1 [+0, +3] | 174, +4, 6 / 2 | 1 / 1 / 0 | +3 (165) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 135 / 38 / 1 / 18 | +1 [-7, +9] | -2 [-6, +2] | +3 [+0, +7] | 172, +1, 7 / 6 | 3 / 1 / 0 | +2 (145) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 144 / 17 / 1 / 30 | +10 [-1, +21] | -14 [-23, -6] | +15 [+8, +23] | 158, +15, 18 / 3 | 17 / 3 / 1 | +8 (121) |

=== 2026-11-13 10:33:46 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

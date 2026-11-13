TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=0.5 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 109 / 55 / 0 / 28 | 3.2 | +6 [+1, +12] | -2 [-5, +0] | +3 [+0, +7] | 164, +6 [+1, +12], 7 / 1 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | +7 (162) |
| x.rand.p0.5 | 102 / 61 / 1 / 28 | 3.3 | -1 [-3, +0] | -3 [-7, +0] | +3 [+0, +7] | 163, +0 [+0, +0], 0 / 0 | 3 / 0 / 0 | 1 / 2 / 0 / 0 | +0 (162) |
| x.rand.m0.5 | 103 / 64 / 1 / 24 | 3.3 | +0 [-3, +3] | +1 [+0, +3] | -1 [-3, +0] | 166, +0 [-3, +3], 1 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +1 (164) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 137 / 39 / 1 / 15; dR +3 [-3, +10]; answered-rate change +0.5 pp [-1.0, +2.6]; out / in 1 / 2; mean words 10.1 -> 10.2
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p0.5 +7 [+2, +13]; vs x.rand.m0.5 +6 [+1, +12] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +7 on 162 questions; mean words 3.2 -> 3.2
KEY family=E judge=allenai layer=8 A=met B=not_met dR=+6 dR_F=+3 dAns=-2 dAns_F=+1 rand_margin=+6 EXPLORATORY base=x main=x.w.p0.5 strength=0.5

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

=== 2026-11-13 10:31:15 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

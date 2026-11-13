TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=0.5 judges=allenai
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 106 / 54 / 0 / 32 | 3.1 | +3 [-2, +8] | -6 [-11, -2] | +7 [+2, +12] | 160, +4 [+0, +9], 5 / 1 | 6 / 0 / 0 | 1 / 5 / 0 / 0 | +4 (158) |
| x.rand.p0.5 | 102 / 64 / 2 / 24 | 3.2 | -1 [-3, +0] | +0 [-3, +3] | -1 [-3, +0] | 165, +0 [+0, +0], 0 / 0 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (165) |
| x.rand.m0.5 | 105 / 61 / 1 / 25 | 3.2 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 166, +2 [+0, +5], 2 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +2 (165) |

- F, complete-sentence background (base x_F): x_F 134 / 41 / 2 / 15 -> with the E write 133 / 41 / 2 / 16; dR -1 [-6, +4]; answered-rate change -0.5 pp [-1.6, +0.0]; out / in 1 / 0; mean words 10.1 -> 10.0
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p0.5 +4 [-1, +10]; vs x.rand.m0.5 +1 [-4, +6] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.1 pp [-5.7, -1.0]; moved out / in 6 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +4 on 158 questions; mean words 3.2 -> 3.1
KEY family=E judge=allenai layer=17 A=not_met B=not_met dR=+3 dR_F=-1 dAns=-6 dAns_F=-1 rand_margin=+1 EXPLORATORY base=x main=x.w.p0.5 strength=0.5

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 101 / 59 / 2 / 30 | 108 / 46 / 0 / 38 | +7 [+0, +14] | -6 [-14, +2] | +8 [+0, +16] | 149, +5, 6 / 1 | 11 / 5 / 0 | +5 (142) |
| caut_E2.wq.m1 | 117 / 35 / 0 / 40 | 115 / 46 / 1 / 30 | -2 [-9, +5] | +9 [+4, +15] | -10 [-17, -5] | 152, -5, 2 / 7 | 0 / 9 / 0 | -4 (140) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 110 / 77 / 3 / 2 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 187, -1, 0 / 1 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 111 / 76 / 3 / 2 | 109 / 77 / 2 / 4 | -2 [-9, +5] | -1 [-3, +0] | +2 [+0, +5] | 186, -1, 6 / 7 | 1 / 0 / 0 | +1 (178) |
| conf.rand.m1 | 111 / 76 / 3 / 2 | 114 / 73 / 3 / 2 | +3 [+0, +7] | +0 [+0, +0] | +0 [+0, +0] | 187, +3, 3 / 0 | 0 / 0 / 0 | +1 (183) |
| conf.rand.p1 | 111 / 76 / 3 / 2 | 110 / 77 / 2 / 3 | -1 [-5, +2] | +0 [+0, +0] | +1 [+0, +3] | 187, -1, 1 / 2 | 0 / 0 / 0 | -2 (183) |
| conf.w.p1 | 111 / 76 / 3 / 2 | 129 / 58 / 3 / 2 | +18 [+11, +26] | +0 [+0, +0] | +0 [+0, +0] | 187, +18, 18 / 0 | 0 / 0 / 0 | +16 (181) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 131 / 56 / 3 / 2 | +20 [+12, +29] | +0 [+0, +0] | +0 [+0, +0] | 187, +20, 20 / 0 | 0 / 0 / 0 | +17 (179) |
| conf_E2.wq.m1 | 135 / 54 / 1 / 2 | 124 / 64 / 2 / 2 | -11 [-21, -2] | -1 [-3, +0] | +0 [+0, +0] | 188, -10, 7 / 17 | 1 / 0 / 1 | -7 (166) |
| x_F.w.p0.5 | 134 / 41 / 2 / 15 | 133 / 41 / 2 / 16 | -1 [-6, +4] | -1 [-3, +0] | +1 [+0, +3] | 174, +0, 3 / 3 | 1 / 0 / 0 | +0 (164) |
| x_F.w.p1 | 134 / 41 / 2 / 15 | 135 / 38 / 1 / 18 | +1 [-5, +8] | -2 [-6, +2] | +3 [+0, +7] | 172, +2, 5 / 3 | 3 / 1 / 0 | +1 (153) |
| x_F.w.p2 | 134 / 41 / 2 / 15 | 135 / 27 / 0 / 30 | +1 [-9, +11] | -13 [-22, -5] | +15 [+8, +23] | 159, +6, 11 / 5 | 16 / 3 / 0 | +4 (135) |

=== 2026-11-13 10:19:15 [qwen3.5-9b_E2 E tables gpunone e_on=E2] tables done ===

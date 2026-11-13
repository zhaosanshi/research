TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b_side_E2 e_on=E2 work=judge_tune_proX_E_L<layer> strength=1 judges=allenai pro
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 80 / 75 / 3 / 34 | 4.1 | +2 [-6, +10] | -11 [-18, -5] | +11 [+5, +18] | 155, +6 [-1, +13], 10 / 4 | 11 / 0 / 0 | 4 / 7 / 0 / 0 | +4 (138) |
| x.w.m1 | 79 / 96 / 1 / 16 | 3.8 | +1 [-7, +9] | +9 [+4, +15] | -7 [-13, -2] | 166, -4 [-11, +3], 4 / 8 | 0 / 9 / 0 | 0 / 0 / 5 / 4 | -4 (161) |
| x.md.p1 | 81 / 84 / 2 / 25 | 3.8 | +3 [-2, +8] | -1 [-5, +2] | +2 [+0, +5] | 164, +3 [-1, +8], 4 / 1 | 2 / 1 / 0 | 1 / 1 / 1 / 0 | +3 (161) |
| x.rand.p1 | 79 / 87 / 3 / 23 | 4.2 | +1 [-3, +5] | +0 [+0, +0] | +0 [+0, +0] | 166, +1 [-3, +5], 3 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (162) |
| x.rand.m1 | 81 / 86 / 1 / 24 | 4.0 | +3 [+0, +7] | +1 [-2, +4] | +1 [+0, +3] | 165, +2 [+0, +5], 2 / 0 | 1 / 2 / 0 | 0 / 1 / 1 / 1 | +1 (162) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 104 / 49 / 0 / 39; dR +2 [-8, +11]; answered-rate change -4.2 pp [-7.3, -1.0]; out / in 9 / 1; mean words 9.3 -> 9.7
- main effect: dR +2 (1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +1 [-8, +10]; vs x.rand.m1 -1 [-9, +7] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -5.7 pp [-9.4, -2.6]; moved out / in 11 / 0 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +4 on 138 questions; mean words 4.0 -> 4.1
KEY family=E judge=allenai layer=8 A=not_met B=not_met dR=+2 dR_F=+2 dAns=-11 dAns_F=-8 rand_margin=-1

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 8

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 82 / 82 / 1 / 27 | 82 / 74 / 3 / 33 | +0 [-7, +7] | -8 [-14, -3] | +6 [+2, +11] | 156, +2, 6 / 4 | 8 / 0 / 2 | +3 (146) |
| caut_E2.wq.m1 | 102 / 53 / 0 / 37 | 95 / 64 / 1 / 32 | -7 [-15, +0] | +4 [-1, +10] | -5 [-10, +0] | 153, -5, 4 / 9 | 2 / 6 / 1 | -5 (142) |
| conf.iso.p1 | 88 / 101 / 1 / 2 | 87 / 102 / 1 / 2 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 189, -1, 1 / 2 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 88 / 101 / 1 / 2 | 85 / 104 / 1 / 2 | -3 [-7, +0] | +0 [-3, +3] | +0 [+0, +0] | 188, -2, 0 / 2 | 1 / 1 / 1 | -1 (183) |
| conf.rand.m1 | 88 / 101 / 1 / 2 | 86 / 103 / 1 / 2 | -2 [-7, +2] | +0 [-3, +3] | +0 [+0, +0] | 188, -1, 2 / 3 | 1 / 1 / 1 | +0 (186) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 88 / 101 / 1 / 2 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 189, +0, 2 / 2 | 0 / 0 / 0 | -1 (185) |
| conf.w.p1 | 88 / 101 / 1 / 2 | 94 / 96 / 1 / 1 | +6 [-2, +14] | +1 [-2, +5] | -1 [-3, +0] | 188, +6, 11 / 5 | 1 / 2 / 1 | +3 (166) |
| conf.wq.p1 | 88 / 101 / 1 / 2 | 96 / 94 / 1 / 1 | +8 [+0, +17] | +1 [-2, +5] | -1 [-3, +0] | 188, +8, 13 / 5 | 1 / 2 / 1 | +6 (167) |
| conf_E2.wq.m1 | 117 / 74 / 1 / 0 | 106 / 85 / 1 / 0 | -11 [-20, -2] | +0 [+0, +0] | +0 [+0, +0] | 191, -11, 6 / 17 | 0 / 0 / 0 | -10 (185) |
| x_F.w.p1 | 102 / 59 / 1 / 30 | 104 / 49 / 0 / 39 | +2 [-8, +11] | -8 [-14, -2] | +9 [+3, +16] | 152, +5, 12 / 7 | 9 / 1 / 0 | +5 (109) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 88 / 101 / 1 / 2; rule abstentions 2; mean words 3.4

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 94 / 96 / 1 / 1 | 3.9 | +6 [-2, +14] | +1 [-2, +5] | -1 [-3, +0] | 188, +6 [-2, +14], 11 / 5 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +3 (166) |
| conf.md.p1 | 85 / 104 / 1 / 2 | 3.4 | -3 [-7, +0] | +0 [-3, +3] | +0 [+0, +0] | 188, -2 [-5, +0], 0 / 2 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | -1 (183) |
| conf.rand.p1 | 88 / 101 / 1 / 2 | 3.4 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 189, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (185) |
| conf.rand.m1 | 86 / 103 / 1 / 2 | 3.4 | -2 [-7, +2] | +0 [-3, +3] | +0 [+0, +0] | 188, -1 [-5, +3], 2 / 3 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (186) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +6 [-3, +15]; vs conf.rand.m1 +8 [+0, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change +0.5 pp [-1.0, +2.6]; moved out / in 1 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +3 on 166 questions; mean words 3.4 -> 3.9
KEY family=E judge=allenai layer=8 A=incomplete B=not_met dR=+6 dR_F=NA dAns=+1 dAns_F=NA rand_margin=+6 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 65 / 87 / 6 / 34 | 4.1 | +1 [-8, +10] | -10 [-17, -3] | +10 [+5, +16] | 150, +4 [-3, +11], 9 / 5 | 12 / 2 / 2 | 5 / 7 / 2 / 0 | +6 (135) |
| x.w.m1 | 62 / 106 / 7 / 17 | 3.8 | -2 [-9, +5] | +6 [+1, +12] | -7 [-13, -2] | 161, -3 [-9, +3], 3 / 6 | 1 / 7 / 1 | 1 / 0 / 2 / 5 | -3 (158) |
| x.md.p1 | 65 / 96 / 5 / 26 | 3.8 | +1 [-3, +6] | -1 [-4, +2] | +2 [+0, +5] | 160, +0 [-4, +4], 2 / 2 | 2 / 1 / 0 | 0 / 2 / 1 / 0 | +1 (157) |
| x.rand.p1 | 64 / 99 / 5 / 24 | 4.2 | +0 [-3, +3] | +1 [+0, +3] | +0 [+0, +0] | 162, -1 [-3, +0], 0 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | -1 (157) |
| x.rand.m1 | 65 / 97 / 5 / 25 | 4.0 | +1 [-2, +5] | +0 [-3, +3] | +1 [+0, +3] | 161, +0 [-3, +3], 1 / 1 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | +1 (158) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 79 / 70 / 4 / 39; dR -1 [-10, +8]; answered-rate change -5.7 pp [-9.4, -2.1]; out / in 12 / 1; mean words 9.3 -> 9.7
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +1 [-7, +10]; vs x.rand.m1 +0 [-8, +8] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -5.2 pp [-8.9, -1.6]; moved out / in 12 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +6 on 135 questions; mean words 4.0 -> 4.1
KEY family=E judge=pro layer=8 A=not_met B=not_met dR=+1 dR_F=-1 dAns=-10 dAns_F=-11 rand_margin=+0

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 8

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 64 / 94 / 6 / 28 | 66 / 87 / 5 / 34 | +2 [-6, +10] | -5 [-11, +0] | +6 [+2, +11] | 151, +2, 7 / 5 | 7 / 2 / 1 | +4 (140) |
| caut_E2.wq.m1 | 81 / 69 / 2 / 40 | 73 / 82 / 4 / 33 | -8 [-16, -1] | +5 [+0, +10] | -7 [-13, -2] | 149, -7, 3 / 10 | 1 / 6 / 1 | -4 (137) |
| conf.iso.p1 | 70 / 112 / 9 / 1 | 74 / 107 / 9 / 2 | +4 [+1, +8] | -1 [-3, +0] | +1 [+0, +3] | 181, +4, 4 / 0 | 1 / 0 / 0 | +4 (178) |
| conf.md.p1 | 70 / 112 / 9 / 1 | 69 / 111 / 10 / 2 | -1 [-6, +4] | -2 [-6, +2] | +1 [+0, +3] | 179, -1, 2 / 3 | 3 / 1 / 2 | -1 (175) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 70 / 112 / 9 / 1 | +0 [-4, +4] | +0 [-3, +3] | +0 [+0, +0] | 181, +0, 1 / 1 | 1 / 1 / 1 | +0 (179) |
| conf.rand.p1 | 70 / 112 / 9 / 1 | 70 / 111 / 9 / 2 | +0 [-4, +4] | -1 [-5, +2] | +1 [+0, +3] | 180, -1, 1 / 2 | 2 / 1 / 1 | -1 (176) |
| conf.w.p1 | 70 / 112 / 9 / 1 | 78 / 103 / 10 / 1 | +8 [-2, +18] | -1 [-6, +4] | +0 [-3, +3] | 178, +7, 14 / 7 | 4 / 3 / 3 | +3 (158) |
| conf.wq.p1 | 70 / 112 / 9 / 1 | 81 / 100 / 10 / 1 | +11 [+2, +20] | -1 [-5, +3] | +0 [-3, +3] | 179, +10, 15 / 5 | 3 / 2 / 2 | +6 (160) |
| conf_E2.wq.m1 | 98 / 86 / 4 / 4 | 87 / 97 / 7 / 1 | -11 [-22, +0] | +0 [-5, +5] | -3 [-7, +0] | 181, -11, 8 / 19 | 3 / 3 / 3 | -12 (175) |
| x_F.w.p1 | 80 / 80 / 2 / 30 | 79 / 70 / 4 / 39 | -1 [-10, +8] | -11 [-18, -4] | +9 [+3, +16] | 148, +3, 9 / 6 | 12 / 1 / 2 | +3 (107) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 70 / 112 / 9 / 1; rule abstentions 2; mean words 3.4

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 78 / 103 / 10 / 1 | 3.9 | +8 [-2, +18] | -1 [-6, +4] | +0 [-3, +3] | 178, +7 [-2, +16], 14 / 7 | 4 / 3 / 3 | 2 / 2 / 3 / 0 | +3 (158) |
| conf.md.p1 | 69 / 111 / 10 / 2 | 3.4 | -1 [-6, +4] | -2 [-6, +2] | +1 [+0, +3] | 179, -1 [-5, +3], 2 / 3 | 3 / 1 / 2 | 1 / 2 / 1 / 0 | -1 (175) |
| conf.rand.p1 | 70 / 111 / 9 / 2 | 3.4 | +0 [-4, +4] | -1 [-5, +2] | +1 [+0, +3] | 180, -1 [-5, +2], 1 / 2 | 2 / 1 / 1 | 0 / 2 / 1 / 0 | -1 (176) |
| conf.rand.m1 | 70 / 112 / 9 / 1 | 3.4 | +0 [-4, +4] | +0 [-3, +3] | +0 [+0, +0] | 181, +0 [-3, +3], 1 / 1 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +0 (179) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p1 +8 [-2, +18]; vs conf.rand.m1 +8 [-2, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-3.1, +2.1]; moved out / in 4 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +3 on 158 questions; mean words 3.4 -> 3.9
KEY family=E judge=pro layer=8 A=incomplete B=not_met dR=+8 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+8 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=met

=== 2026-11-13 12:52:25 [llama3.1-8b run=llama3.1-8b_side_E2 E tables gpunone e_on=E2] tables done ===

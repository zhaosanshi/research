# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 25

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 65 / 52 / 2 / 73 | 4.7 | +4 [-1, +10] | +0 [-4, +4] | +0 [-4, +4] | 115, +3 [-1, +8], 4 / 1 | 2 / 2 / 0 | 1 / 1 / 2 / 0 | +1 (105) |
| x.w.m1 | 60 / 57 / 2 / 73 | 4.5 | -1 [-5, +2] | +0 [-3, +3] | +0 [-3, +3] | 116, +0 [-3, +3], 1 / 1 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +0 (110) |
| x.md.p1 | 64 / 56 / 2 / 70 | 4.8 | +3 [-3, +9] | +3 [-1, +8] | -3 [-8, +1] | 116, +1 [-3, +6], 3 / 2 | 1 / 4 / 0 | 1 / 0 / 3 / 1 | +1 (109) |
| x.rand.p1 | 63 / 53 / 2 / 74 | 4.6 | +2 [+0, +5] | -1 [-3, +0] | +1 [+0, +3] | 116, +2 [+0, +5], 2 / 0 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +2 (112) |
| x.rand.m1 | 62 / 56 / 2 / 72 | 4.8 | +1 [-3, +5] | +1 [+0, +3] | -1 [-3, +0] | 117, +0 [-4, +4], 2 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | -1 (110) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 74 / 61 / 1 / 56; dR +3 [-1, +8]; answered-rate change +0.0 pp [-1.6, +1.6]; out / in 1 / 1; mean words 7.9 -> 7.9
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +2 [-4, +8]; vs x.rand.m1 +3 [-2, +8] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-2.1, +2.1]; moved out / in 2 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 105 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=25 A=not_met B=not_met dR=+4 dR_F=+3 dAns=+0 dAns_F=+0 rand_margin=+2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 25

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 47 / 46 / 0 / 99 | +3 [-1, +8] | +0 [-5, +5] | +2 [-2, +6] | 90, +1, 1 / 0 | 3 / 3 / 0 | +1 (85) |
| caut_E1.wq.m1 | 54 / 47 / 0 / 91 | 53 / 52 / 1 / 86 | -1 [-5, +3] | +4 [+0, +9] | -5 [-10, +0] | 100, -2, 0 / 2 | 1 / 5 / 0 | -2 (98) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 102 / 88 / 2 / 0 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, +2, 2 / 0 | 0 / 0 / 0 | +2 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-6, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 5 / 4 | 2 / 0 / 1 | +0 (176) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 97 / 92 / 3 / 0 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 1 / 3 | 1 / 0 / 1 | -1 (180) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 99 / 89 / 3 / 1 | -1 [-3, +0] | -2 [-5, +0] | +1 [+0, +3] | 188, +0, 0 / 0 | 2 / 0 / 1 | +0 (181) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 101 / 87 / 3 / 1 | +1 [-5, +8] | -2 [-5, +0] | +1 [+0, +3] | 188, +2, 6 / 4 | 2 / 0 / 1 | +1 (171) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 102 / 86 / 3 / 1 | +2 [-4, +8] | -2 [-5, +0] | +1 [+0, +3] | 188, +3, 6 / 3 | 2 / 0 / 1 | +1 (174) |
| conf_E1.wq.m1 | 105 / 83 / 3 / 1 | 100 / 88 / 3 / 1 | -5 [-12, +1] | +0 [-3, +3] | +0 [+0, +0] | 187, -5, 2 / 7 | 1 / 1 / 1 | -5 (175) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 74 / 61 / 1 / 56 | +3 [-1, +8] | +0 [-3, +3] | +0 [-3, +3] | 134, +4, 4 / 0 | 1 / 1 / 0 | +3 (126) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 25

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 101 / 87 / 3 / 1 | 5.4 | +1 [-5, +8] | -2 [-5, +0] | +1 [+0, +3] | 188, +2 [-4, +8], 6 / 4 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +1 (171) |
| conf.md.p1 | 100 / 88 / 3 / 1 | 4.8 | +0 [-6, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-5, +7], 5 / 4 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +0 (176) |
| conf.rand.p1 | 99 / 89 / 3 / 1 | 4.8 | -1 [-3, +0] | -2 [-5, +0] | +1 [+0, +3] | 188, +0 [+0, +0], 0 / 0 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +0 (181) |
| conf.rand.m1 | 97 / 92 / 3 / 0 | 5.0 | -3 [-8, +1] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-6, +2], 1 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | -1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +2 [-4, +8]; vs conf.rand.m1 +4 [-3, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 171 questions; mean words 4.9 -> 5.4
KEY family=E judge=allenai layer=25 A=not_met B=not_met dR=+1 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+2 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 25

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 55 / 59 / 4 / 74 | 4.7 | +1 [-5, +7] | -1 [-5, +3] | +0 [-4, +4] | 112, +1 [-3, +5], 3 / 2 | 3 / 2 / 1 | 2 / 1 / 2 / 0 | +1 (102) |
| x.w.m1 | 53 / 62 / 4 / 73 | 4.5 | -1 [-4, +2] | +0 [-3, +3] | -1 [-5, +2] | 114, +0 [-3, +3], 1 / 1 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +0 (108) |
| x.md.p1 | 56 / 62 / 3 / 71 | 4.8 | +2 [-3, +8] | +3 [-2, +8] | -3 [-8, +1] | 113, +1 [-2, +5], 2 / 1 | 2 / 5 / 1 | 2 / 0 / 3 / 2 | +0 (107) |
| x.rand.p1 | 52 / 60 / 6 / 74 | 4.6 | -2 [-5, +0] | -3 [-7, +0] | +0 [-3, +3] | 112, -1 [-3, +0], 0 / 1 | 3 / 0 / 2 | 1 / 2 / 0 / 0 | -1 (108) |
| x.rand.m1 | 54 / 61 / 4 / 73 | 4.8 | +0 [-3, +3] | +0 [-3, +3] | -1 [-3, +0] | 114, +0 [+0, +0], 0 / 0 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +0 (107) |

- F, complete-sentence background (base x_F): x_F 61 / 70 / 5 / 56 -> with the E write 64 / 67 / 5 / 56; dR +3 [-1, +8]; answered-rate change +0.0 pp [-1.6, +1.6]; out / in 1 / 1; mean words 7.9 -> 7.9
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +3 [-2, +8]; vs x.rand.m1 +1 [-5, +7] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-2.6, +1.6]; moved out / in 3 / 2 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 102 questions; mean words 4.8 -> 4.7
KEY family=E judge=pro layer=25 A=not_met B=not_met dR=+1 dR_F=+3 dAns=-1 dAns_F=+0 rand_margin=+1

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 25

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 42 / 52 / 1 / 97 | 42 / 49 / 2 / 99 | +0 [-5, +5] | -3 [-8, +1] | +2 [-2, +6] | 90, +1, 2 / 1 | 4 / 1 / 1 | +2 (85) |
| caut_E1.wq.m1 | 49 / 52 / 0 / 91 | 48 / 57 / 1 / 86 | -1 [-5, +2] | +4 [+0, +9] | -5 [-10, +0] | 100, +0, 1 / 1 | 1 / 5 / 0 | +0 (98) |
| conf.iso.p1 | 87 / 97 / 5 / 3 | 89 / 95 / 5 / 3 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 184, +2, 2 / 0 | 0 / 0 / 0 | +2 (180) |
| conf.md.p1 | 87 / 97 / 5 / 3 | 86 / 96 / 7 / 3 | -1 [-8, +7] | -2 [-6, +2] | +0 [+0, +0] | 181, +0, 6 / 6 | 3 / 1 / 3 | -1 (170) |
| conf.rand.m1 | 87 / 97 / 5 / 3 | 86 / 98 / 4 / 4 | -1 [-5, +2] | +0 [+0, +0] | +1 [+0, +3] | 184, -1, 1 / 2 | 0 / 0 / 0 | -2 (176) |
| conf.rand.p1 | 87 / 97 / 5 / 3 | 85 / 99 / 5 / 3 | -2 [-7, +3] | +0 [+0, +0] | +0 [+0, +0] | 184, -2, 2 / 4 | 0 / 0 / 0 | -2 (177) |
| conf.w.p1 | 87 / 97 / 5 / 3 | 86 / 96 / 6 / 4 | -1 [-8, +6] | -2 [-6, +2] | +1 [+0, +3] | 181, +0, 5 / 5 | 3 / 1 / 3 | -2 (166) |
| conf.wq.p1 | 87 / 97 / 5 / 3 | 89 / 93 / 6 / 4 | +2 [-5, +9] | -2 [-6, +2] | +1 [+0, +3] | 181, +3, 6 / 3 | 3 / 1 / 3 | +0 (169) |
| conf_E1.wq.m1 | 95 / 89 / 4 / 4 | 93 / 90 / 5 / 4 | -2 [-9, +5] | -1 [-5, +2] | +0 [+0, +0] | 182, -1, 5 / 6 | 2 / 1 / 2 | +1 (170) |
| x_F.w.p1 | 61 / 70 / 5 / 56 | 64 / 67 / 5 / 56 | +3 [-1, +8] | +0 [-3, +3] | +0 [-3, +3] | 130, +4, 4 / 0 | 1 / 1 / 0 | +3 (122) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 87 / 97 / 5 / 3; rule abstentions 0; mean words 4.9

## layer 25

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 86 / 96 / 6 / 4 | 5.4 | -1 [-8, +6] | -2 [-6, +2] | +1 [+0, +3] | 181, +0 [-6, +6], 5 / 5 | 3 / 1 / 3 | 2 / 1 / 1 / 0 | -2 (166) |
| conf.md.p1 | 86 / 96 / 7 / 3 | 4.8 | -1 [-8, +7] | -2 [-6, +2] | +0 [+0, +0] | 181, +0 [-7, +7], 6 / 6 | 3 / 1 / 3 | 2 / 1 / 1 / 0 | -1 (170) |
| conf.rand.p1 | 85 / 99 / 5 / 3 | 4.8 | -2 [-7, +3] | +0 [+0, +0] | +0 [+0, +0] | 184, -2 [-7, +3], 2 / 4 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -2 (177) |
| conf.rand.m1 | 86 / 98 / 4 / 4 | 5.0 | -1 [-5, +2] | +0 [+0, +0] | +1 [+0, +3] | 184, -1 [-5, +2], 1 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -2 (176) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +1 [-7, +9]; vs conf.rand.m1 +0 [-7, +7] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-3.1, +1.0]; moved out / in 3 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same -2 on 166 questions; mean words 4.9 -> 5.4
KEY family=E judge=pro layer=25 A=not_met B=not_met dR=-1 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+0 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 26

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 62 / 54 / 2 / 74 | 4.6 | +1 [-4, +6] | -1 [-4, +2] | +1 [-2, +4] | 115, +1 [-3, +5], 3 / 2 | 2 / 1 / 0 | 1 / 1 / 1 / 0 | +1 (107) |
| x.w.m1 | 61 / 56 / 2 / 73 | 4.5 | +0 [-4, +4] | +0 [-3, +3] | +0 [-3, +3] | 116, +1 [-2, +5], 2 / 1 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +1 (110) |
| x.md.p1 | 60 / 57 / 2 / 73 | 4.6 | -1 [-7, +5] | +0 [-4, +4] | +0 [-4, +4] | 115, -1 [-6, +4], 3 / 4 | 2 / 2 / 0 | 1 / 1 / 1 / 1 | +0 (110) |
| x.rand.p1 | 59 / 57 / 2 / 74 | 4.8 | -2 [-6, +2] | -1 [-3, +0] | +1 [+0, +3] | 116, -1 [-5, +2], 1 / 2 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | -2 (109) |
| x.rand.m1 | 60 / 58 / 2 / 72 | 4.7 | -1 [-3, +0] | +1 [+0, +3] | -1 [-3, +0] | 117, -1 [-3, +0], 0 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -1 (116) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 73 / 62 / 2 / 55; dR +2 [-3, +7]; answered-rate change +0.0 pp [-1.6, +1.6]; out / in 1 / 1; mean words 7.9 -> 7.8
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +3 [-2, +8]; vs x.rand.m1 +2 [-4, +8] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-2.1, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 107 questions; mean words 4.8 -> 4.6
KEY family=E judge=allenai layer=26 A=not_met B=not_met dR=+1 dR_F=+2 dAns=-1 dAns_F=+0 rand_margin=+2

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 26

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 47 / 47 / 0 / 98 | +3 [-2, +8] | +1 [-3, +5] | +1 [-2, +4] | 91, +1, 2 / 1 | 2 / 3 / 0 | +2 (85) |
| caut_E1.wq.m1 | 54 / 47 / 0 / 91 | 53 / 49 / 1 / 89 | -1 [-4, +2] | +1 [-2, +4] | -2 [-6, +2] | 100, -1, 0 / 1 | 1 / 2 / 0 | -1 (98) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 100 / 90 / 2 / 0 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, +0, 0 / 0 | 0 / 0 / 0 | +0 (186) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 98 / 90 / 3 / 1 | -2 [-8, +4] | -2 [-5, +0] | +1 [+0, +3] | 188, -1, 4 / 5 | 2 / 0 / 1 | -1 (178) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 101 / 88 / 3 / 0 | +1 [-3, +5] | -1 [-3, +0] | +0 [+0, +0] | 189, +1, 3 / 2 | 1 / 0 / 1 | +1 (186) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 101 / 89 / 2 / 0 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, +1, 2 / 1 | 0 / 0 / 0 | +0 (183) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 100 / 88 / 3 / 1 | +0 [-6, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, +1, 4 / 3 | 2 / 0 / 1 | +1 (175) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 101 / 87 / 3 / 1 | +1 [-5, +7] | -2 [-5, +0] | +1 [+0, +3] | 188, +2, 5 / 3 | 2 / 0 / 1 | +2 (175) |
| conf_E1.wq.m1 | 105 / 83 / 3 / 1 | 100 / 88 / 3 / 1 | -5 [-11, +1] | +0 [-3, +3] | +0 [+0, +0] | 187, -5, 1 / 6 | 1 / 1 / 1 | -4 (175) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 73 / 62 / 2 / 55 | +2 [-3, +7] | +0 [-3, +3] | -1 [-3, +0] | 134, +3, 4 / 1 | 1 / 1 / 1 | +2 (129) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 26

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 88 / 3 / 1 | 5.1 | +0 [-6, +6] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-4, +6], 4 / 3 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +1 (175) |
| conf.md.p1 | 98 / 90 / 3 / 1 | 4.7 | -2 [-8, +4] | -2 [-5, +0] | +1 [+0, +3] | 188, -1 [-7, +5], 4 / 5 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (178) |
| conf.rand.p1 | 101 / 89 / 2 / 0 | 4.9 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, +1 [-2, +5], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (183) |
| conf.rand.m1 | 101 / 88 / 3 / 0 | 4.8 | +1 [-3, +5] | -1 [-3, +0] | +0 [+0, +0] | 189, +1 [-3, +5], 3 / 2 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | +1 (186) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 -1 [-7, +5]; vs conf.rand.m1 -1 [-6, +4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 175 questions; mean words 4.9 -> 5.1
KEY family=E judge=allenai layer=26 A=not_met B=not_met dR=+0 dR_F=NA dAns=-2 dAns_F=NA rand_margin=-1 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 26

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 52 / 61 / 4 / 75 | 4.6 | -2 [-8, +3] | -2 [-6, +2] | +1 [-2, +4] | 112, -1 [-5, +3], 2 / 3 | 3 / 1 / 1 | 2 / 1 / 1 / 0 | +0 (104) |
| x.w.m1 | 54 / 61 / 4 / 73 | 4.5 | +0 [-4, +4] | +0 [-3, +3] | -1 [-5, +2] | 114, +1 [-2, +5], 2 / 1 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +1 (108) |
| x.md.p1 | 53 / 63 / 2 / 74 | 4.6 | -1 [-6, +4] | +1 [-4, +6] | +0 [-4, +4] | 112, +0 [-4, +4], 2 / 2 | 3 / 4 / 1 | 2 / 1 / 1 / 3 | -1 (109) |
| x.rand.p1 | 52 / 62 / 3 / 75 | 4.8 | -2 [-5, +0] | -1 [-3, +0] | +1 [+0, +3] | 114, -1 [-3, +0], 0 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | -1 (107) |
| x.rand.m1 | 53 / 63 / 3 / 73 | 4.7 | -1 [-3, +0] | +1 [+0, +3] | -1 [-3, +0] | 115, -1 [-3, +0], 0 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -1 (114) |

- F, complete-sentence background (base x_F): x_F 61 / 70 / 5 / 56 -> with the E write 66 / 66 / 5 / 55; dR +5 [+1, +10]; answered-rate change +0.5 pp [+0.0, +1.6]; out / in 0 / 1; mean words 7.9 -> 7.8
- main effect: dR -2 (-1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +0 [-5, +6]; vs x.rand.m1 -1 [-6, +4] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: True
- retention on x: answered-rate change -1.0 pp [-3.1, +1.0]; moved out / in 3 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +0 on 104 questions; mean words 4.8 -> 4.6
KEY family=E judge=pro layer=26 A=not_met B=not_met dR=-2 dR_F=+5 dAns=-2 dAns_F=+1 rand_margin=-1

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 26

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 42 / 52 / 1 / 97 | 42 / 51 / 1 / 98 | +0 [-4, +4] | -1 [-4, +2] | +1 [-2, +4] | 92, +0, 1 / 1 | 2 / 1 / 0 | +1 (85) |
| caut_E1.wq.m1 | 49 / 52 / 0 / 91 | 48 / 54 / 1 / 89 | -1 [-3, +0] | +1 [-2, +4] | -2 [-6, +2] | 100, +0, 0 / 0 | 1 / 2 / 0 | +0 (98) |
| conf.iso.p1 | 87 / 97 / 5 / 3 | 86 / 97 / 6 / 3 | -1 [-3, +0] | -1 [-3, +0] | +0 [+0, +0] | 183, +0, 0 / 0 | 1 / 0 / 1 | +0 (180) |
| conf.md.p1 | 87 / 97 / 5 / 3 | 85 / 97 / 7 / 3 | -2 [-9, +5] | -2 [-6, +2] | +0 [+0, +0] | 181, -1, 5 / 6 | 3 / 1 / 3 | -1 (172) |
| conf.rand.m1 | 87 / 97 / 5 / 3 | 90 / 94 / 5 / 3 | +3 [-1, +8] | +0 [+0, +0] | +0 [+0, +0] | 184, +3, 4 / 1 | 0 / 0 / 0 | +3 (181) |
| conf.rand.p1 | 87 / 97 / 5 / 3 | 88 / 96 / 5 / 3 | +1 [-2, +4] | +0 [+0, +0] | +0 [+0, +0] | 184, +1, 2 / 1 | 0 / 0 / 0 | +1 (178) |
| conf.w.p1 | 87 / 97 / 5 / 3 | 86 / 96 / 6 / 4 | -1 [-7, +5] | -2 [-5, +0] | +1 [+0, +3] | 182, +0, 4 / 4 | 2 / 0 / 2 | -1 (171) |
| conf.wq.p1 | 87 / 97 / 5 / 3 | 88 / 95 / 6 / 3 | +1 [-5, +8] | -1 [-5, +2] | +0 [+0, +0] | 182, +1, 5 / 4 | 2 / 1 / 2 | +0 (170) |
| conf_E1.wq.m1 | 95 / 89 / 4 / 4 | 91 / 92 / 5 / 4 | -4 [-11, +3] | -1 [-4, +2] | +0 [+0, +0] | 182, -3, 4 / 7 | 2 / 1 / 2 | +0 (170) |
| x_F.w.p1 | 61 / 70 / 5 / 56 | 66 / 66 / 5 / 55 | +5 [+1, +10] | +1 [+0, +3] | -1 [-3, +0] | 131, +5, 5 / 0 | 0 / 1 / 0 | +4 (126) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 87 / 97 / 5 / 3; rule abstentions 0; mean words 4.9

## layer 26

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 86 / 96 / 6 / 4 | 5.1 | -1 [-7, +5] | -2 [-5, +0] | +1 [+0, +3] | 182, +0 [-6, +6], 4 / 4 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | -1 (171) |
| conf.md.p1 | 85 / 97 / 7 / 3 | 4.7 | -2 [-9, +5] | -2 [-6, +2] | +0 [+0, +0] | 181, -1 [-7, +5], 5 / 6 | 3 / 1 / 3 | 2 / 1 / 1 / 0 | -1 (172) |
| conf.rand.p1 | 88 / 96 / 5 / 3 | 4.9 | +1 [-2, +4] | +0 [+0, +0] | +0 [+0, +0] | 184, +1 [-2, +4], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (178) |
| conf.rand.m1 | 90 / 94 / 5 / 3 | 4.8 | +3 [-1, +8] | +0 [+0, +0] | +0 [+0, +0] | 184, +3 [-1, +8], 4 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +3 (181) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 -2 [-7, +3]; vs conf.rand.m1 -4 [-10, +1] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -1 on 171 questions; mean words 4.9 -> 5.1
KEY family=E judge=pro layer=26 A=not_met B=not_met dR=-1 dR_F=NA dAns=-2 dAns_F=NA rand_margin=-4 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 64 / 52 / 2 / 74 | 4.7 | +3 [-2, +8] | -1 [-4, +2] | +1 [-2, +4] | 115, +3 [-1, +8], 4 / 1 | 2 / 1 / 0 | 1 / 1 / 1 / 0 | +1 (107) |
| x.w.m1 | 62 / 54 / 2 / 74 | 4.5 | +1 [-3, +6] | -1 [-5, +2] | +1 [-2, +5] | 115, +2 [-2, +6], 3 / 1 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +2 (111) |
| x.md.p1 | 63 / 54 / 2 / 73 | 4.7 | +2 [-3, +7] | +0 [-3, +3] | +0 [-3, +3] | 116, +2 [-2, +6], 3 / 1 | 1 / 1 / 0 | 1 / 0 / 1 / 0 | +2 (109) |
| x.rand.p1 | 62 / 56 / 2 / 72 | 4.8 | +1 [+0, +3] | +1 [+0, +3] | -1 [-3, +0] | 117, +1 [+0, +3], 1 / 0 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (113) |
| x.rand.m1 | 63 / 53 / 2 / 74 | 4.8 | +2 [-2, +6] | -1 [-3, +0] | +1 [+0, +3] | 116, +2 [-2, +6], 3 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +2 (112) |

- F, complete-sentence background (base x_F): x_F 71 / 64 / 1 / 56 -> with the E write 69 / 66 / 2 / 55; dR -2 [-6, +2]; answered-rate change +0.0 pp [-1.6, +1.6]; out / in 1 / 1; mean words 7.9 -> 8.0
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +2 [-3, +7]; vs x.rand.m1 +1 [-5, +7] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-2.1, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 107 questions; mean words 4.8 -> 4.7
KEY family=E judge=allenai layer=27 A=not_met B=not_met dR=+3 dR_F=-2 dAns=-1 dAns_F=+0 rand_margin=+1

# E arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 27

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 44 / 49 / 2 / 97 | 49 / 46 / 0 / 97 | +5 [+1, +10] | +2 [-2, +6] | +0 [-3, +3] | 92, +2, 2 / 0 | 1 / 3 / 0 | +1 (89) |
| caut_E1.wq.m1 | 54 / 47 / 0 / 91 | 53 / 48 / 0 / 91 | -1 [-3, +0] | +0 [-3, +3] | +0 [-3, +3] | 100, +0, 0 / 0 | 1 / 1 / 0 | +0 (99) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 100 / 90 / 2 / 0 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 190, +0, 0 / 0 | 0 / 0 / 0 | +0 (185) |
| conf.md.p1 | 100 / 90 / 2 / 0 | 97 / 91 / 3 / 1 | -3 [-10, +3] | -2 [-5, +0] | +1 [+0, +3] | 188, -2, 4 / 6 | 2 / 0 / 1 | -1 (179) |
| conf.rand.m1 | 100 / 90 / 2 / 0 | 98 / 91 / 2 / 1 | -2 [-6, +2] | -1 [-3, +0] | +1 [+0, +3] | 189, -1, 1 / 2 | 1 / 0 / 0 | +0 (185) |
| conf.rand.p1 | 100 / 90 / 2 / 0 | 98 / 91 / 3 / 0 | -2 [-6, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -2, 1 / 3 | 1 / 0 / 1 | -2 (182) |
| conf.w.p1 | 100 / 90 / 2 / 0 | 100 / 89 / 3 / 0 | +0 [-6, +6] | -1 [-3, +0] | +0 [+0, +0] | 189, +0, 4 / 4 | 1 / 0 / 1 | -1 (175) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 103 / 86 / 3 / 0 | +3 [-3, +9] | -1 [-3, +0] | +0 [+0, +0] | 189, +3, 6 / 3 | 1 / 0 / 1 | +1 (175) |
| conf_E1.wq.m1 | 105 / 83 / 3 / 1 | 102 / 86 / 3 / 1 | -3 [-8, +2] | +0 [-3, +3] | +0 [+0, +0] | 187, -3, 1 / 4 | 1 / 1 / 1 | -2 (177) |
| x_F.w.p1 | 71 / 64 / 1 / 56 | 69 / 66 / 2 / 55 | -2 [-6, +2] | +0 [-3, +3] | -1 [-3, +0] | 134, -1, 1 / 2 | 1 / 1 / 1 | -1 (128) |

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 100 / 89 / 3 / 0 | 4.9 | +0 [-6, +6] | -1 [-3, +0] | +0 [+0, +0] | 189, +0 [-6, +6], 4 / 4 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -1 (175) |
| conf.md.p1 | 97 / 91 / 3 / 1 | 4.8 | -3 [-10, +3] | -2 [-5, +0] | +1 [+0, +3] | 188, -2 [-8, +4], 4 / 6 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -1 (179) |
| conf.rand.p1 | 98 / 91 / 3 / 0 | 5.0 | -2 [-6, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -2 [-6, +2], 1 / 3 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -2 (182) |
| conf.rand.m1 | 98 / 91 / 2 / 1 | 4.7 | -2 [-6, +2] | -1 [-3, +0] | +1 [+0, +3] | 189, -1 [-5, +2], 1 / 2 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +0 (185) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 +2 [-4, +8]; vs conf.rand.m1 +2 [-4, +8] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-1.6, +0.0]; moved out / in 1 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same -1 on 175 questions; mean words 4.9 -> 4.9
KEY family=E judge=allenai layer=27 A=not_met B=not_met dR=+0 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+2 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 54 / 59 / 4 / 75 | 4.7 | +0 [-5, +5] | -2 [-6, +2] | +1 [-2, +4] | 112, +1 [-2, +5], 2 / 1 | 3 / 1 / 1 | 2 / 1 / 1 / 0 | +0 (104) |
| x.w.m1 | 54 / 60 / 4 / 74 | 4.5 | +0 [-4, +4] | -1 [-5, +2] | +0 [-4, +4] | 113, +1 [-2, +5], 2 / 1 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +1 (109) |
| x.md.p1 | 55 / 60 / 3 / 74 | 4.7 | +1 [-3, +6] | +0 [-4, +4] | +0 [-3, +3] | 113, +2 [+0, +5], 2 / 0 | 2 / 2 / 1 | 2 / 0 / 1 / 1 | +1 (107) |
| x.rand.p1 | 54 / 62 / 3 / 73 | 4.8 | +0 [+0, +0] | +1 [+0, +3] | -1 [-3, +0] | 115, +0 [+0, +0], 0 / 0 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (111) |
| x.rand.m1 | 54 / 60 / 3 / 75 | 4.8 | +0 [-3, +3] | -1 [-3, +0] | +1 [+0, +3] | 114, +0 [-3, +3], 1 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +0 (110) |

- F, complete-sentence background (base x_F): x_F 61 / 70 / 5 / 56 -> with the E write 64 / 68 / 5 / 55; dR +3 [+0, +7]; answered-rate change +0.5 pp [+0.0, +1.6]; out / in 0 / 1; mean words 7.9 -> 8.0
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +0 [-5, +5]; vs x.rand.m1 +0 [-6, +6] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.0 pp [-3.1, +1.0]; moved out / in 3 / 1 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +0 on 104 questions; mean words 4.8 -> 4.7
KEY family=E judge=pro layer=27 A=not_met B=not_met dR=+0 dR_F=+3 dAns=-2 dAns_F=+1 rand_margin=+0

# E arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 27

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.p1 | 42 / 52 / 1 / 97 | 43 / 51 / 1 / 97 | +1 [-2, +5] | +0 [-3, +3] | +0 [-3, +3] | 93, +0, 1 / 1 | 1 / 1 / 0 | +1 (89) |
| caut_E1.wq.m1 | 49 / 52 / 0 / 91 | 48 / 53 / 0 / 91 | -1 [-3, +0] | +0 [-3, +3] | +0 [-3, +3] | 100, +0, 0 / 0 | 1 / 1 / 0 | +0 (99) |
| conf.iso.p1 | 87 / 97 / 5 / 3 | 87 / 97 / 5 / 3 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 184, +0, 0 / 0 | 0 / 0 / 0 | +0 (179) |
| conf.md.p1 | 87 / 97 / 5 / 3 | 85 / 96 / 8 / 3 | -2 [-9, +5] | -3 [-7, +0] | +0 [+0, +0] | 181, +0, 5 / 5 | 3 / 0 / 3 | -2 (173) |
| conf.rand.m1 | 87 / 97 / 5 / 3 | 87 / 97 / 5 / 3 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 184, +0, 2 / 2 | 0 / 0 / 0 | -1 (180) |
| conf.rand.p1 | 87 / 97 / 5 / 3 | 88 / 96 / 5 / 3 | +1 [-2, +4] | +0 [+0, +0] | +0 [+0, +0] | 184, +1, 2 / 1 | 0 / 0 / 0 | +1 (178) |
| conf.w.p1 | 87 / 97 / 5 / 3 | 86 / 95 / 7 / 4 | -1 [-7, +6] | -3 [-7, +0] | +1 [+0, +3] | 181, +1, 5 / 4 | 3 / 0 / 3 | -1 (169) |
| conf.wq.p1 | 87 / 97 / 5 / 3 | 88 / 94 / 6 / 4 | +1 [-5, +8] | -2 [-6, +2] | +1 [+0, +3] | 181, +2, 5 / 3 | 3 / 1 / 3 | +0 (169) |
| conf_E1.wq.m1 | 95 / 89 / 4 / 4 | 94 / 89 / 5 / 4 | -1 [-7, +5] | -1 [-4, +2] | +0 [+0, +0] | 182, +0, 3 / 3 | 2 / 1 / 2 | +1 (172) |
| x_F.w.p1 | 61 / 70 / 5 / 56 | 64 / 68 / 5 / 55 | +3 [+0, +7] | +1 [+0, +3] | -1 [-3, +0] | 131, +3, 3 / 0 | 0 / 1 / 0 | +2 (125) |

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p1, strength 1 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 87 / 97 / 5 / 3; rule abstentions 0; mean words 4.9

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p1 | 86 / 95 / 7 / 4 | 4.9 | -1 [-7, +6] | -3 [-7, +0] | +1 [+0, +3] | 181, +1 [-5, +7], 5 / 4 | 3 / 0 / 3 | 2 / 1 / 0 / 0 | -1 (169) |
| conf.md.p1 | 85 / 96 / 8 / 3 | 4.8 | -2 [-9, +5] | -3 [-7, +0] | +0 [+0, +0] | 181, +0 [-6, +6], 5 / 5 | 3 / 0 / 3 | 2 / 1 / 0 / 0 | -2 (173) |
| conf.rand.p1 | 88 / 96 / 5 / 3 | 5.0 | +1 [-2, +4] | +0 [+0, +0] | +0 [+0, +0] | 184, +1 [-2, +4], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (178) |
| conf.rand.m1 | 87 / 97 / 5 / 3 | 4.7 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 184, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (180) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR -1 (-0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p1 -2 [-8, +4]; vs conf.rand.m1 -1 [-8, +6] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same -1 on 169 questions; mean words 4.9 -> 4.9
KEY family=E judge=pro layer=27 A=not_met B=not_met dR=-1 dR_F=NA dAns=-3 dAns_F=NA rand_margin=-2 EXPLORATORY base=conf main=conf.w.p1 strength=1 tested_A_without_F=not_met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 25

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 108 / 79 / 4 / 1 | 5.3 | +8 [-1, +17] | -3 [-7, +0] | +1 [+0, +3] | 187, +10 [+2, +18], 14 / 4 | 3 / 0 / 2 | 2 / 1 / 0 / 0 | +6 (160) |
| conf.md.p2 | 99 / 89 / 3 / 1 | 4.8 | -1 [-10, +8] | -2 [-5, +0] | +1 [+0, +3] | 188, +0 [-9, +9], 10 / 10 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +1 (171) |
| conf.rand.p2 | 96 / 92 / 3 / 1 | 4.8 | -4 [-9, +1] | -2 [-5, +0] | +1 [+0, +3] | 188, -3 [-8, +1], 1 / 4 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | -3 (175) |
| conf.rand.m2 | 98 / 91 / 3 / 0 | 4.7 | -2 [-8, +4] | -1 [-3, +0] | +0 [+0, +0] | 189, -1 [-7, +5], 4 / 5 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (179) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs conf.rand.p2 +12 [+4, +20]; vs conf.rand.m2 +10 [+1, +19] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 160 questions; mean words 4.9 -> 5.3
KEY family=E judge=allenai layer=25 A=incomplete B=not_met dR=+8 dR_F=NA dAns=-3 dAns_F=NA rand_margin=+10 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 87 / 97 / 5 / 3; rule abstentions 0; mean words 4.9

## layer 25

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 96 / 87 / 5 / 4 | 5.3 | +9 [+1, +18] | -1 [-5, +2] | +1 [+0, +3] | 182, +9 [+1, +17], 13 / 4 | 2 / 1 / 2 | 1 / 1 / 1 / 0 | +4 (155) |
| conf.md.p2 | 89 / 94 / 5 / 4 | 4.8 | +2 [-7, +11] | -1 [-5, +2] | +1 [+0, +3] | 182, +2 [-6, +10], 10 / 8 | 2 / 1 / 2 | 1 / 1 / 1 / 0 | +1 (167) |
| conf.rand.p2 | 82 / 102 / 5 / 3 | 4.8 | -5 [-11, +1] | +0 [+0, +0] | +0 [+0, +0] | 184, -5 [-11, +1], 2 / 7 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -5 (172) |
| conf.rand.m2 | 87 / 97 / 4 / 4 | 4.7 | +0 [-4, +4] | +0 [+0, +0] | +1 [+0, +3] | 184, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (175) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +9 (4.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs conf.rand.p2 +14 [+6, +23]; vs conf.rand.m2 +9 [+0, +18] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +4 on 155 questions; mean words 4.9 -> 5.3
KEY family=E judge=pro layer=25 A=incomplete B=not_met dR=+9 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+9 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=met

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 100 / 90 / 2 / 0; rule abstentions 0; mean words 4.9

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 101 / 87 / 4 / 0 | 5.1 | +1 [-7, +9] | -2 [-5, +0] | +0 [+0, +0] | 188, +2 [-5, +9], 8 / 6 | 2 / 0 / 2 | 1 / 1 / 0 / 0 | +2 (169) |
| conf.md.p2 | 100 / 88 / 3 / 1 | 4.7 | +0 [-8, +8] | -2 [-5, +0] | +1 [+0, +3] | 188, +1 [-6, +9], 8 / 7 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +1 (176) |
| conf.rand.p2 | 96 / 93 / 3 / 0 | 4.8 | -4 [-11, +2] | -1 [-3, +0] | +0 [+0, +0] | 189, -4 [-11, +2], 3 / 7 | 1 / 0 / 1 | 0 / 1 / 0 / 0 | -3 (179) |
| conf.rand.m2 | 94 / 93 / 4 / 1 | 4.5 | -6 [-12, +0] | -3 [-7, +0] | +1 [+0, +3] | 187, -3 [-8, +2], 2 / 5 | 3 / 0 / 2 | 3 / 0 / 0 / 0 | -3 (182) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p2 +5 [-3, +14]; vs conf.rand.m2 +7 [-2, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -1.0 pp [-2.6, +0.0]; moved out / in 2 / 0 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +2 on 169 questions; mean words 4.9 -> 5.1
KEY family=E judge=allenai layer=27 A=not_met B=not_met dR=+1 dR_F=NA dAns=-2 dAns_F=NA rand_margin=+5 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=not_met

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_conf)
EXPLORATORY: base prompt conf, main arm conf.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline conf: right / wrong / hedge-or-uninformative / abstained = 87 / 97 / 5 / 3; rule abstentions 0; mean words 4.9

## layer 27

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| conf.w.p2 | 89 / 94 / 5 / 4 | 5.1 | +2 [-5, +9] | -1 [-5, +2] | +1 [+0, +3] | 182, +2 [-5, +9], 7 / 5 | 2 / 1 / 2 | 1 / 1 / 1 / 0 | +1 (163) |
| conf.md.p2 | 87 / 95 / 7 / 3 | 4.7 | +0 [-9, +9] | -2 [-6, +2] | +0 [+0, +0] | 181, +1 [-7, +9], 9 / 8 | 3 / 1 / 3 | 2 / 1 / 1 / 0 | +1 (170) |
| conf.rand.p2 | 84 / 98 / 7 / 3 | 4.8 | -3 [-10, +3] | -2 [-5, +0] | +0 [+0, +0] | 182, -1 [-7, +5], 4 / 5 | 2 / 0 / 2 | 2 / 0 / 0 / 0 | -1 (175) |
| conf.rand.m2 | 88 / 96 / 5 / 3 | 4.5 | +1 [-4, +6] | +0 [+0, +0] | +0 [+0, +0] | 184, +1 [-4, +6], 4 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (178) |

- F, complete-sentence background: NOT TESTED for base conf (no such prompt was frozen); the column status stays incomplete
- main effect: dR +2 (1.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs conf.rand.p2 +5 [-2, +12]; vs conf.rand.m2 +1 [-7, +9] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: not tested | B: dR interval above 0 and answered-rate interval inside: not tested
- retention on conf: answered-rate change -0.5 pp [-2.6, +1.0]; moved out / in 2 / 1 | A: point inside: True | B: whole interval inside: True
- diagnostic: G_same +1 on 163 questions; mean words 4.9 -> 5.1
KEY family=E judge=pro layer=27 A=not_met B=not_met dR=+2 dR_F=NA dAns=-1 dAns_F=NA rand_margin=+1 EXPLORATORY base=conf main=conf.w.p2 strength=2 tested_A_without_F=not_met

=== 2026-11-13 15:55:22 [qwen3-8b E tables gpunone] tables done ===

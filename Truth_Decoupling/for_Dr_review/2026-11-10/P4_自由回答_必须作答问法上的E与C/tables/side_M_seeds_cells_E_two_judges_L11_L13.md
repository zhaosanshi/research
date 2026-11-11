# seed 1051
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 95 / 72 / 2 / 23 | 6.6 | +17 [+6, +28] | +1 [-6, +8] | +0 [-6, +6] | 160, +13 [+3, +23], 20 / 7 | 6 / 7 / 1 | 2 / 4 / 6 / 1 | +8 (109) |
| x.w.m1 | 61 / 105 / 4 / 22 | 3.2 | -17 [-27, -7] | +0 [-5, +5] | -1 [-6, +4] | 162, -14 [-24, -5], 5 / 19 | 4 / 4 / 1 | 4 / 0 / 1 / 3 | -14 (161) |
| x.md.p1 | 86 / 83 / 2 / 21 | 4.5 | +8 [+2, +15] | +3 [-1, +8] | -2 [-5, +0] | 165, +6 [+1, +12], 7 / 1 | 1 / 4 / 1 | 1 / 0 / 3 / 1 | +3 (149) |
| x.rand.p1 | 79 / 89 / 2 / 22 | 4.0 | +1 [-3, +5] | +2 [+0, +5] | -1 [-3, +0] | 166, +0 [-4, +4], 2 / 2 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +0 (160) |
| x.rand.m1 | 81 / 82 / 2 / 27 | 4.1 | +3 [-3, +9] | -3 [-8, +1] | +4 [+1, +8] | 162, +4 [+0, +9], 5 / 1 | 4 / 1 / 0 | 2 / 2 / 1 / 0 | +3 (155) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 114 / 51 / 1 / 26; dR +12 [+0, +24]; answered-rate change +2.1 pp [-1.6, +5.7]; out / in 4 / 8; mean words 9.3 -> 12.7
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +16 [+5, +27]; vs x.rand.m1 +14 [+4, +24] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.5 pp [-3.1, +4.2]; moved out / in 6 / 7 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +8 on 109 questions; mean words 4.0 -> 6.6
KEY family=E judge=allenai layer=11 A=met B=not_met dR=+17 dR_F=+12 dAns=+1 dAns_F=+4 rand_margin=+14


# seed 1051
# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 73 / 89 / 5 / 25 | 6.6 | +9 [-1, +19] | +0 [-7, +7] | +1 [-5, +7] | 156, +6 [-3, +15], 14 / 8 | 6 / 6 / 1 | 1 / 5 / 4 / 2 | +7 (106) |
| x.w.m1 | 52 / 113 / 5 / 22 | 3.2 | -12 [-21, -3] | +3 [-3, +9] | -2 [-7, +3] | 159, -10 [-18, -2], 4 / 14 | 3 / 6 / 1 | 3 / 0 / 1 / 5 | -10 (158) |
| x.md.p1 | 68 / 97 / 5 / 22 | 4.5 | +4 [-2, +10] | +3 [+0, +7] | -2 [-5, +0] | 162, +2 [-3, +8], 5 / 3 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | +2 (147) |
| x.rand.p1 | 66 / 99 / 4 / 23 | 4.0 | +2 [-2, +6] | +3 [+0, +7] | -1 [-3, +0] | 162, +0 [-3, +3], 1 / 1 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | +1 (157) |
| x.rand.m1 | 64 / 95 / 5 / 28 | 4.1 | +0 [-5, +6] | -3 [-8, +1] | +4 [+1, +8] | 158, -1 [-6, +4], 3 / 4 | 4 / 1 / 0 | 0 / 4 / 1 / 0 | +0 (151) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 84 / 80 / 2 / 26; dR +4 [-8, +16]; answered-rate change +2.1 pp [-1.6, +5.7]; out / in 4 / 8; mean words 9.3 -> 12.7
- main effect: dR +9 (4.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +7 [-3, +17]; vs x.rand.m1 +9 [-1, +19] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-3.6, +3.6]; moved out / in 6 / 6 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +7 on 106 questions; mean words 4.0 -> 6.6
KEY family=E judge=pro layer=11 A=met B=not_met dR=+9 dR_F=+4 dAns=+0 dAns_F=+4 rand_margin=+7


# seed 1051
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 93 / 73 / 2 / 24 | 7.5 | +15 [+5, +26] | +0 [-6, +6] | +1 [-4, +6] | 161, +12 [+2, +22], 19 / 7 | 5 / 5 / 1 | 1 / 4 / 4 / 1 | +5 (97) |
| x.w.m1 | 70 / 105 / 2 / 15 | 3.1 | -8 [-20, +4] | +9 [+2, +16] | -8 [-14, -2] | 164, -12 [-23, -2], 9 / 21 | 2 / 11 / 1 | 2 / 0 / 6 / 5 | -13 (161) |
| x.md.p1 | 88 / 79 / 2 / 23 | 6.1 | +10 [-1, +21] | +1 [-5, +7] | +0 [-5, +5] | 162, +9 [-1, +19], 17 / 8 | 4 / 5 / 1 | 3 / 1 / 4 / 1 | +5 (115) |
| x.rand.p1 | 78 / 87 / 3 / 24 | 4.2 | +0 [-6, +6] | -1 [-3, +0] | +1 [+0, +3] | 165, +0 [-6, +6], 5 / 5 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -1 (157) |
| x.rand.m1 | 79 / 84 / 2 / 27 | 4.3 | +1 [-6, +8] | -3 [-8, +1] | +4 [+1, +8] | 162, +2 [-4, +8], 6 / 4 | 4 / 1 / 0 | 2 / 2 / 1 / 0 | -1 (153) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 105 / 57 / 2 / 28; dR +3 [-9, +15]; answered-rate change +0.5 pp [-2.1, +3.1]; out / in 3 / 4; mean words 9.3 -> 11.5
- main effect: dR +15 (7.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +15 [+6, +25]; vs x.rand.m1 +14 [+3, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-3.1, +3.1]; moved out / in 5 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 97 questions; mean words 4.0 -> 7.5
KEY family=E judge=allenai layer=13 A=met B=not_met dR=+15 dR_F=+3 dAns=+0 dAns_F=+1 rand_margin=+14


# seed 1051
# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 65 / 94 / 7 / 26 | 7.5 | +1 [-9, +11] | -3 [-9, +3] | +2 [-3, +7] | 156, +2 [-8, +11], 13 / 11 | 6 / 3 / 2 | 2 / 4 / 1 / 2 | +5 (94) |
| x.w.m1 | 60 / 114 / 3 / 15 | 3.1 | -4 [-14, +6] | +12 [+6, +19] | -9 [-15, -4] | 162, -7 [-17, +2], 8 / 15 | 0 / 12 / 0 | 0 / 0 / 3 / 9 | -7 (159) |
| x.md.p1 | 62 / 100 / 8 / 22 | 6.1 | -2 [-12, +8] | +0 [-5, +5] | -2 [-6, +2] | 159, -2 [-11, +7], 9 / 11 | 3 / 3 / 2 | 2 / 1 / 2 / 1 | +0 (113) |
| x.rand.p1 | 65 / 96 / 6 / 25 | 4.2 | +1 [-5, +7] | -1 [-3, +0] | +1 [+0, +3] | 161, +1 [-5, +7], 5 / 4 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +4 (152) |
| x.rand.m1 | 64 / 96 / 4 / 28 | 4.3 | +0 [-6, +6] | -2 [-7, +3] | +4 [+1, +8] | 158, -2 [-7, +3], 2 / 4 | 4 / 2 / 0 | 0 / 4 / 2 / 0 | -1 (150) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 84 / 78 / 2 / 28; dR +4 [-7, +15]; answered-rate change +1.0 pp [-1.6, +4.2]; out / in 3 / 5; mean words 9.3 -> 11.5
- main effect: dR +1 (0.5% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +0 [-10, +10]; vs x.rand.m1 +1 [-9, +10] | A: above each (point): False | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.6 pp [-4.7, +1.6]; moved out / in 6 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 94 questions; mean words 4.0 -> 7.5
KEY family=E judge=pro layer=13 A=not_met B=not_met dR=+1 dR_F=+4 dAns=-3 dAns_F=+2 rand_margin=+0


# seed 2045
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 96 / 70 / 2 / 24 | 6.7 | +18 [+7, +29] | +0 [-7, +7] | +1 [-5, +7] | 160, +14 [+4, +24], 21 / 7 | 6 / 6 / 1 | 2 / 4 / 6 / 0 | +8 (106) |
| x.w.m1 | 62 / 104 / 4 / 22 | 3.2 | -16 [-26, -6] | +0 [-5, +5] | -1 [-6, +4] | 162, -13 [-22, -4], 5 / 18 | 4 / 4 / 1 | 4 / 0 / 1 / 3 | -13 (161) |
| x.md.p1 | 86 / 83 / 2 / 21 | 4.5 | +8 [+2, +15] | +3 [-1, +8] | -2 [-5, +0] | 165, +6 [+1, +12], 7 / 1 | 1 / 4 / 1 | 1 / 0 / 3 / 1 | +3 (149) |
| x.rand.p1 | 81 / 87 / 2 / 22 | 4.1 | +3 [-1, +8] | +2 [+0, +5] | -1 [-3, +0] | 166, +2 [-2, +6], 3 / 1 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +1 (162) |
| x.rand.m1 | 81 / 84 / 1 / 26 | 4.2 | +3 [-3, +9] | -1 [-5, +3] | +3 [+0, +7] | 163, +4 [+0, +9], 5 / 1 | 3 / 2 / 0 | 2 / 1 / 1 / 1 | +4 (157) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 114 / 51 / 1 / 26; dR +12 [+0, +24]; answered-rate change +2.1 pp [-1.6, +5.7]; out / in 4 / 8; mean words 9.3 -> 12.7
- main effect: dR +18 (9.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +15 [+4, +26]; vs x.rand.m1 +15 [+5, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-3.6, +3.6]; moved out / in 6 / 6 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +8 on 106 questions; mean words 4.0 -> 6.7
KEY family=E judge=allenai layer=11 A=met B=not_met dR=+18 dR_F=+12 dAns=+0 dAns_F=+4 rand_margin=+15


# seed 2045
# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 72 / 90 / 5 / 25 | 6.7 | +8 [-2, +18] | +0 [-7, +7] | +1 [-5, +7] | 156, +4 [-5, +13], 13 / 9 | 6 / 6 / 1 | 1 / 5 / 5 / 1 | +5 (103) |
| x.w.m1 | 54 / 111 / 5 / 22 | 3.2 | -10 [-19, -1] | +3 [-3, +9] | -2 [-7, +3] | 159, -8 [-16, +0], 5 / 13 | 3 / 6 / 1 | 3 / 0 / 1 / 5 | -8 (158) |
| x.md.p1 | 68 / 97 / 5 / 22 | 4.5 | +4 [-2, +10] | +3 [+0, +7] | -2 [-5, +0] | 162, +2 [-3, +8], 5 / 3 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | +2 (147) |
| x.rand.p1 | 65 / 99 / 5 / 23 | 4.1 | +1 [-2, +5] | +2 [+0, +5] | -1 [-3, +0] | 162, +0 [-3, +3], 1 / 1 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +1 (158) |
| x.rand.m1 | 65 / 95 / 5 / 27 | 4.2 | +1 [-5, +7] | -2 [-6, +2] | +3 [+0, +7] | 159, +0 [-6, +6], 4 / 4 | 3 / 1 / 0 | 0 / 3 / 1 / 0 | +2 (153) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 82 / 81 / 2 / 27; dR +2 [-10, +14]; answered-rate change +1.6 pp [-1.6, +5.2]; out / in 4 / 7; mean words 9.3 -> 12.7
- main effect: dR +8 (4.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +7 [-3, +17]; vs x.rand.m1 +7 [-2, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-3.6, +3.6]; moved out / in 6 / 6 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 103 questions; mean words 4.0 -> 6.7
KEY family=E judge=pro layer=11 A=met B=not_met dR=+8 dR_F=+2 dAns=+0 dAns_F=+3 rand_margin=+7


# seed 2045
# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 93 / 70 / 2 / 27 | 8.0 | +15 [+4, +26] | -3 [-10, +3] | +4 [-1, +10] | 159, +14 [+4, +24], 21 / 7 | 7 / 4 / 1 | 2 / 5 / 3 / 1 | +5 (93) |
| x.w.m1 | 71 / 104 / 2 / 15 | 3.1 | -7 [-19, +5] | +9 [+2, +16] | -8 [-14, -2] | 164, -11 [-21, -1], 9 / 20 | 2 / 11 / 1 | 2 / 0 / 6 / 5 | -12 (161) |
| x.md.p1 | 88 / 79 / 2 / 23 | 6.1 | +10 [-1, +21] | +1 [-5, +7] | +0 [-5, +5] | 162, +9 [-1, +19], 17 / 8 | 4 / 5 / 1 | 3 / 1 / 4 / 1 | +5 (115) |
| x.rand.p1 | 78 / 88 / 3 / 23 | 4.2 | +0 [-6, +6] | +0 [+0, +0] | +0 [+0, +0] | 166, +0 [-6, +6], 5 / 5 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (158) |
| x.rand.m1 | 79 / 86 / 2 / 25 | 4.4 | +1 [-7, +9] | -1 [-5, +3] | +2 [-2, +6] | 163, +1 [-6, +8], 7 / 6 | 3 / 2 / 0 | 2 / 1 / 2 / 0 | -3 (151) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 104 / 57 / 2 / 29; dR +2 [-9, +14]; answered-rate change +0.0 pp [-2.6, +3.1]; out / in 4 / 4; mean words 9.3 -> 11.2
- main effect: dR +15 (7.8% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +15 [+5, +25]; vs x.rand.m1 +14 [+3, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.6 pp [-5.2, +1.6]; moved out / in 7 / 4 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 93 questions; mean words 4.0 -> 8.0
KEY family=E judge=allenai layer=13 A=met B=not_met dR=+15 dR_F=+2 dAns=-3 dAns_F=+0 rand_margin=+14


# seed 2045
# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 69 / 88 / 7 / 28 | 8.0 | +5 [-6, +15] | -5 [-12, +1] | +4 [-1, +10] | 154, +6 [-4, +16], 16 / 10 | 8 / 3 / 2 | 2 / 6 / 1 / 2 | +6 (90) |
| x.w.m1 | 61 / 113 / 3 / 15 | 3.1 | -3 [-13, +7] | +12 [+6, +19] | -9 [-15, -4] | 162, -6 [-15, +3], 8 / 14 | 0 / 12 / 0 | 0 / 0 / 3 / 9 | -6 (159) |
| x.md.p1 | 62 / 100 / 8 / 22 | 6.1 | -2 [-12, +8] | +0 [-5, +5] | -2 [-6, +2] | 159, -2 [-11, +7], 9 / 11 | 3 / 3 / 2 | 2 / 1 / 2 / 1 | +0 (113) |
| x.rand.p1 | 64 / 98 / 6 / 24 | 4.2 | +0 [-6, +5] | +0 [+0, +0] | +0 [+0, +0] | 162, +0 [-6, +5], 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +3 (153) |
| x.rand.m1 | 65 / 97 / 4 / 26 | 4.4 | +1 [-5, +7] | +0 [-5, +5] | +2 [-2, +6] | 159, -1 [-6, +4], 3 / 4 | 3 / 3 / 0 | 0 / 3 / 2 / 1 | -1 (148) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 84 / 76 / 3 / 29; dR +4 [-7, +15]; answered-rate change +0.0 pp [-3.1, +3.1]; out / in 4 / 4; mean words 9.3 -> 11.2
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-5, +15]; vs x.rand.m1 +4 [-6, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.6 pp [-6.2, +0.5]; moved out / in 8 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 90 questions; mean words 4.0 -> 8.0
KEY family=E judge=pro layer=13 A=not_met B=not_met dR=+5 dR_F=+4 dAns=-5 dAns_F=+0 rand_margin=+4



# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 95 / 71 / 2 / 24 | 6.5 | +17 [+6, +28] | +0 [-7, +7] | +1 [-5, +7] | 160, +13 [+3, +23], 20 / 7 | 6 / 6 / 1 | 2 / 4 / 6 / 0 | +8 (107) |
| x.w.m1 | 62 / 106 / 3 / 21 | 3.2 | -16 [-27, -6] | +2 [-3, +8] | -2 [-7, +3] | 163, -15 [-25, -6], 5 / 20 | 3 / 5 / 1 | 3 / 0 / 2 / 3 | -15 (162) |
| x.md.p1 | 86 / 83 / 2 / 21 | 4.5 | +8 [+2, +15] | +3 [-1, +8] | -2 [-5, +0] | 165, +6 [+1, +12], 7 / 1 | 1 / 4 / 1 | 1 / 0 / 3 / 1 | +3 (149) |
| x.rand.p1 | 79 / 89 / 2 / 22 | 4.0 | +1 [-3, +5] | +2 [+0, +5] | -1 [-3, +0] | 166, +0 [-4, +4], 2 / 2 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +0 (161) |
| x.rand.m1 | 82 / 81 / 2 / 27 | 4.2 | +4 [-2, +10] | -3 [-8, +1] | +4 [+1, +8] | 162, +5 [+0, +10], 6 / 1 | 4 / 1 / 0 | 2 / 2 / 1 / 0 | +4 (156) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 115 / 49 / 1 / 27; dR +13 [+2, +24]; answered-rate change +1.6 pp [-2.1, +5.2]; out / in 5 / 8; mean words 9.3 -> 12.3
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +16 [+5, +27]; vs x.rand.m1 +13 [+3, +23] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +0.0 pp [-3.6, +3.6]; moved out / in 6 / 6 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +8 on 107 questions; mean words 4.0 -> 6.5
KEY family=E judge=allenai layer=11 A=met B=not_met dR=+17 dR_F=+13 dAns=+0 dAns_F=+3 rand_margin=+13


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 71 / 90 / 5 / 26 | 6.5 | +7 [-3, +17] | -1 [-8, +5] | +2 [-3, +8] | 156, +4 [-5, +14], 14 / 10 | 6 / 5 / 1 | 1 / 5 / 4 / 1 | +6 (104) |
| x.w.m1 | 54 / 113 / 4 / 21 | 3.2 | -10 [-19, -1] | +5 [-1, +11] | -3 [-8, +1] | 160, -10 [-18, -2], 4 / 14 | 2 / 7 / 1 | 2 / 0 / 2 / 5 | -10 (159) |
| x.md.p1 | 68 / 97 / 5 / 22 | 4.5 | +4 [-2, +10] | +3 [+0, +7] | -2 [-5, +0] | 162, +2 [-3, +8], 5 / 3 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | +2 (147) |
| x.rand.p1 | 66 / 99 / 4 / 23 | 4.0 | +2 [-2, +6] | +3 [+0, +7] | -1 [-3, +0] | 162, +0 [-3, +3], 1 / 1 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | +1 (158) |
| x.rand.m1 | 64 / 95 / 5 / 28 | 4.2 | +0 [-5, +6] | -3 [-8, +1] | +4 [+1, +8] | 158, -1 [-6, +4], 3 / 4 | 4 / 1 / 0 | 0 / 4 / 1 / 0 | +1 (152) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 82 / 80 / 2 / 28; dR +2 [-10, +14]; answered-rate change +1.0 pp [-2.6, +4.7]; out / in 5 / 7; mean words 9.3 -> 12.3
- main effect: dR +7 (3.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-5, +15]; vs x.rand.m1 +7 [-3, +17] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -0.5 pp [-4.2, +2.6]; moved out / in 6 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 104 questions; mean words 4.0 -> 6.5
KEY family=E judge=pro layer=11 A=met B=not_met dR=+7 dR_F=+2 dAns=-1 dAns_F=+2 rand_margin=+5


# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 94 / 70 / 2 / 26 | 6.8 | +16 [+6, +26] | -2 [-9, +5] | +3 [-3, +9] | 159, +14 [+5, +23], 19 / 5 | 7 / 5 / 1 | 2 / 5 / 4 / 1 | +7 (107) |
| x.w.m1 | 70 / 103 / 2 / 17 | 3.2 | -8 [-18, +2] | +7 [+1, +14] | -6 [-12, -1] | 164, -11 [-20, -2], 5 / 16 | 2 / 9 / 1 | 2 / 0 / 5 / 4 | -11 (163) |
| x.md.p1 | 88 / 80 / 2 / 22 | 4.2 | +10 [+3, +17] | +2 [-2, +6] | -1 [-3, +0] | 165, +9 [+3, +16], 10 / 1 | 1 / 3 / 1 | 1 / 0 / 2 / 1 | +6 (153) |
| x.rand.p1 | 77 / 86 / 2 / 27 | 3.9 | -1 [-7, +5] | -3 [-8, +1] | +4 [+1, +8] | 162, +0 [-5, +5], 3 / 3 | 4 / 1 / 0 | 2 / 2 / 1 / 0 | +0 (156) |
| x.rand.m1 | 80 / 89 / 2 / 21 | 4.2 | +2 [-3, +8] | +3 [+0, +7] | -2 [-5, +0] | 166, +1 [-4, +6], 4 / 3 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +0 (160) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 115 / 45 / 1 / 31; dR +13 [+1, +25]; answered-rate change -0.5 pp [-4.2, +2.6]; out / in 6 / 5; mean words 9.3 -> 11.9
- main effect: dR +16 (8.3% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +17 [+7, +27]; vs x.rand.m1 +14 [+4, +24] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.0 pp [-4.7, +2.6]; moved out / in 7 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +7 on 107 questions; mean words 4.0 -> 6.8
KEY family=E judge=allenai layer=12 A=met B=not_met dR=+16 dR_F=+13 dAns=-2 dAns_F=-1 rand_margin=+14


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 70 / 86 / 8 / 28 | 6.8 | +6 [-4, +16] | -6 [-13, +0] | +4 [-1, +10] | 153, +7 [-2, +16], 15 / 8 | 9 / 3 / 3 | 2 / 7 / 1 / 2 | +6 (104) |
| x.w.m1 | 59 / 110 / 6 / 17 | 3.2 | -5 [-14, +4] | +7 [+2, +13] | -7 [-12, -2] | 161, -5 [-14, +3], 7 / 12 | 1 / 8 / 1 | 1 / 0 / 1 / 7 | -5 (160) |
| x.md.p1 | 69 / 96 / 4 / 23 | 4.2 | +5 [-1, +11] | +3 [+0, +7] | -1 [-3, +0] | 162, +3 [-2, +8], 5 / 2 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | +3 (150) |
| x.rand.p1 | 65 / 94 / 5 / 28 | 3.9 | +1 [-5, +7] | -3 [-8, +2] | +4 [+1, +8] | 157, +0 [-5, +5], 3 / 3 | 5 / 2 / 1 | 1 / 4 / 2 / 0 | +1 (153) |
| x.rand.m1 | 64 / 100 / 5 / 23 | 4.2 | +0 [-4, +4] | +2 [+0, +5] | -1 [-3, +0] | 162, -1 [-5, +2], 1 / 2 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +1 (156) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 92 / 68 / 1 / 31; dR +12 [+0, +24]; answered-rate change +0.0 pp [-3.6, +3.6]; out / in 6 / 6; mean words 9.3 -> 11.9
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +5 [-5, +15]; vs x.rand.m1 +6 [-3, +15] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.1 pp [-6.8, +0.0]; moved out / in 9 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +6 on 104 questions; mean words 4.0 -> 6.8
KEY family=E judge=pro layer=12 A=not_met B=not_met dR=+6 dR_F=+12 dAns=-6 dAns_F=+0 rand_margin=+5


# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 95 / 69 / 2 / 26 | 7.9 | +17 [+6, +28] | -2 [-9, +5] | +3 [-3, +9] | 159, +15 [+5, +25], 22 / 7 | 7 / 5 / 1 | 2 / 5 / 4 / 1 | +5 (94) |
| x.w.m1 | 69 / 107 / 2 / 14 | 3.1 | -9 [-21, +3] | +10 [+3, +17] | -9 [-16, -3] | 164, -13 [-23, -3], 8 / 21 | 2 / 12 / 1 | 2 / 0 / 6 / 6 | -14 (161) |
| x.md.p1 | 88 / 79 / 2 / 23 | 6.1 | +10 [-1, +21] | +1 [-5, +7] | +0 [-5, +5] | 162, +9 [-1, +19], 17 / 8 | 4 / 5 / 1 | 3 / 1 / 4 / 1 | +5 (115) |
| x.rand.p1 | 78 / 87 / 3 / 24 | 4.2 | +0 [-6, +6] | -1 [-3, +0] | +1 [+0, +3] | 165, +0 [-6, +6], 5 / 5 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -1 (157) |
| x.rand.m1 | 81 / 84 / 2 / 25 | 4.4 | +3 [-4, +10] | -1 [-6, +3] | +2 [-2, +6] | 163, +2 [-4, +8], 6 / 4 | 3 / 2 / 0 | 1 / 2 / 2 / 0 | -1 (152) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 104 / 56 / 2 / 30; dR +2 [-10, +14]; answered-rate change -0.5 pp [-3.6, +2.6]; out / in 5 / 4; mean words 9.3 -> 11.3
- main effect: dR +17 (8.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +17 [+7, +28]; vs x.rand.m1 +14 [+3, +25] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -1.0 pp [-4.7, +2.6]; moved out / in 7 / 5 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +5 on 94 questions; mean words 4.0 -> 7.9
KEY family=E judge=allenai layer=13 A=met B=not_met dR=+17 dR_F=+2 dAns=-2 dAns_F=-1 rand_margin=+14


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 70 / 87 / 7 / 28 | 7.9 | +6 [-5, +16] | -5 [-12, +1] | +4 [-1, +10] | 154, +7 [-3, +17], 17 / 10 | 8 / 3 / 2 | 2 / 6 / 1 / 2 | +6 (91) |
| x.w.m1 | 62 / 113 / 3 / 14 | 3.1 | -2 [-12, +8] | +13 [+7, +20] | -10 [-16, -5] | 162, -6 [-15, +3], 8 / 14 | 0 / 13 / 0 | 0 / 0 / 4 / 9 | -6 (159) |
| x.md.p1 | 62 / 100 / 8 / 22 | 6.1 | -2 [-12, +8] | +0 [-5, +5] | -2 [-6, +2] | 159, -2 [-11, +7], 9 / 11 | 3 / 3 / 2 | 2 / 1 / 2 / 1 | +0 (113) |
| x.rand.p1 | 64 / 97 / 6 / 25 | 4.2 | +0 [-6, +5] | -1 [-3, +0] | +1 [+0, +3] | 161, +0 [-6, +5], 4 / 4 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +3 (152) |
| x.rand.m1 | 64 / 96 / 6 / 26 | 4.4 | +0 [-6, +6] | -2 [-7, +3] | +2 [-2, +6] | 158, -2 [-7, +3], 2 / 4 | 4 / 2 / 1 | 0 / 4 / 2 / 0 | -1 (149) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 83 / 76 / 3 / 30; dR +3 [-8, +13]; answered-rate change -0.5 pp [-3.6, +2.6]; out / in 5 / 4; mean words 9.3 -> 11.3
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +6 [-4, +16]; vs x.rand.m1 +6 [-5, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.6 pp [-6.2, +0.5]; moved out / in 8 / 3 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +6 on 91 questions; mean words 4.0 -> 7.9
KEY family=E judge=pro layer=13 A=met B=not_met dR=+6 dR_F=+3 dAns=-5 dAns_F=-1 rand_margin=+6


# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 89 / 72 / 1 / 30 | 4.7 | +11 [+3, +19] | -5 [-11, +1] | +7 [+2, +12] | 159, +11 [+4, +19], 13 / 2 | 7 / 2 / 0 | 2 / 5 / 2 / 0 | +9 (131) |
| x.w.m1 | 72 / 98 / 4 / 18 | 3.8 | -6 [-16, +4] | +4 [-1, +10] | -5 [-10, -1] | 164, -6 [-15, +3], 8 / 14 | 2 / 6 / 2 | 2 / 0 / 2 / 4 | -8 (157) |
| x.md.p1 | 82 / 77 / 2 / 31 | 4.9 | +4 [-5, +13] | -7 [-13, -2] | +8 [+3, +14] | 158, +6 [-2, +15], 13 / 7 | 8 / 1 / 0 | 3 / 5 / 1 / 0 | +4 (136) |
| x.rand.p1 | 79 / 87 / 3 / 23 | 4.2 | +1 [-5, +7] | +0 [+0, +0] | +0 [+0, +0] | 166, +1 [-5, +7], 6 / 5 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (158) |
| x.rand.m1 | 79 / 86 / 1 / 26 | 4.0 | +1 [-5, +7] | -1 [-6, +3] | +3 [+0, +7] | 163, +1 [-5, +7], 5 / 4 | 3 / 2 / 0 | 1 / 2 / 1 / 1 | +1 (159) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 99 / 52 / 2 / 39; dR -3 [-14, +8]; answered-rate change -5.2 pp [-8.9, -1.6]; out / in 12 / 2; mean words 9.3 -> 10.0
- main effect: dR +11 (5.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +10 [+1, +19]; vs x.rand.m1 +10 [+3, +18] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -2.6 pp [-5.7, +0.5]; moved out / in 7 / 2 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same +9 on 131 questions; mean words 4.0 -> 4.7
KEY family=E judge=allenai layer=17 A=not_met B=not_met dR=+11 dR_F=-3 dAns=-5 dAns_F=-10 rand_margin=+10


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 68 / 86 / 7 / 31 | 4.7 | +4 [-5, +13] | -8 [-15, -2] | +7 [+2, +12] | 152, +5 [-2, +13], 10 / 5 | 10 / 2 / 3 | 3 / 7 / 2 / 0 | +7 (127) |
| x.w.m1 | 59 / 106 / 8 / 19 | 3.8 | -5 [-14, +3] | +3 [-1, +8] | -5 [-10, -1] | 161, -5 [-13, +3], 6 / 11 | 1 / 4 / 1 | 1 / 0 / 1 / 3 | -5 (154) |
| x.md.p1 | 64 / 91 / 6 / 31 | 4.9 | +0 [-10, +10] | -7 [-14, +0] | +7 [+2, +12] | 152, +1 [-7, +10], 10 / 9 | 10 / 3 / 3 | 3 / 7 / 2 / 1 | +5 (130) |
| x.rand.p1 | 64 / 98 / 6 / 24 | 4.2 | +0 [-6, +6] | +0 [-3, +3] | +0 [+0, +0] | 161, +0 [-5, +5], 3 / 3 | 1 / 1 / 1 | 1 / 0 / 1 / 0 | +2 (153) |
| x.rand.m1 | 65 / 94 / 6 / 27 | 4.0 | +1 [-4, +6] | -3 [-7, +0] | +3 [+0, +7] | 159, +1 [-4, +6], 4 / 3 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | +1 (155) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 78 / 73 / 3 / 38; dR -2 [-12, +8]; answered-rate change -4.7 pp [-8.3, -1.6]; out / in 10 / 1; mean words 9.3 -> 10.0
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +4 [-3, +12]; vs x.rand.m1 +3 [-5, +11] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -4.2 pp [-7.8, -1.0]; moved out / in 10 / 2 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +7 on 127 questions; mean words 4.0 -> 4.7
KEY family=E judge=pro layer=17 A=not_met B=not_met dR=+4 dR_F=-2 dAns=-8 dAns_F=-9 rand_margin=+3


# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 96 / 63 / 2 / 31 | 5.9 | +18 [+7, +29] | -7 [-15, +0] | +8 [+1, +15] | 155, +17 [+8, +26], 21 / 4 | 11 / 4 / 0 | 3 / 8 / 4 / 0 | +12 (114) |
| x.w.m1 | 67 / 105 / 3 / 17 | 3.2 | -11 [-21, -2] | +6 [+0, +12] | -6 [-12, -1] | 164, -12 [-21, -4], 4 / 16 | 2 / 8 / 1 | 2 / 0 / 3 / 5 | -12 (162) |
| x.md.p1 | 84 / 82 / 1 / 25 | 4.3 | +6 [-1, +13] | +0 [-5, +5] | +2 [-2, +6] | 163, +6 [+0, +12], 8 / 2 | 3 / 3 / 0 | 2 / 1 / 2 / 1 | +4 (157) |
| x.rand.p1 | 79 / 90 / 1 / 22 | 4.3 | +1 [-4, +6] | +3 [+0, +7] | -1 [-3, +0] | 166, -1 [-5, +3], 2 / 3 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | -1 (158) |
| x.rand.m1 | 77 / 85 / 2 / 28 | 3.8 | -1 [-8, +6] | -4 [-10, +1] | +5 [+1, +10] | 160, +2 [-3, +8], 5 / 3 | 6 / 2 / 1 | 4 / 2 / 1 / 1 | +2 (155) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 112 / 47 / 1 / 32; dR +10 [-1, +21]; answered-rate change -1.0 pp [-5.2, +3.1]; out / in 9 / 7; mean words 9.3 -> 11.7
- main effect: dR +18 (9.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +17 [+6, +28]; vs x.rand.m1 +19 [+8, +30] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.6 pp [-7.8, +0.0]; moved out / in 11 / 4 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +12 on 114 questions; mean words 4.0 -> 5.9
KEY family=E judge=allenai layer=10 A=not_met B=not_met dR=+18 dR_F=+10 dAns=-7 dAns_F=-2 rand_margin=+17


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 69 / 85 / 5 / 33 | 5.9 | +5 [-5, +15] | -8 [-16, +0] | +9 [+2, +16] | 149, +4 [-5, +13], 12 / 8 | 13 / 5 / 2 | 3 / 10 / 4 / 1 | +6 (109) |
| x.w.m1 | 55 / 114 / 6 / 17 | 3.2 | -9 [-17, -1] | +7 [+2, +13] | -7 [-13, -2] | 161, -8 [-16, -1], 4 / 12 | 1 / 8 / 1 | 1 / 0 / 0 / 8 | -8 (159) |
| x.md.p1 | 69 / 93 / 5 / 25 | 4.3 | +5 [+0, +11] | +0 [-4, +4] | +1 [-2, +4] | 160, +4 [-1, +10], 6 / 2 | 2 / 2 / 0 | 0 / 2 / 1 / 1 | +5 (153) |
| x.rand.p1 | 62 / 101 / 6 / 23 | 4.3 | -2 [-7, +3] | +1 [-2, +4] | -1 [-3, +0] | 161, -2 [-6, +2], 1 / 3 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +0 (154) |
| x.rand.m1 | 65 / 93 / 6 / 28 | 3.8 | +1 [-4, +6] | -4 [-8, -1] | +4 [+1, +8] | 158, +1 [-4, +6], 4 / 3 | 4 / 0 / 0 | 0 / 4 / 0 / 0 | +2 (153) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 84 / 73 / 2 / 33; dR +4 [-8, +16]; answered-rate change -1.6 pp [-5.7, +2.6]; out / in 10 / 7; mean words 9.3 -> 11.7
- main effect: dR +5 (2.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +7 [-2, +16]; vs x.rand.m1 +4 [-6, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: True | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -4.2 pp [-8.3, +0.0]; moved out / in 13 / 5 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +6 on 109 questions; mean words 4.0 -> 5.9
KEY family=E judge=pro layer=10 A=not_met B=not_met dR=+5 dR_F=+4 dAns=-8 dAns_F=-3 rand_margin=+4



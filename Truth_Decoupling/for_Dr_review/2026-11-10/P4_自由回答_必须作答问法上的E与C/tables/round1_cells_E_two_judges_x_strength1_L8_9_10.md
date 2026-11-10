# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 89 / 71 / 2 / 30 | 4.8 | +11 [+3, +19] | -6 [-12, -1] | +7 [+2, +13] | 159, +10 [+2, +18], 14 / 4 | 7 / 1 / 0 | 0 / 7 / 1 / 0 | +7 (136) |
| x.w.m1 | 77 / 96 / 1 / 18 | 3.7 | -1 [-10, +8] | +7 [+1, +14] | -5 [-11, +1] | 164, -4 [-12, +3], 6 / 10 | 2 / 9 / 0 | 2 / 0 / 5 / 4 | -4 (158) |
| x.md.p1 | 84 / 82 / 3 / 23 | 4.2 | +6 [+1, +12] | +0 [+0, +0] | +0 [+0, +0] | 166, +6 [+1, +12], 7 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +4 (163) |
| x.rand.p1 | 78 / 89 / 2 / 23 | 4.1 | +0 [-5, +5] | +1 [+0, +3] | +0 [+0, +0] | 166, +0 [-5, +5], 3 / 3 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +1 (162) |
| x.rand.m1 | 83 / 83 / 1 / 25 | 3.8 | +5 [+1, +10] | +0 [-4, +4] | +2 [+0, +5] | 164, +3 [+0, +7], 3 / 0 | 2 / 2 / 0 | 0 / 2 / 2 / 0 | +3 (162) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 106 / 48 / 0 / 38; dR +4 [-6, +14]; answered-rate change -3.6 pp [-6.8, -1.0]; out / in 8 / 1; mean words 9.3 -> 11.2
- main effect: dR +11 (5.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +11 [+2, +20]; vs x.rand.m1 +6 [-2, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.1 pp [-6.2, -0.5]; moved out / in 7 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +7 on 136 questions; mean words 4.0 -> 4.8
KEY family=E judge=allenai layer=8 A=not_met B=not_met dR=+11 dR_F=+4 dAns=-6 dAns_F=-7 rand_margin=+6

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 8

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 67 / 89 / 6 / 30 | 4.8 | +3 [-5, +12] | -6 [-13, +1] | +6 [+1, +12] | 153, +3 [-4, +10], 8 / 5 | 9 / 3 / 2 | 3 / 6 / 3 / 0 | +5 (133) |
| x.w.m1 | 61 / 105 / 8 / 18 | 3.7 | -3 [-10, +4] | +4 [-1, +10] | -6 [-12, -1] | 160, -4 [-10, +2], 3 / 7 | 2 / 6 / 1 | 1 / 1 / 2 / 4 | -4 (156) |
| x.md.p1 | 66 / 97 / 5 / 24 | 4.2 | +2 [-3, +7] | +1 [+0, +3] | +0 [+0, +0] | 162, +1 [-3, +6], 3 / 2 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (158) |
| x.rand.p1 | 66 / 97 / 5 / 24 | 4.1 | +2 [-2, +6] | +1 [+0, +3] | +0 [+0, +0] | 162, +1 [-2, +5], 2 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (157) |
| x.rand.m1 | 66 / 95 / 5 / 26 | 3.8 | +2 [-3, +7] | -1 [-5, +2] | +2 [+0, +5] | 160, +1 [-3, +5], 3 / 2 | 2 / 1 / 0 | 0 / 2 / 1 / 0 | +2 (158) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 77 / 75 / 3 / 37; dR -3 [-13, +7]; answered-rate change -4.2 pp [-8.3, -0.5]; out / in 11 / 3; mean words 9.3 -> 11.2
- main effect: dR +3 (1.6% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +1 [-7, +10]; vs x.rand.m1 +1 [-8, +10] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -3.1 pp [-6.8, +0.5]; moved out / in 9 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 133 questions; mean words 4.0 -> 4.8
KEY family=E judge=pro layer=8 A=not_met B=not_met dR=+3 dR_F=-3 dAns=-6 dAns_F=-8 rand_margin=+1

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 89 / 69 / 3 / 31 | 5.0 | +11 [+2, +20] | -8 [-15, -2] | +8 [+3, +14] | 157, +11 [+3, +19], 15 / 4 | 9 / 1 / 1 | 1 / 8 / 1 / 0 | +8 (134) |
| x.w.m1 | 76 / 97 / 1 / 18 | 3.7 | -2 [-12, +7] | +7 [+1, +14] | -5 [-11, +1] | 164, -5 [-13, +3], 6 / 11 | 2 / 9 / 0 | 2 / 0 / 5 / 4 | -4 (157) |
| x.md.p1 | 82 / 83 / 1 / 26 | 4.0 | +4 [-2, +10] | -1 [-5, +3] | +3 [+0, +7] | 163, +3 [-2, +8], 5 / 2 | 3 / 2 / 0 | 1 / 2 / 2 / 0 | +2 (156) |
| x.rand.p1 | 81 / 85 / 2 / 24 | 4.1 | +3 [-2, +8] | +0 [-3, +3] | +1 [+0, +3] | 165, +3 [-2, +8], 5 / 2 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | +2 (162) |
| x.rand.m1 | 81 / 86 / 2 / 23 | 4.2 | +3 [-1, +8] | +1 [+0, +3] | +0 [+0, +0] | 166, +2 [-2, +6], 3 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (156) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 105 / 48 / 0 / 39; dR +3 [-7, +13]; answered-rate change -4.2 pp [-7.8, -1.0]; out / in 10 / 2; mean words 9.3 -> 11.2
- main effect: dR +11 (5.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +8 [-1, +17]; vs x.rand.m1 +8 [+0, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -4.2 pp [-7.8, -1.0]; moved out / in 9 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 134 questions; mean words 4.0 -> 5.0
KEY family=E judge=allenai layer=9 A=not_met B=not_met dR=+11 dR_F=+3 dAns=-8 dAns_F=-8 rand_margin=+8

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 68 / 86 / 7 / 31 | 5.0 | +4 [-5, +13] | -8 [-16, -1] | +7 [+2, +13] | 151, +5 [-2, +13], 10 / 5 | 11 / 3 / 3 | 4 / 7 / 3 / 0 | +5 (131) |
| x.w.m1 | 60 / 107 / 7 / 18 | 3.7 | -4 [-12, +4] | +5 [-1, +11] | -6 [-12, -1] | 160, -5 [-12, +2], 4 / 9 | 2 / 7 / 1 | 1 / 1 / 2 / 5 | -5 (155) |
| x.md.p1 | 64 / 96 / 5 / 27 | 4.0 | +0 [-7, +7] | -2 [-6, +2] | +3 [+0, +7] | 159, +0 [-6, +6], 5 / 5 | 3 / 1 / 0 | 1 / 2 / 1 / 0 | +1 (153) |
| x.rand.p1 | 65 / 96 / 6 / 25 | 4.1 | +1 [-3, +5] | -1 [-3, +0] | +1 [+0, +3] | 161, +1 [-3, +5], 3 / 2 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | +2 (158) |
| x.rand.m1 | 63 / 100 / 5 / 24 | 4.2 | -1 [-5, +3] | +1 [+0, +3] | +0 [+0, +0] | 162, -2 [-6, +2], 1 / 3 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (152) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 84 / 67 / 3 / 38; dR +4 [-7, +15]; answered-rate change -4.7 pp [-8.9, -1.0]; out / in 12 / 3; mean words 9.3 -> 11.2
- main effect: dR +4 (2.1% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 +3 [-5, +11]; vs x.rand.m1 +5 [-3, +14] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -4.2 pp [-8.3, -0.5]; moved out / in 11 / 3 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 131 questions; mean words 4.0 -> 5.0
KEY family=E judge=pro layer=9 A=not_met B=not_met dR=+4 dR_F=+4 dAns=-8 dAns_F=-9 rand_margin=+3

# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 88 / 68 / 3 / 33 | 5.0 | +10 [+0, +20] | -10 [-17, -4] | +10 [+5, +17] | 155, +12 [+3, +21], 17 / 5 | 11 / 1 / 1 | 3 / 8 / 1 / 0 | +9 (125) |
| x.w.m1 | 77 / 96 / 1 / 18 | 3.6 | -1 [-11, +9] | +7 [+0, +14] | -5 [-11, +1] | 163, -4 [-12, +3], 6 / 10 | 3 / 10 / 1 | 3 / 0 / 6 / 4 | -4 (158) |
| x.md.p1 | 82 / 84 / 1 / 25 | 3.8 | +4 [-2, +10] | +0 [-4, +4] | +2 [+0, +5] | 164, +4 [-1, +9], 6 / 2 | 2 / 2 / 0 | 1 / 1 / 1 / 1 | +3 (157) |
| x.rand.p1 | 79 / 90 / 1 / 22 | 4.2 | +1 [-5, +7] | +3 [+0, +7] | -1 [-3, +0] | 166, -1 [-6, +4], 3 / 4 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | -1 (159) |
| x.rand.m1 | 78 / 84 / 2 / 28 | 3.8 | +0 [-7, +7] | -4 [-10, +1] | +5 [+1, +10] | 160, +3 [-2, +8], 5 / 2 | 6 / 2 / 1 | 4 / 2 / 1 / 1 | +3 (155) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 104 / 50 / 0 / 38; dR +2 [-9, +13]; answered-rate change -3.6 pp [-7.3, +0.0]; out / in 10 / 3; mean words 9.3 -> 11.5
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +9 [-1, +19]; vs x.rand.m1 +10 [+0, +20] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -5.2 pp [-8.9, -2.1]; moved out / in 11 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +9 on 125 questions; mean words 4.0 -> 5.0
KEY family=E judge=allenai layer=10 A=not_met B=not_met dR=+10 dR_F=+2 dAns=-10 dAns_F=-7 rand_margin=+9

# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 70 / 84 / 6 / 32 | 5.0 | +6 [-4, +16] | -8 [-16, +0] | +8 [+2, +14] | 150, +7 [-1, +15], 12 / 5 | 12 / 4 / 3 | 5 / 7 / 4 / 0 | +7 (124) |
| x.w.m1 | 63 / 104 / 7 / 18 | 3.6 | -1 [-8, +7] | +5 [-1, +12] | -6 [-12, -1] | 159, -2 [-8, +4], 4 / 6 | 3 / 8 / 2 | 2 / 1 / 3 / 5 | -2 (155) |
| x.md.p1 | 64 / 98 / 5 / 25 | 3.8 | +0 [-5, +5] | +0 [-3, +3] | +1 [+0, +3] | 161, -1 [-6, +3], 2 / 3 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | +0 (155) |
| x.rand.p1 | 64 / 100 / 5 / 23 | 4.2 | +0 [-4, +4] | +2 [+0, +5] | -1 [-3, +0] | 162, -1 [-5, +2], 1 / 2 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +1 (156) |
| x.rand.m1 | 64 / 93 / 7 / 28 | 3.8 | +0 [-6, +5] | -5 [-10, -1] | +4 [+1, +8] | 157, +1 [-4, +6], 4 / 3 | 5 / 0 / 1 | 1 / 4 / 0 / 0 | +2 (152) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 87 / 65 / 2 / 38; dR +7 [-4, +18]; answered-rate change -4.2 pp [-8.3, -0.5]; out / in 12 / 4; mean words 9.3 -> 11.5
- main effect: dR +6 (3.1% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +6 [-3, +15]; vs x.rand.m1 +6 [-4, +16] | A: above each (point): True | B: each interval above 0: False
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change -4.2 pp [-8.3, +0.0]; moved out / in 12 / 4 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +7 on 124 questions; mean words 4.0 -> 5.0
KEY family=E judge=pro layer=10 A=not_met B=not_met dR=+6 dR_F=+7 dAns=-8 dAns_F=-8 rand_margin=+6


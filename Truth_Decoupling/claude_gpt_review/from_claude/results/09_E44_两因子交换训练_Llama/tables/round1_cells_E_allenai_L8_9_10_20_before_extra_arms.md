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

- F, complete-sentence background: arm x_F.w.p1 not judged
- main effect: dR +11 (5.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +11 [+2, +20]; MISSING x.rand.m1 | A: above each (point): None | B: each interval above 0: None
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: None | B: dR interval above 0 and answered-rate interval inside: None
- retention on x: answered-rate change -3.1 pp [-6.2, -0.5]; moved out / in 7 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +7 on 136 questions; mean words 4.0 -> 4.8
KEY family=E judge=allenai layer=8 A=not_met B=not_met dR=+11 dR_F=NA dAns=-6 dAns_F=NA rand_margin=NA

## layer 9

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 89 / 69 / 3 / 31 | 5.0 | +11 [+2, +20] | -8 [-15, -2] | +8 [+3, +14] | 157, +11 [+3, +19], 15 / 4 | 9 / 1 / 1 | 1 / 8 / 1 / 0 | +8 (134) |
| x.w.m1 | 76 / 97 / 1 / 18 | 3.7 | -2 [-12, +7] | +7 [+1, +14] | -5 [-11, +1] | 164, -5 [-13, +3], 6 / 11 | 2 / 9 / 0 | 2 / 0 / 5 / 4 | -4 (157) |
| x.md.p1 | 82 / 83 / 1 / 26 | 4.0 | +4 [-2, +10] | -1 [-5, +3] | +3 [+0, +7] | 163, +3 [-2, +8], 5 / 2 | 3 / 2 / 0 | 1 / 2 / 2 / 0 | +2 (156) |
| x.rand.p1 | 81 / 85 / 2 / 24 | 4.1 | +3 [-2, +8] | +0 [-3, +3] | +1 [+0, +3] | 165, +3 [-2, +8], 5 / 2 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | +2 (162) |

- F, complete-sentence background: arm x_F.w.p1 not judged
- main effect: dR +11 (5.7% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p1 +8 [-1, +17]; MISSING x.rand.m1 | A: above each (point): None | B: each interval above 0: None
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: None | B: dR interval above 0 and answered-rate interval inside: None
- retention on x: answered-rate change -4.2 pp [-7.8, -1.0]; moved out / in 9 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 134 questions; mean words 4.0 -> 5.0
KEY family=E judge=allenai layer=9 A=not_met B=not_met dR=+11 dR_F=NA dAns=-8 dAns_F=NA rand_margin=NA

## layer 10

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 88 / 68 / 3 / 33 | 5.0 | +10 [+0, +20] | -10 [-17, -4] | +10 [+5, +17] | 155, +12 [+3, +21], 17 / 5 | 11 / 1 / 1 | 3 / 8 / 1 / 0 | +9 (125) |
| x.w.m1 | 77 / 96 / 1 / 18 | 3.6 | -1 [-11, +9] | +7 [+0, +14] | -5 [-11, +1] | 163, -4 [-12, +3], 6 / 10 | 3 / 10 / 1 | 3 / 0 / 6 / 4 | -4 (158) |
| x.md.p1 | 82 / 84 / 1 / 25 | 3.8 | +4 [-2, +10] | +0 [-4, +4] | +2 [+0, +5] | 164, +4 [-1, +9], 6 / 2 | 2 / 2 / 0 | 1 / 1 / 1 / 1 | +3 (157) |
| x.rand.p1 | 79 / 90 / 1 / 22 | 4.2 | +1 [-5, +7] | +3 [+0, +7] | -1 [-3, +0] | 166, -1 [-6, +4], 3 / 4 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | -1 (159) |

- F, complete-sentence background: arm x_F.w.p1 not judged
- main effect: dR +10 (5.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: False
- random controls: vs x.rand.p1 +9 [-1, +19]; MISSING x.rand.m1 | A: above each (point): None | B: each interval above 0: None
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: None | B: dR interval above 0 and answered-rate interval inside: None
- retention on x: answered-rate change -5.2 pp [-8.9, -2.1]; moved out / in 11 / 1 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +9 on 125 questions; mean words 4.0 -> 5.0
KEY family=E judge=allenai layer=10 A=not_met B=not_met dR=+10 dR_F=NA dAns=-10 dAns_F=NA rand_margin=NA

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 78 / 85 / 3 / 26 | 4.4 | +0 [-7, +7] | -3 [-7, +0] | +3 [+0, +7] | 163, +1 [-6, +8], 7 / 6 | 3 / 0 / 0 | 1 / 2 / 0 / 0 | -2 (148) |
| x.w.m1 | 85 / 83 / 1 / 23 | 3.8 | +7 [+1, +14] | +2 [+0, +5] | +0 [+0, +0] | 166, +5 [-1, +11], 7 / 2 | 0 / 2 / 0 | 0 / 0 / 2 / 0 | +4 (164) |
| x.md.p1 | 76 / 85 / 3 / 28 | 4.2 | -2 [-9, +5] | -5 [-10, -1] | +5 [+1, +10] | 161, +1 [-5, +8], 6 / 5 | 5 / 0 / 0 | 3 / 2 / 0 / 0 | -2 (153) |
| x.rand.p1 | 82 / 84 / 2 / 24 | 3.9 | +4 [-1, +10] | +0 [-3, +3] | +1 [+0, +3] | 165, +3 [-2, +8], 5 / 2 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | +2 (162) |

- F, complete-sentence background: arm x_F.w.p1 not judged
- main effect: dR +0 (0.0% of the questions) | A: beyond the tolerance: False | B: and interval above 0: False
- random controls: vs x.rand.p1 -4 [-12, +4]; MISSING x.rand.m1 | A: above each (point): None | B: each interval above 0: None
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: None | B: dR interval above 0 and answered-rate interval inside: None
- retention on x: answered-rate change -1.6 pp [-3.6, +0.0]; moved out / in 3 / 0 | A: point inside: True | B: whole interval inside: False
- diagnostic: G_same -2 on 148 questions; mean words 4.0 -> 4.4
KEY family=E judge=allenai layer=20 A=not_met B=not_met dR=+0 dR_F=NA dAns=-3 dAns_F=NA rand_margin=NA


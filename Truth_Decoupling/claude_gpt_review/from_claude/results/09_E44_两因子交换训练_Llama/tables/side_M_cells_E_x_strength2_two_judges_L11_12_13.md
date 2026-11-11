# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 120 / 54 / 3 / 15 | 13.3 | +42 [+29, +55] | +8 [-1, +17] | -8 [-17, +1] | 159, +28 [+16, +40], 34 / 6 | 7 / 15 / 1 | 0 / 7 / 14 / 1 | +8 (59) |
| x.rand.p2 | 79 / 90 / 2 / 21 | 4.0 | +1 [-7, +10] | +3 [-1, +8] | -2 [-6, +2] | 165, -1 [-8, +7], 7 / 8 | 1 / 4 / 0 | 1 / 0 / 3 / 1 | -2 (159) |
| x.rand.m2 | 86 / 77 / 1 / 28 | 4.3 | +8 [+1, +15] | -3 [-8, +2] | +5 [+1, +10] | 161, +8 [+2, +14], 9 / 1 | 5 / 2 / 0 | 2 / 3 / 2 / 0 | +6 (153) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 136 / 40 / 1 / 15; dR +34 [+20, +48]; answered-rate change +7.8 pp [+3.1, +12.5]; out / in 4 / 19; mean words 9.3 -> 17.9
- main effect: dR +42 (21.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +41 [+27, +55]; vs x.rand.m2 +34 [+21, +47] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +4.2 pp [-0.5, +8.9]; moved out / in 7 / 15 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 59 questions; mean words 4.0 -> 13.3
KEY family=E judge=allenai layer=11 A=not_met B=not_met dR=+42 dR_F=+34 dAns=+8 dAns_F=+15 rand_margin=+34 EXPLORATORY base=x main=x.w.p2 strength=2


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 92 / 76 / 6 / 18 | 13.3 | +28 [+15, +41] | +6 [-3, +15] | -6 [-14, +2] | 155, +17 [+6, +29], 27 / 10 | 7 / 13 / 2 | 1 / 6 / 12 / 1 | +5 (56) |
| x.rand.p2 | 64 / 102 / 5 / 21 | 4.0 | +0 [-7, +7] | +4 [+0, +9] | -3 [-7, +0] | 161, -1 [-7, +5], 4 / 5 | 1 / 5 / 1 | 1 / 0 / 2 / 3 | -1 (156) |
| x.rand.m2 | 64 / 93 / 6 / 29 | 4.3 | +0 [-6, +6] | -5 [-10, +0] | +5 [+1, +10] | 156, +0 [-6, +6], 4 / 4 | 6 / 1 / 1 | 1 / 5 / 1 / 0 | +1 (149) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 99 / 72 / 6 / 15; dR +19 [+5, +33]; answered-rate change +5.7 pp [+1.6, +9.9]; out / in 3 / 14; mean words 9.3 -> 17.9
- main effect: dR +28 (14.6% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +28 [+14, +42]; vs x.rand.m2 +28 [+15, +41] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +3.1 pp [-1.6, +7.8]; moved out / in 7 / 13 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 56 questions; mean words 4.0 -> 13.3
KEY family=E judge=pro layer=11 A=not_met B=not_met dR=+28 dR_F=+19 dAns=+6 dAns_F=+11 rand_margin=+28 EXPLORATORY base=x main=x.w.p2 strength=2


# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 121 / 52 / 1 / 18 | 12.9 | +43 [+30, +56] | +7 [-1, +16] | -5 [-13, +3] | 160, +31 [+19, +43], 37 / 6 | 6 / 13 / 0 | 0 / 6 / 12 / 1 | +6 (56) |
| x.rand.p2 | 77 / 86 / 1 / 28 | 3.9 | -1 [-9, +7] | -3 [-8, +2] | +5 [+1, +10] | 161, -1 [-8, +5], 5 / 6 | 5 / 2 / 0 | 2 / 3 / 2 / 0 | -2 (151) |
| x.rand.m2 | 78 / 92 / 1 / 21 | 4.2 | +0 [-7, +7] | +4 [+1, +8] | -2 [-5, +0] | 166, -2 [-9, +5], 5 / 7 | 0 / 4 / 0 | 0 / 0 / 2 / 2 | -2 (155) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 131 / 44 / 1 / 16; dR +29 [+15, +43]; answered-rate change +7.3 pp [+2.6, +12.5]; out / in 5 / 19; mean words 9.3 -> 16.2
- main effect: dR +43 (22.4% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +44 [+31, +57]; vs x.rand.m2 +43 [+30, +56] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +3.6 pp [-0.5, +8.3]; moved out / in 6 / 13 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +6 on 56 questions; mean words 4.0 -> 12.9
KEY family=E judge=allenai layer=12 A=not_met B=not_met dR=+43 dR_F=+29 dAns=+7 dAns_F=+14 rand_margin=+43 EXPLORATORY base=x main=x.w.p2 strength=2


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 89 / 79 / 4 / 20 | 12.9 | +25 [+13, +37] | +6 [-2, +14] | -4 [-11, +3] | 156, +14 [+4, +24], 21 / 7 | 6 / 12 / 1 | 0 / 6 / 11 / 1 | +6 (55) |
| x.rand.p2 | 61 / 95 / 7 / 29 | 3.9 | -3 [-10, +4] | -6 [-12, +0] | +5 [+1, +10] | 154, -2 [-8, +3], 3 / 5 | 8 / 2 / 3 | 3 / 5 / 2 / 0 | +1 (146) |
| x.rand.m2 | 67 / 97 / 5 / 23 | 4.2 | +3 [-5, +11] | +2 [-3, +7] | -1 [-3, +0] | 160, +2 [-5, +9], 7 / 5 | 2 / 4 / 2 | 2 / 0 / 3 / 1 | +3 (150) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 93 / 73 / 8 / 18; dR +13 [+0, +25]; answered-rate change +3.1 pp [-1.6, +7.8]; out / in 7 / 13; mean words 9.3 -> 16.2
- main effect: dR +25 (13.0% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +28 [+16, +40]; vs x.rand.m2 +22 [+10, +34] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +3.1 pp [-1.0, +7.3]; moved out / in 6 / 12 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +6 on 55 questions; mean words 4.0 -> 12.9
KEY family=E judge=pro layer=12 A=not_met B=not_met dR=+25 dR_F=+13 dAns=+6 dAns_F=+6 rand_margin=+22 EXPLORATORY base=x main=x.w.p2 strength=2


# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 111 / 63 / 3 / 15 | 16.3 | +33 [+20, +46] | +8 [+1, +15] | -8 [-14, -3] | 164, +26 [+14, +38], 33 / 7 | 2 / 10 / 2 | 1 / 1 / 8 / 2 | +4 (37) |
| x.rand.p2 | 77 / 88 / 2 / 25 | 4.2 | -1 [-9, +7] | -1 [-5, +2] | +2 [+0, +5] | 164, +0 [-8, +8], 8 / 8 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | -3 (152) |
| x.rand.m2 | 83 / 79 / 2 / 28 | 4.5 | +5 [-4, +14] | -4 [-10, +1] | +5 [+0, +10] | 160, +6 [-2, +14], 11 / 5 | 6 / 2 / 0 | 3 / 3 / 2 / 0 | +3 (144) |

- F, complete-sentence background (base x_F): x_F 102 / 59 / 1 / 30 -> with the E write 126 / 48 / 0 / 18; dR +24 [+10, +38]; answered-rate change +6.8 pp [+3.1, +10.9]; out / in 1 / 14; mean words 9.3 -> 14.7
- main effect: dR +33 (17.2% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +34 [+21, +47]; vs x.rand.m2 +28 [+15, +41] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +4.2 pp [+0.5, +7.8]; moved out / in 2 / 10 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +4 on 37 questions; mean words 4.0 -> 16.3
KEY family=E judge=allenai layer=13 A=not_met B=not_met dR=+33 dR_F=+24 dAns=+8 dAns_F=+13 rand_margin=+28 EXPLORATORY base=x main=x.w.p2 strength=2


# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 13

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p2 | 88 / 83 / 4 / 17 | 16.3 | +24 [+11, +38] | +9 [+3, +16] | -7 [-13, -2] | 161, +16 [+4, +29], 29 / 13 | 1 / 10 / 1 | 1 / 0 / 9 / 1 | +5 (38) |
| x.rand.p2 | 63 / 97 / 7 / 25 | 4.2 | -1 [-9, +7] | -2 [-5, +0] | +1 [+0, +3] | 160, +0 [-7, +7], 7 / 7 | 2 / 0 / 1 | 1 / 1 / 0 / 0 | +2 (149) |
| x.rand.m2 | 63 / 95 / 6 / 28 | 4.5 | -1 [-10, +8] | -4 [-10, +2] | +4 [+0, +9] | 155, -1 [-9, +7], 7 / 8 | 7 / 3 / 2 | 2 / 5 / 2 / 1 | +0 (140) |

- F, complete-sentence background (base x_F): x_F 80 / 80 / 2 / 30 -> with the E write 93 / 77 / 4 / 18; dR +13 [-1, +27]; answered-rate change +5.2 pp [+1.6, +8.9]; out / in 2 / 12; mean words 9.3 -> 14.7
- main effect: dR +24 (12.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: vs x.rand.p2 +25 [+11, +39]; vs x.rand.m2 +25 [+12, +39] | A: above each (point): True | B: each interval above 0: True
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: False | B: dR interval above 0 and answered-rate interval inside: False
- retention on x: answered-rate change +4.7 pp [+1.6, +8.3]; moved out / in 1 / 10 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +5 on 38 questions; mean words 4.0 -> 16.3
KEY family=E judge=pro layer=13 A=not_met B=not_met dR=+24 dR_F=+13 dAns=+9 dAns_F=+10 rand_margin=+25 EXPLORATORY base=x main=x.w.p2 strength=2



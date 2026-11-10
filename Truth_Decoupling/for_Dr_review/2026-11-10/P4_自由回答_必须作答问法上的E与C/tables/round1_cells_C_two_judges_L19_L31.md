# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 19

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 58 / 62 / 2 / 70 | 3.6 | -20 [-31, -10] | -46 [-58, -34] | +47 [+36, +59] | 119, -1 [-8, +5], 5 / 6 | 47 / 1 / 0 | 20 / 27 / 1 / 0 | -2 (109) |
| x.w.m1 | 86 / 96 / 2 / 8 | 4.3 | +8 [-2, +18] | +16 [+9, +24] | -15 [-23, -8] | 166, -2 [-10, +6], 7 / 9 | 0 / 16 / 0 | 0 / 0 / 10 / 6 | -2 (157) |
| x.md.p1 | 60 / 62 / 2 / 68 | 3.6 | -18 [-29, -7] | -44 [-56, -32] | +45 [+34, +57] | 121, +0 [-7, +7], 7 / 7 | 45 / 1 / 0 | 19 / 26 / 1 / 0 | -1 (114) |
| x.md.m1 | 80 / 101 / 1 / 10 | 4.2 | +2 [-9, +13] | +15 [+8, +23] | -13 [-20, -6] | 166, -8 [-17, +0], 6 / 14 | 0 / 15 / 0 | 0 / 0 / 10 / 5 | -9 (156) |
| x.rand.p1 | 75 / 90 / 2 / 25 | 4.1 | -3 [-8, +2] | -1 [-4, +2] | +2 [+0, +5] | 164, -2 [-7, +3], 2 / 4 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | -1 (154) |
| x.rand.m1 | 86 / 81 / 2 / 23 | 4.3 | +8 [+2, +14] | +1 [+0, +3] | +0 [+0, +0] | 166, +7 [+2, +13], 8 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +6 (161) |

- C toward caution (x.w.p1): dAns -46 (-24.0%), dX +47 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -45 [-57, -34]; vs x.rand.m1 -47 [-59, -35] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 119 questions answered both times (need at least 83): right-rate change -0.8 pp [-6.4, +4.6], W>R 5 / R>W 6, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 48.3% (answers dropped: 20 right, 27 wrong; added: 1 right, 0 wrong)
- C toward confident (x.w.m1): dAns +16 (8.3%), dX -15 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +17 [+9, +25]; vs x.rand.m1 +15 [+8, +23] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 166 questions answered both times (need at least 83): right-rate change -1.2 pp [-6.0, +3.6], W>R 7 / R>W 9, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 47.3% (answers dropped: 0 right, 0 wrong; added: 10 right, 6 wrong)
KEY family=C judge=allenai layer=19 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+16 Aplus=+46 Aminus=+16

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 19

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 47 / 72 / 3 / 70 | 3.6 | -17 [-27, -8] | -43 [-55, -31] | +46 [+35, +58] | 117, -5 [-11, +1], 2 / 7 | 45 / 2 / 2 | 14 / 31 / 2 / 0 | -4 (110) |
| x.w.m1 | 67 / 106 / 9 / 10 | 4.3 | +3 [-5, +11] | +11 [+5, +18] | -14 [-22, -7] | 162, -1 [-8, +6], 6 / 7 | 0 / 11 / 0 | 0 / 0 / 4 / 7 | -1 (153) |
| x.md.p1 | 52 / 69 / 4 / 67 | 3.6 | -12 [-22, -2] | -41 [-53, -30] | +43 [+32, +55] | 120, +0 [-7, +7], 6 / 6 | 42 / 1 / 2 | 13 / 29 / 1 / 0 | +0 (113) |
| x.md.m1 | 66 / 106 / 8 / 12 | 4.2 | +2 [-6, +10] | +10 [+4, +17] | -12 [-19, -6] | 162, -2 [-9, +5], 6 / 8 | 0 / 10 / 0 | 0 / 0 / 4 / 6 | -2 (152) |
| x.rand.p1 | 62 / 100 / 5 / 25 | 4.1 | -2 [-8, +4] | +0 [-4, +4] | +1 [+0, +3] | 160, -3 [-8, +2], 2 / 5 | 2 / 2 / 1 | 1 / 1 / 2 / 0 | +0 (152) |
| x.rand.m1 | 66 / 96 / 6 / 24 | 4.3 | +2 [-3, +8] | +0 [+0, +0] | +0 [+0, +0] | 162, +2 [-3, +8], 5 / 3 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +3 (157) |

- C toward caution (x.w.p1): dAns -43 (-22.4%), dX +46 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -43 [-55, -32]; vs x.rand.m1 -43 [-55, -31] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 117 questions answered both times (need at least 81): right-rate change -4.3 pp [-9.3, +0.8], W>R 2 / R>W 7, answered -> hedge 2 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 39.5% (answers dropped: 14 right, 31 wrong; added: 2 right, 0 wrong)
- C toward confident (x.w.m1): dAns +11 (5.7%), dX -14 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +11 [+4, +18]; vs x.rand.m1 +11 [+5, +18] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 162 questions answered both times (need at least 81): right-rate change -0.6 pp [-4.9, +3.7], W>R 6 / R>W 7, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 38.7% (answers dropped: 0 right, 0 wrong; added: 4 right, 7 wrong)
KEY family=C judge=pro layer=19 A=ONE-WAY(x.w.m1) B=not_met A_effect_only=TWO-WAY T=+11 Aplus=+43 Aminus=+11

# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 31

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 61 / 62 / 2 / 67 | 3.4 | -17 [-27, -7] | -43 [-55, -31] | +44 [+33, +56] | 121, +1 [-3, +5], 3 / 2 | 45 / 2 / 1 | 20 / 25 / 2 / 0 | +1 (119) |
| x.w.m1 | 86 / 96 / 3 / 7 | 4.3 | +8 [-1, +17] | +16 [+9, +24] | -16 [-24, -9] | 166, -1 [-7, +6], 5 / 6 | 0 / 16 / 0 | 0 / 0 / 9 / 7 | -2 (155) |
| x.md.p1 | 61 / 68 / 0 / 63 | 3.7 | -17 [-27, -8] | -37 [-49, -25] | +40 [+29, +51] | 126, -4 [-9, +0], 1 / 5 | 40 / 3 / 0 | 16 / 24 / 3 / 0 | -5 (124) |
| x.md.m1 | 87 / 95 / 3 / 7 | 4.6 | +9 [-1, +19] | +16 [+8, +24] | -16 [-24, -9] | 165, +1 [-7, +10], 10 / 9 | 1 / 17 / 1 | 1 / 0 / 9 / 8 | -1 (146) |
| x.rand.p1 | 77 / 87 / 3 / 25 | 4.1 | -1 [-6, +4] | -2 [-5, +0] | +2 [+0, +5] | 164, +0 [-5, +5], 3 / 3 | 2 / 0 / 0 | 1 / 1 / 0 / 0 | +0 (160) |
| x.rand.m1 | 79 / 86 / 4 / 23 | 4.3 | +1 [-5, +7] | -1 [-3, +0] | +0 [+0, +0] | 165, +2 [-3, +8], 5 / 3 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +1 (159) |

- C toward caution (x.w.p1): dAns -43 (-22.4%), dX +44 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -41 [-53, -30]; vs x.rand.m1 -42 [-54, -30] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 121 questions answered both times (need at least 83): right-rate change +0.8 pp [-2.7, +4.5], W>R 3 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 49.6% (answers dropped: 20 right, 25 wrong; added: 2 right, 0 wrong)
- C toward confident (x.w.m1): dAns +16 (8.3%), dX -16 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +18 [+10, +26]; vs x.rand.m1 +17 [+10, +25] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 166 questions answered both times (need at least 83): right-rate change -0.6 pp [-4.5, +3.5], W>R 5 / R>W 6, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 47.3% (answers dropped: 0 right, 0 wrong; added: 9 right, 7 wrong)
KEY family=C judge=allenai layer=31 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+16 Aplus=+43 Aminus=+16

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 31

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 48 / 75 / 3 / 66 | 3.4 | -16 [-25, -8] | -39 [-51, -28] | +42 [+31, +54] | 122, -4 [-9, +0], 1 / 5 | 40 / 1 / 0 | 13 / 27 / 1 / 0 | -3 (121) |
| x.w.m1 | 73 / 104 / 8 / 7 | 4.3 | +9 [+1, +17] | +15 [+8, +23] | -17 [-25, -10] | 162, +3 [-3, +10], 7 / 4 | 0 / 15 / 0 | 0 / 0 / 6 / 9 | +4 (151) |
| x.md.p1 | 50 / 76 / 5 / 61 | 3.7 | -14 [-22, -6] | -36 [-48, -25] | +37 [+26, +48] | 124, -2 [-6, +2], 1 / 3 | 38 / 2 / 2 | 13 / 25 / 1 / 1 | -1 (121) |
| x.md.m1 | 70 / 107 / 9 / 6 | 4.6 | +6 [-3, +15] | +15 [+8, +23] | -18 [-26, -10] | 161, +0 [-7, +7], 7 / 7 | 1 / 16 / 1 | 1 / 0 / 7 / 9 | +2 (144) |
| x.rand.p1 | 65 / 96 / 5 / 26 | 4.1 | +1 [-4, +6] | -1 [-4, +2] | +2 [+0, +5] | 160, +0 [-5, +5], 3 / 3 | 2 / 1 / 0 | 0 / 2 / 1 / 0 | +2 (155) |
| x.rand.m1 | 67 / 97 / 4 / 24 | 4.3 | +3 [-2, +8] | +2 [+0, +5] | +0 [+0, +0] | 162, +1 [-3, +5], 3 / 2 | 0 / 2 / 0 | 0 / 0 / 2 / 0 | +2 (157) |

- C toward caution (x.w.p1): dAns -39 (-20.3%), dX +42 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -38 [-50, -27]; vs x.rand.m1 -41 [-52, -30] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 122 questions answered both times (need at least 81): right-rate change -3.3 pp [-7.4, +0.0], W>R 1 / R>W 5, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 39.0% (answers dropped: 13 right, 27 wrong; added: 1 right, 0 wrong)
- C toward confident (x.w.m1): dAns +15 (7.8%), dX -17 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +16 [+9, +24]; vs x.rand.m1 +13 [+6, +21] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 162 questions answered both times (need at least 81): right-rate change +1.9 pp [-1.9, +6.1], W>R 7 / R>W 4, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 41.2% (answers dropped: 0 right, 0 wrong; added: 6 right, 9 wrong)
KEY family=C judge=pro layer=31 A=ONE-WAY(x.w.m1) B=not_met A_effect_only=TWO-WAY T=+15 Aplus=+39 Aminus=+15


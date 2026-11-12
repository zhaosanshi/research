# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 66 / 30 / 0 / 96 | 2.8 | -37 [-48, -26] | -70 [-83, -57] | +71 [+58, +84] | 96, -2 [-6, +2], 1 / 3 | 70 / 0 / 0 | 35 / 35 / 0 / 0 | -2 (91) |
| x.w.m1 | 112 / 74 / 2 / 4 | 4.3 | +9 [+0, +18] | +20 [+12, +29] | -21 [-30, -13] | 165, +0 [-6, +6], 5 / 5 | 1 / 21 / 1 | 1 / 0 / 10 / 11 | -2 (139) |
| x.md.p1 | 104 / 60 / 2 / 26 | 3.4 | +1 [-6, +7] | -2 [-6, +2] | +1 [-2, +5] | 163, +2 [-4, +8], 6 / 4 | 3 / 1 / 1 | 1 / 2 / 0 / 1 | +2 (145) |
| x.md.m1 | 98 / 60 / 1 / 33 | 3.3 | -5 [-13, +2] | -8 [-15, -2] | +8 [+3, +14] | 157, +0 [-6, +6], 5 / 5 | 9 / 1 / 1 | 5 / 4 / 0 / 1 | +0 (149) |
| x.rand.p1 | 105 / 59 / 1 / 27 | 3.3 | +2 [-3, +7] | -2 [-5, +0] | +2 [+0, +5] | 164, +2 [-3, +7], 4 / 2 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | +2 (157) |
| x.rand.m1 | 99 / 63 / 2 / 28 | 3.4 | -4 [-10, +1] | -4 [-9, +0] | +3 [-1, +8] | 161, +0 [-4, +4], 2 / 2 | 5 / 1 / 1 | 4 / 1 / 0 / 1 | +1 (157) |

- C toward caution (x.w.p1): dAns -70 (-36.5%), dX +71 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -68 [-81, -55]; vs x.rand.m1 -66 [-79, -53] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 96 questions answered both times (need at least 83): right-rate change -2.1 pp [-6.4, +2.0], W>R 1 / R>W 3, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 68.8% (answers dropped: 35 right, 35 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +20 (10.4%), dX -21 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +22 [+13, +32]; vs x.rand.m1 +24 [+15, +34] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 165 questions answered both times (need at least 83): right-rate change +0.0 pp [-3.7, +3.7], W>R 5 / R>W 5, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 60.2% (answers dropped: 1 right, 0 wrong; added: 10 right, 11 wrong)
KEY family=C judge=allenai layer=12 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+20 Aplus=+70 Aminus=+20

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 101 / 59 / 2 / 30 | 110 / 75 / 3 / 4 | +9 [+0, +19] | +25 [+16, +34] | -26 [-35, -17] | 160, -4, 3 / 7 | 0 / 25 / 0 | -2 (130) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 134 / 55 / 0 / 3 | +9 [+0, +18] | +13 [+7, +20] | -13 [-20, -7] | 176, +1, 7 / 6 | 0 / 13 / 0 | +2 (137) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 76 / 3 / 2 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 1 / 1 | 0 / 0 / 0 | +0 (185) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 111 / 67 / 1 / 13 | +0 [-8, +8] | -9 [-15, -4] | +11 [+5, +18] | 178, +6, 9 / 3 | 9 / 0 / 0 | +5 (174) |
| conf_E1.wq.p1 | 136 / 54 / 0 / 2 | 136 / 51 / 0 / 5 | +0 [-7, +7] | -3 [-7, +0] | +3 [+0, +7] | 187, +3, 7 / 4 | 3 / 0 / 0 | +0 (182) |

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 59 / 31 / 6 / 96 | 2.8 | -40 [-52, -28] | -73 [-86, -60] | +71 [+58, +84] | 90, -4 [-10, +1], 2 / 6 | 73 / 0 / 3 | 36 / 37 / 0 / 0 | -4 (88) |
| x.w.m1 | 100 / 86 / 1 / 5 | 4.3 | +1 [-8, +10] | +23 [+14, +32] | -20 [-29, -12] | 163, -7 [-14, -1], 2 / 9 | 0 / 23 / 0 | 0 / 0 / 8 / 15 | -3 (137) |
| x.md.p1 | 99 / 63 / 4 / 26 | 3.4 | +0 [-7, +7] | -1 [-5, +2] | +1 [-2, +5] | 161, +0 [-7, +7], 6 / 6 | 2 / 1 / 0 | 0 / 2 / 0 / 1 | +0 (143) |
| x.md.m1 | 90 / 68 / 1 / 33 | 3.3 | -9 [-16, -2] | -5 [-12, +1] | +8 [+3, +14] | 155, -6 [-12, -1], 1 / 7 | 8 / 3 / 0 | 4 / 4 / 1 / 2 | -4 (145) |
| x.rand.p1 | 99 / 62 / 4 / 27 | 3.3 | +0 [-5, +5] | -2 [-6, +2] | +2 [+0, +5] | 160, -1 [-6, +4], 3 / 4 | 3 / 1 / 1 | 0 / 3 / 1 / 0 | -1 (153) |
| x.rand.m1 | 95 / 65 / 4 / 28 | 3.4 | -4 [-10, +1] | -3 [-8, +1] | +3 [-1, +8] | 159, -1 [-5, +3], 2 / 3 | 4 / 1 / 0 | 3 / 1 / 0 / 1 | +0 (154) |

- C toward caution (x.w.p1): dAns -73 (-38.0%), dX +71 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -71 [-84, -58]; vs x.rand.m1 -70 [-83, -57] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 90 questions answered both times (need at least 82): right-rate change -4.4 pp [-11.0, +1.2], W>R 2 / R>W 6, answered -> hedge 3 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 65.6% (answers dropped: 36 right, 37 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +23 (12.0%), dX -20 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +25 [+16, +35]; vs x.rand.m1 +26 [+17, +36] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 163 questions answered both times (need at least 82): right-rate change -4.3 pp [-8.3, -0.6], W>R 2 / R>W 9, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 53.8% (answers dropped: 0 right, 0 wrong; added: 8 right, 15 wrong)
KEY family=C judge=pro layer=12 A=not_met B=not_met A_effect_only=TWO-WAY T=+23 Aplus=+73 Aminus=+23

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 92 / 65 / 5 / 30 | 101 / 84 / 2 / 5 | +9 [+1, +18] | +28 [+18, +38] | -25 [-34, -16] | 156, -3, 2 / 5 | 1 / 29 / 1 | -2 (128) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 121 / 65 / 1 / 5 | +7 [-2, +16] | +15 [+8, +23] | -12 [-19, -6] | 170, -1, 6 / 7 | 1 / 16 / 1 | -1 (135) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 100 / 87 / 1 / 4 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 187, +0, 1 / 1 | 0 / 0 / 0 | +0 (185) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 108 / 69 / 2 / 13 | +8 [+0, +16] | -10 [-17, -4] | +9 [+4, +15] | 177, +11, 12 / 1 | 10 / 0 / 1 | +10 (173) |
| conf_E1.wq.p1 | 117 / 68 / 2 / 5 | 116 / 64 / 6 / 6 | -1 [-8, +6] | -5 [-10, +0] | +1 [+0, +3] | 179, +0, 5 / 5 | 6 / 1 / 5 | +0 (177) |

# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 66 / 28 / 1 / 97 | 2.7 | -37 [-48, -26] | -72 [-85, -59] | +72 [+59, +85] | 94, +0 [-3, +3], 1 / 1 | 72 / 0 / 1 | 37 / 35 / 0 / 0 | +0 (91) |
| x.w.m1 | 110 / 76 / 2 / 4 | 4.2 | +7 [-2, +16] | +20 [+12, +29] | -21 [-30, -13] | 165, -2 [-8, +4], 4 / 6 | 1 / 21 / 1 | 1 / 0 / 10 / 11 | -3 (136) |
| x.md.p1 | 108 / 48 / 1 / 35 | 3.1 | +5 [-2, +12] | -10 [-17, -4] | +10 [+4, +16] | 156, +9 [+4, +15], 9 / 0 | 10 / 0 / 1 | 4 / 6 / 0 / 0 | +6 (145) |
| x.md.m1 | 101 / 77 / 1 / 13 | 4.0 | -2 [-11, +7] | +12 [+5, +20] | -12 [-19, -6] | 165, -5 [-13, +3], 6 / 11 | 1 / 13 / 1 | 1 / 0 / 4 / 9 | -3 (142) |
| x.rand.p1 | 99 / 60 / 1 / 32 | 3.1 | -4 [-10, +1] | -7 [-12, -2] | +7 [+2, +12] | 159, -2 [-7, +3], 2 / 4 | 7 / 0 / 0 | 2 / 5 / 0 / 0 | -1 (158) |
| x.rand.m1 | 106 / 59 / 3 / 24 | 3.6 | +3 [-3, +9] | -1 [-5, +3] | -1 [-5, +2] | 163, +5 [+0, +10], 6 / 1 | 3 / 2 / 2 | 2 / 1 / 0 / 2 | +4 (149) |

- C toward caution (x.w.p1): dAns -72 (-37.5%), dX +72 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -65 [-78, -53]; vs x.rand.m1 -71 [-84, -58] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 94 questions answered both times (need at least 83): right-rate change +0.0 pp [-3.2, +3.1], W>R 1 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 70.2% (answers dropped: 37 right, 35 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +20 (10.4%), dX -21 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +27 [+18, +37]; vs x.rand.m1 +21 [+13, +30] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 165 questions answered both times (need at least 83): right-rate change -1.2 pp [-4.9, +2.5], W>R 4 / R>W 6, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 59.1% (answers dropped: 1 right, 0 wrong; added: 10 right, 11 wrong)
KEY family=C judge=allenai layer=16 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+20 Aplus=+72 Aminus=+20

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 101 / 59 / 2 / 30 | 109 / 78 / 3 / 2 | +8 [-2, +18] | +27 [+18, +36] | -28 [-38, -19] | 160, -7, 2 / 9 | 0 / 27 / 0 | -7 (135) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 136 / 54 / 0 / 2 | +11 [+3, +19] | +14 [+7, +21] | -14 [-21, -7] | 176, +2, 5 / 3 | 0 / 14 / 0 | +1 (135) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 111 / 75 / 3 / 3 | +0 [-4, +4] | -1 [-3, +0] | +1 [+0, +3] | 186, +1, 2 / 1 | 1 / 0 / 0 | +0 (183) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 108 / 62 / 1 / 21 | -3 [-12, +6] | -17 [-25, -10] | +19 [+11, +27] | 170, +8, 10 / 2 | 17 / 0 / 0 | +5 (163) |
| conf_E1.wq.p1 | 136 / 54 / 0 / 2 | 129 / 52 / 0 / 11 | -7 [-15, +0] | -9 [-15, -4] | +9 [+4, +15] | 181, -2, 4 / 6 | 9 / 0 / 0 | -3 (176) |

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 60 / 31 / 4 / 97 | 2.7 | -39 [-51, -27] | -72 [-85, -59] | +72 [+59, +85] | 91, -5 [-11, +1], 2 / 7 | 72 / 0 / 1 | 34 / 38 / 0 / 0 | -4 (89) |
| x.w.m1 | 96 / 90 / 1 / 5 | 4.2 | -3 [-11, +5] | +23 [+14, +32] | -20 [-29, -12] | 163, -10 [-17, -4], 0 / 10 | 0 / 23 / 0 | 0 / 0 / 7 / 16 | -5 (133) |
| x.md.p1 | 98 / 51 / 7 / 36 | 3.1 | -1 [-9, +7] | -14 [-21, -7] | +11 [+5, +18] | 149, +5 [-1, +12], 8 / 3 | 14 / 0 / 3 | 6 / 8 / 0 / 0 | +4 (140) |
| x.md.m1 | 93 / 85 / 1 / 13 | 4.0 | -6 [-15, +2] | +15 [+8, +23] | -12 [-19, -6] | 163, -10 [-18, -3], 2 / 12 | 0 / 15 / 0 | 0 / 0 / 4 / 11 | -7 (139) |
| x.rand.p1 | 95 / 63 / 2 / 32 | 3.1 | -4 [-10, +2] | -5 [-11, +0] | +7 [+2, +12] | 156, -3 [-8, +2], 2 / 5 | 7 / 2 / 0 | 2 / 5 / 1 / 1 | -3 (155) |
| x.rand.m1 | 97 / 67 / 4 / 24 | 3.6 | -2 [-8, +4] | +1 [-3, +5] | -1 [-5, +2] | 161, -2 [-8, +3], 3 / 5 | 2 / 3 / 1 | 1 / 1 / 1 / 2 | +0 (146) |

- C toward caution (x.w.p1): dAns -72 (-37.5%), dX +72 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -67 [-80, -55]; vs x.rand.m1 -73 [-86, -60] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 91 questions answered both times (need at least 82): right-rate change -5.5 pp [-12.1, +1.1], W>R 2 / R>W 7, answered -> hedge 1 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 65.9% (answers dropped: 34 right, 38 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +23 (12.0%), dX -20 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +28 [+19, +38]; vs x.rand.m1 +22 [+14, +31] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 163 questions answered both times (need at least 82): right-rate change -6.1 pp [-10.1, -2.7], W>R 0 / R>W 10, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 51.6% (answers dropped: 0 right, 0 wrong; added: 7 right, 16 wrong)
KEY family=C judge=pro layer=16 A=not_met B=not_met A_effect_only=TWO-WAY T=+23 Aplus=+72 Aminus=+23

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 92 / 65 / 5 / 30 | 100 / 86 / 2 / 4 | +8 [+0, +17] | +29 [+19, +39] | -26 [-36, -17] | 156, -4, 2 / 6 | 1 / 30 / 1 | -6 (132) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 118 / 67 / 2 / 5 | +4 [-5, +13] | +14 [+7, +22] | -12 [-19, -6] | 170, -4, 3 / 7 | 1 / 15 / 1 | +0 (133) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 100 / 86 / 1 / 5 | +0 [-3, +3] | -1 [-3, +0] | +1 [+0, +3] | 186, +0, 1 / 1 | 1 / 0 / 0 | +0 (183) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 98 / 68 / 4 / 22 | -2 [-11, +7] | -21 [-30, -13] | +18 [+11, +26] | 166, +6, 9 / 3 | 21 / 0 / 3 | +5 (162) |
| conf_E1.wq.p1 | 117 / 68 / 2 / 5 | 117 / 60 / 3 / 12 | +0 [-8, +8] | -8 [-14, -3] | +7 [+2, +13] | 177, +3, 9 / 6 | 8 / 0 / 1 | +1 (172) |

# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 68 / 32 / 1 / 91 | 2.8 | -35 [-46, -24] | -66 [-79, -53] | +66 [+53, +79] | 100, -1 [-5, +2], 1 / 2 | 66 / 0 / 1 | 34 / 32 / 0 / 0 | -1 (97) |
| x.w.m1 | 114 / 72 / 2 / 4 | 4.3 | +11 [+1, +21] | +20 [+12, +29] | -21 [-30, -13] | 165, +2 [-5, +9], 8 / 6 | 1 / 21 / 1 | 1 / 0 / 10 / 11 | +1 (132) |
| x.md.p1 | 104 / 56 / 2 / 30 | 2.9 | +1 [-5, +7] | -6 [-12, -1] | +5 [+0, +10] | 159, +5 [+1, +10], 5 / 0 | 7 / 1 / 2 | 4 / 3 / 0 / 1 | +5 (156) |
| x.md.m1 | 110 / 62 / 1 / 19 | 4.4 | +7 [-1, +15] | +6 [+0, +13] | -6 [-12, +0] | 163, +5 [-2, +12], 9 / 4 | 3 / 9 / 1 | 1 / 2 / 3 / 6 | +5 (132) |
| x.rand.p1 | 102 / 64 / 2 / 24 | 3.3 | -1 [-8, +5] | +0 [-3, +3] | -1 [-3, +0] | 165, +0 [-6, +6], 5 / 5 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +1 (156) |
| x.rand.m1 | 105 / 58 / 2 / 27 | 3.1 | +2 [-2, +6] | -3 [-7, +0] | +2 [+0, +5] | 163, +3 [+0, +7], 3 / 0 | 3 / 0 / 1 | 1 / 2 / 0 / 0 | +3 (161) |

- C toward caution (x.w.p1): dAns -66 (-34.4%), dX +66 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -66 [-79, -53]; vs x.rand.m1 -63 [-76, -50] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 100 questions answered both times (need at least 83): right-rate change -1.0 pp [-4.6, +2.2], W>R 1 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 68.0% (answers dropped: 34 right, 32 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +20 (10.4%), dX -21 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +20 [+12, +29]; vs x.rand.m1 +23 [+14, +33] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 165 questions answered both times (need at least 83): right-rate change +1.2 pp [-3.1, +5.7], W>R 8 / R>W 6, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 61.3% (answers dropped: 1 right, 0 wrong; added: 10 right, 11 wrong)
KEY family=C judge=allenai layer=17 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+20 Aplus=+66 Aminus=+20

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 101 / 59 / 2 / 30 | 111 / 78 / 1 / 2 | +10 [+0, +20] | +29 [+20, +39] | -28 [-38, -19] | 160, -6, 3 / 9 | 0 / 29 / 0 | -6 (136) |
| caut_E1.wq.m1 | 125 / 51 / 0 / 16 | 136 / 54 / 0 / 2 | +11 [+2, +20] | +14 [+7, +21] | -14 [-21, -7] | 176, +2, 7 / 5 | 0 / 14 / 0 | +2 (128) |
| conf.iso.p1 | 111 / 76 / 3 / 2 | 110 / 75 / 3 / 4 | -1 [-5, +3] | -2 [-5, +0] | +2 [+0, +5] | 185, +1, 2 / 1 | 2 / 0 / 0 | +1 (182) |
| conf.wq.p1 | 111 / 76 / 3 / 2 | 108 / 66 / 2 / 16 | -3 [-11, +5] | -13 [-20, -7] | +14 [+7, +21] | 174, +5, 7 / 2 | 13 / 0 / 1 | +3 (167) |
| conf_E1.wq.p1 | 136 / 54 / 0 / 2 | 128 / 54 / 0 / 10 | -8 [-15, -1] | -8 [-14, -3] | +8 [+3, +14] | 182, -3, 3 / 6 | 8 / 0 / 0 | -4 (177) |

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 64 / 33 / 4 / 91 | 2.8 | -35 [-46, -24] | -66 [-79, -53] | +66 [+53, +79] | 97, -4 [-10, +1], 2 / 6 | 66 / 0 / 1 | 31 / 35 / 0 / 0 | -4 (95) |
| x.w.m1 | 98 / 86 / 4 / 4 | 4.3 | -1 [-9, +7] | +21 [+12, +30] | -21 [-30, -13] | 162, -8 [-15, -2], 1 / 9 | 1 / 22 / 1 | 0 / 1 / 7 / 15 | -3 (128) |
| x.md.p1 | 97 / 60 / 5 / 30 | 2.9 | -2 [-9, +5] | -6 [-12, -1] | +5 [+0, +10] | 156, +1 [-5, +7], 6 / 5 | 7 / 1 / 1 | 3 / 4 / 0 / 1 | +2 (153) |
| x.md.m1 | 94 / 75 / 4 / 19 | 4.4 | -5 [-12, +2] | +6 [+0, +13] | -6 [-12, +0] | 160, -6 [-13, +1], 3 / 9 | 3 / 9 / 1 | 0 / 3 / 1 / 8 | -3 (128) |
| x.rand.p1 | 93 / 71 / 4 / 24 | 3.3 | -6 [-12, -1] | +1 [-2, +5] | -1 [-3, +0] | 162, -5 [-11, +0], 1 / 6 | 1 / 2 / 1 | 1 / 0 / 0 / 2 | -3 (153) |
| x.rand.m1 | 98 / 63 / 4 / 27 | 3.1 | -1 [-6, +3] | -2 [-5, +0] | +2 [+0, +5] | 161, -1 [-6, +3], 2 / 3 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | +1 (159) |

- C toward caution (x.w.p1): dAns -66 (-34.4%), dX +66 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -67 [-80, -54]; vs x.rand.m1 -64 [-77, -51] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 97 questions answered both times (need at least 82): right-rate change -4.1 pp [-10.1, +1.1], W>R 2 / R>W 6, answered -> hedge 1 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 66.0% (answers dropped: 31 right, 35 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +21 (10.9%), dX -21 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +20 [+12, +29]; vs x.rand.m1 +23 [+14, +33] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 162 questions answered both times (need at least 82): right-rate change -4.9 pp [-8.9, -1.3], W>R 1 / R>W 9, answered -> hedge 1 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 53.3% (answers dropped: 0 right, 1 wrong; added: 7 right, 15 wrong)
KEY family=C judge=pro layer=17 A=not_met B=not_met A_effect_only=TWO-WAY T=+21 Aplus=+66 Aminus=+21

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 92 / 65 / 5 / 30 | 97 / 88 / 3 / 4 | +5 [-3, +13] | +28 [+18, +38] | -26 [-36, -17] | 156, -5, 2 / 7 | 1 / 29 / 1 | -6 (135) |
| caut_E1.wq.m1 | 114 / 57 / 4 / 17 | 120 / 67 / 0 / 5 | +6 [-3, +15] | +16 [+9, +24] | -12 [-19, -6] | 171, -4, 4 / 8 | 0 / 16 / 0 | -1 (126) |
| conf.iso.p1 | 100 / 87 / 1 / 4 | 100 / 86 / 1 / 5 | +0 [-3, +3] | -1 [-3, +0] | +1 [+0, +3] | 186, +0, 1 / 1 | 1 / 0 / 0 | +0 (183) |
| conf.wq.p1 | 100 / 87 / 1 / 4 | 103 / 71 / 2 / 16 | +3 [-3, +9] | -13 [-20, -7] | +12 [+6, +19] | 174, +7, 7 / 0 | 13 / 0 / 1 | +5 (166) |
| conf_E1.wq.p1 | 117 / 68 / 2 / 5 | 116 / 62 / 3 / 11 | -1 [-9, +7] | -7 [-12, -2] | +6 [+2, +11] | 178, +2, 7 / 5 | 7 / 0 / 1 | +0 (173) |

=== 2026-11-13 00:57:47 [qwen3.5-9b C tables gpunone] tables done ===

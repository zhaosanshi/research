# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 40 / 30 / 1 / 121 | 4.0 | -21 [-31, -11] | -47 [-59, -36] | +48 [+36, +60] | 70, +1 [-4, +6], 4 / 3 | 47 / 0 / 0 | 22 / 25 / 0 / 0 | +2 (64) |
| x.w.m1 | 92 / 79 / 6 / 15 | 5.2 | +31 [+20, +43] | +54 [+42, +67] | -58 [-70, -46] | 115, -3 [-8, +2], 2 / 5 | 2 / 56 / 2 | 0 / 2 / 34 / 22 | -1 (105) |
| x.md.p1 | 63 / 54 / 2 / 73 | 4.6 | +2 [-3, +8] | +0 [-3, +3] | +0 [-3, +3] | 116, +3 [-2, +8], 5 / 2 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +0 (104) |
| x.md.m1 | 64 / 57 / 1 / 70 | 4.6 | +3 [-3, +10] | +4 [-1, +10] | -3 [-8, +2] | 115, +1 [-4, +6], 4 / 3 | 2 / 6 / 0 | 1 / 1 / 3 / 3 | +2 (106) |
| x.rand.p1 | 59 / 60 / 2 / 71 | 4.6 | -2 [-6, +2] | +2 [+0, +5] | -2 [-5, +0] | 117, -2 [-6, +2], 1 / 3 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -3 (112) |
| x.rand.m1 | 58 / 51 / 3 / 80 | 4.6 | -3 [-8, +2] | -8 [-14, -2] | +7 [+2, +13] | 108, +0 [-4, +4], 2 / 2 | 9 / 1 / 1 | 3 / 6 / 0 / 1 | +1 (100) |

- C toward caution (x.w.p1): dAns -47 (-24.5%), dX +48 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -49 [-61, -37]; vs x.rand.m1 -39 [-50, -28] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 70 questions answered both times (need at least 59): right-rate change +1.4 pp [-5.9, +9.0], W>R 4 / R>W 3, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 57.1% (answers dropped: 22 right, 25 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +54 (28.1%), dX -58 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +52 [+40, +65]; vs x.rand.m1 +62 [+49, +75] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 115 questions answered both times (need at least 59): right-rate change -2.6 pp [-7.2, +1.8], W>R 2 / R>W 5, answered -> hedge 2 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 53.8% (answers dropped: 0 right, 2 wrong; added: 34 right, 22 wrong)
KEY family=C judge=allenai layer=12 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+47 Aplus=+47 Aminus=+54

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 44 / 49 / 2 / 97 | 84 / 87 / 2 / 19 | +40 [+28, +52] | +78 [+65, +91] | -78 [-91, -65] | 93, +0, 2 / 2 | 0 / 78 / 0 | +0 (86) |
| caut_E1.wq.m1 | 54 / 47 / 0 / 91 | 95 / 76 / 4 / 17 | +41 [+29, +54] | +70 [+57, +83] | -74 [-87, -61] | 101, -4, 1 / 5 | 0 / 70 / 0 | -3 (98) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 101 / 88 / 2 / 1 | +1 [-3, +5] | -1 [-3, +0] | +1 [+0, +3] | 189, +2, 3 / 1 | 1 / 0 / 0 | +2 (187) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 90 / 97 / 3 / 2 | -10 [-17, -3] | -3 [-7, +0] | +2 [+0, +5] | 187, -7, 2 / 9 | 3 / 0 / 1 | -5 (168) |
| conf_E1.wq.p1 | 105 / 83 / 3 / 1 | 107 / 82 / 1 / 2 | +2 [-6, +10] | +1 [-2, +4] | +1 [+0, +3] | 187, +0, 8 / 8 | 1 / 2 / 0 | +0 (174) |

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 37 / 33 / 1 / 121 | 4.0 | -17 [-26, -9] | -45 [-57, -34] | +47 [+35, +59] | 70, +1 [-2, +5], 2 / 1 | 45 / 0 / 0 | 18 / 27 / 0 / 0 | +1 (64) |
| x.w.m1 | 81 / 91 / 4 / 16 | 5.2 | +27 [+16, +38] | +57 [+44, +70] | -58 [-70, -46] | 114, -2 [-6, +2], 1 / 3 | 1 / 58 / 1 | 1 / 0 / 30 / 28 | -2 (104) |
| x.md.p1 | 50 / 62 / 7 / 73 | 4.6 | -4 [-10, +2] | -3 [-8, +1] | -1 [-5, +2] | 111, -1 [-6, +4], 3 / 4 | 4 / 1 / 3 | 3 / 1 / 0 / 1 | -1 (99) |
| x.md.m1 | 59 / 61 / 1 / 71 | 4.6 | +5 [+1, +10] | +5 [+0, +11] | -3 [-8, +2] | 114, +3 [+0, +7], 3 / 0 | 1 / 6 / 0 | 0 / 1 / 2 / 4 | +3 (106) |
| x.rand.p1 | 54 / 63 / 3 / 72 | 4.6 | +0 [+0, +0] | +2 [+0, +5] | -2 [-5, +0] | 115, +0 [+0, +0], 0 / 0 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | +0 (110) |
| x.rand.m1 | 51 / 57 / 4 / 80 | 4.6 | -3 [-8, +2] | -7 [-13, -2] | +6 [+0, +12] | 107, +0 [-4, +4], 2 / 2 | 8 / 1 / 1 | 3 / 5 / 0 / 1 | +0 (98) |

- C toward caution (x.w.p1): dAns -45 (-23.4%), dX +47 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -47 [-59, -36]; vs x.rand.m1 -38 [-50, -27] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 70 questions answered both times (need at least 58): right-rate change +1.4 pp [-3.2, +6.6], W>R 2 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 52.9% (answers dropped: 18 right, 27 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +57 (29.7%), dX -58 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +55 [+43, +68]; vs x.rand.m1 +64 [+52, +77] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change -1.8 pp [-5.4, +1.7], W>R 1 / R>W 3, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 47.1% (answers dropped: 1 right, 0 wrong; added: 30 right, 28 wrong)
KEY family=C judge=pro layer=12 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+45 Aplus=+45 Aminus=+57

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 42 / 52 / 1 / 97 | 71 / 94 / 7 / 20 | +29 [+18, +40] | +71 [+58, +85] | -77 [-90, -64] | 93, +0, 2 / 2 | 1 / 72 / 1 | +0 (86) |
| caut_E1.wq.m1 | 49 / 52 / 0 / 91 | 84 / 83 / 6 / 19 | +35 [+24, +46] | +66 [+53, +79] | -72 [-85, -59] | 101, +1, 2 / 1 | 0 / 66 / 0 | +2 (98) |
| conf.iso.p1 | 87 / 97 / 5 / 3 | 88 / 96 / 5 / 3 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 184, +1, 2 / 1 | 0 / 0 / 0 | +1 (183) |
| conf.wq.p1 | 87 / 97 / 5 / 3 | 82 / 99 / 6 / 5 | -5 [-12, +2] | -3 [-7, +0] | +2 [+0, +5] | 181, -3, 4 / 7 | 3 / 0 / 1 | -5 (162) |
| conf_E1.wq.p1 | 95 / 89 / 4 / 4 | 100 / 82 / 5 / 5 | +5 [-5, +15] | -2 [-7, +3] | +1 [-2, +5] | 180, +6, 13 / 7 | 4 / 2 / 2 | +2 (166) |

# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 40 / 32 / 1 / 119 | 3.7 | -21 [-31, -12] | -45 [-57, -34] | +46 [+35, +58] | 72, +0 [-5, +5], 3 / 3 | 45 / 0 / 0 | 21 / 24 / 0 / 0 | -1 (66) |
| x.w.m1 | 90 / 82 / 4 / 16 | 4.8 | +29 [+18, +40] | +55 [+42, +68] | -57 [-69, -45] | 115, +0 [-5, +5], 3 / 3 | 2 / 57 / 2 | 1 / 1 / 30 / 27 | +1 (109) |
| x.md.p1 | 63 / 55 / 1 / 73 | 4.5 | +2 [-3, +8] | +1 [-4, +6] | +0 [-5, +5] | 114, +2 [-3, +7], 4 / 2 | 3 / 4 / 0 | 1 / 2 / 1 / 3 | +0 (108) |
| x.md.m1 | 60 / 56 / 3 / 73 | 4.7 | -1 [-6, +4] | -1 [-5, +3] | +0 [-4, +4] | 114, -1 [-5, +3], 2 / 3 | 3 / 2 / 1 | 1 / 2 / 1 / 1 | +0 (104) |
| x.rand.p1 | 64 / 54 / 2 / 72 | 4.7 | +3 [-2, +8] | +1 [-2, +5] | -1 [-5, +2] | 116, +2 [-3, +7], 4 / 2 | 1 / 2 / 0 | 0 / 1 / 1 / 1 | +3 (111) |
| x.rand.m1 | 61 / 58 / 1 / 72 | 4.6 | +0 [-5, +5] | +2 [+0, +5] | -1 [-3, +0] | 117, -1 [-6, +3], 2 / 3 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | -1 (112) |

- C toward caution (x.w.p1): dAns -45 (-23.4%), dX +46 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -46 [-58, -35]; vs x.rand.m1 -47 [-59, -36] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 72 questions answered both times (need at least 59): right-rate change +0.0 pp [-6.6, +6.7], W>R 3 / R>W 3, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 55.6% (answers dropped: 21 right, 24 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +55 (28.6%), dX -57 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +54 [+41, +67]; vs x.rand.m1 +53 [+40, +66] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 115 questions answered both times (need at least 59): right-rate change +0.0 pp [-4.3, +4.2], W>R 3 / R>W 3, answered -> hedge 2 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 52.3% (answers dropped: 1 right, 1 wrong; added: 30 right, 27 wrong)
KEY family=C judge=allenai layer=11 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+45 Aplus=+45 Aminus=+55

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 44 / 49 / 2 / 97 | 86 / 83 / 2 / 21 | +42 [+31, +54] | +76 [+63, +89] | -76 [-90, -63] | 93, +4, 5 / 1 | 0 / 76 / 0 | +4 (88) |
| caut_E1.wq.m1 | 54 / 47 / 0 / 91 | 98 / 75 / 3 / 16 | +44 [+32, +56] | +72 [+59, +85] | -75 [-88, -62] | 101, -2, 1 / 3 | 0 / 72 / 0 | -1 (96) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 101 / 89 / 2 / 0 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 190, +1, 2 / 1 | 0 / 0 / 0 | +1 (189) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 94 / 95 / 2 / 1 | -6 [-14, +2] | -1 [-3, +0] | +1 [+0, +3] | 189, -5, 5 / 10 | 1 / 0 / 0 | -6 (171) |
| conf_E1.wq.p1 | 105 / 83 / 3 / 1 | 107 / 79 / 4 / 2 | +2 [-6, +10] | -2 [-6, +2] | +1 [+0, +3] | 185, +4, 10 / 6 | 3 / 1 / 2 | +2 (172) |

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 36 / 35 / 2 / 119 | 3.7 | -18 [-27, -10] | -44 [-56, -33] | +45 [+34, +57] | 71, +0 [-3, +3], 1 / 1 | 44 / 0 / 1 | 18 / 26 / 0 / 0 | +0 (65) |
| x.w.m1 | 76 / 93 / 6 / 17 | 4.8 | +22 [+12, +32] | +54 [+42, +67] | -57 [-69, -45] | 114, -1 [-5, +2], 1 / 2 | 1 / 55 / 1 | 1 / 0 / 24 / 31 | -1 (108) |
| x.md.p1 | 52 / 61 / 6 / 73 | 4.5 | -2 [-7, +3] | -2 [-8, +3] | -1 [-6, +4] | 110, -1 [-5, +2], 1 / 2 | 5 / 3 / 2 | 2 / 3 / 1 / 2 | -1 (103) |
| x.md.m1 | 55 / 62 / 1 / 74 | 4.7 | +1 [-2, +5] | +2 [-2, +6] | +0 [-4, +4] | 114, +0 [-3, +3], 1 / 1 | 1 / 3 / 0 | 0 / 1 / 1 / 2 | +0 (105) |
| x.rand.p1 | 56 / 59 / 5 / 72 | 4.7 | +2 [-3, +8] | +0 [-4, +4] | -2 [-6, +2] | 113, +2 [-3, +7], 4 / 2 | 2 / 2 / 1 | 1 / 1 / 1 / 1 | +3 (108) |
| x.rand.m1 | 52 / 64 / 4 / 72 | 4.6 | -2 [-6, +2] | +1 [-2, +5] | -2 [-5, +0] | 114, -1 [-5, +2], 1 / 2 | 1 / 2 / 1 | 1 / 0 / 0 / 2 | +0 (109) |

- C toward caution (x.w.p1): dAns -44 (-22.9%), dX +45 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -44 [-56, -33]; vs x.rand.m1 -45 [-57, -34] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 71 questions answered both times (need at least 58): right-rate change +0.0 pp [-4.1, +3.9], W>R 1 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 50.7% (answers dropped: 18 right, 26 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +54 (28.1%), dX -57 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +54 [+42, +66]; vs x.rand.m1 +53 [+41, +66] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change -0.9 pp [-4.0, +1.9], W>R 1 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 45.0% (answers dropped: 1 right, 0 wrong; added: 24 right, 31 wrong)
KEY family=C judge=pro layer=11 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+44 Aplus=+44 Aminus=+54

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 42 / 52 / 1 / 97 | 74 / 90 / 6 / 22 | +32 [+21, +43] | +70 [+56, +84] | -75 [-88, -62] | 93, +4, 5 / 1 | 1 / 71 / 1 | +4 (88) |
| caut_E1.wq.m1 | 49 / 52 / 0 / 91 | 88 / 82 / 5 / 17 | +39 [+28, +51] | +69 [+56, +82] | -74 [-87, -61] | 101, +1, 3 / 2 | 0 / 69 / 0 | +2 (96) |
| conf.iso.p1 | 87 / 97 / 5 / 3 | 88 / 96 / 5 / 3 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 184, +1, 2 / 1 | 0 / 0 / 0 | +1 (184) |
| conf.wq.p1 | 87 / 97 / 5 / 3 | 79 / 101 / 9 / 3 | -8 [-15, -2] | -4 [-8, -1] | +0 [+0, +0] | 180, -5, 2 / 7 | 4 / 0 / 4 | -5 (164) |
| conf_E1.wq.p1 | 95 / 89 / 4 / 4 | 98 / 85 / 4 / 5 | +3 [-6, +12] | -1 [-5, +3] | +1 [-2, +5] | 181, +4, 10 / 6 | 3 / 2 / 1 | +2 (168) |

# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 0 / 0 / 0 / 192 | 3.0 | -61 [-74, -48] | -117 [-130, -104] | +119 [+106, +132] | 0, +0 [+0, +0], 0 / 0 | 117 / 0 / 0 | 61 / 56 / 0 / 0 | +0 (0) |
| x.w.m1 | 100 / 89 / 3 / 0 | 5.2 | +39 [+26, +52] | +72 [+59, +85] | -73 [-86, -60] | 116, -5 [-11, +1], 2 / 7 | 1 / 73 / 1 | 0 / 1 / 44 / 29 | -1 (102) |
| x.md.p1 | 45 / 40 / 2 / 105 | 4.3 | -16 [-25, -7] | -32 [-44, -21] | +32 [+21, +44] | 81, +2 [-3, +7], 4 / 2 | 36 / 4 / 1 | 18 / 18 / 0 / 4 | +2 (77) |
| x.md.m1 | 86 / 75 / 2 / 29 | 5.7 | +25 [+13, +37] | +44 [+33, +56] | -44 [-56, -33] | 117, -1 [-8, +6], 6 / 7 | 0 / 44 / 0 | 0 / 0 / 26 / 18 | +3 (99) |
| x.rand.p1 | 63 / 51 / 2 / 76 | 4.9 | +2 [-4, +8] | -3 [-8, +2] | +3 [-2, +8] | 112, +4 [-1, +10], 6 / 2 | 5 / 2 / 0 | 2 / 3 / 0 / 2 | +2 (93) |
| x.rand.m1 | 57 / 60 / 3 / 72 | 4.3 | -4 [-10, +1] | +0 [-5, +6] | -1 [-6, +4] | 113, -3 [-8, +1], 1 / 4 | 4 / 4 / 1 | 2 / 2 / 1 / 3 | -2 (107) |

- C toward caution (x.w.p1): dAns -117 (-60.9%), dX +119 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -114 [-127, -101]; vs x.rand.m1 -117 [-130, -104] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 0 questions answered both times (need at least 59): right-rate change undefined, W>R 0 / R>W 0, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> undefined (answers dropped: 61 right, 56 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +72 (37.5%), dX -73 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +75 [+62, +88]; vs x.rand.m1 +72 [+59, +85] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 116 questions answered both times (need at least 59): right-rate change -4.3 pp [-9.6, +0.8], W>R 2 / R>W 7, answered -> hedge 1 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 52.9% (answers dropped: 0 right, 1 wrong; added: 44 right, 29 wrong)
KEY family=C judge=allenai layer=20 A=not_met B=not_met A_effect_only=TWO-WAY T=+72 Aplus=+117 Aminus=+72

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 20

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 44 / 49 / 2 / 97 | 95 / 93 / 4 / 0 | +51 [+39, +64] | +95 [+81, +108] | -97 [-111, -83] | 93, +0, 2 / 2 | 0 / 95 / 0 | +1 (84) |
| caut_E1.wq.m1 | 54 / 47 / 0 / 91 | 106 / 78 / 4 / 4 | +52 [+39, +65] | +83 [+70, +96] | -87 [-100, -74] | 101, -1, 2 / 3 | 0 / 83 / 0 | +0 (94) |
| conf.iso.p1 | 100 / 90 / 2 / 0 | 99 / 90 / 3 / 0 | -1 [-3, +0] | -1 [-3, +0] | +0 [+0, +0] | 189, -1, 0 / 1 | 1 / 0 / 1 | -1 (186) |
| conf.wq.p1 | 100 / 90 / 2 / 0 | 50 / 51 / 4 / 87 | -50 [-64, -36] | -89 [-103, -75] | +87 [+73, +101] | 101, +0, 6 / 6 | 89 / 0 / 3 | +0 (77) |
| conf_E1.wq.p1 | 105 / 83 / 3 / 1 | 73 / 52 / 6 / 61 | -32 [-46, -18] | -63 [-77, -50] | +60 [+47, +73] | 123, +8, 10 / 2 | 65 / 2 / 6 | +6 (106) |

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p1 | 0 / 0 / 0 / 192 | 3.0 | -54 [-67, -42] | -115 [-128, -102] | +118 [+105, +131] | 0, +0 [+0, +0], 0 / 0 | 115 / 0 / 0 | 54 / 61 / 0 / 0 | +0 (0) |
| x.w.m1 | 85 / 99 / 7 / 1 | 5.2 | +31 [+20, +43] | +69 [+56, +82] | -73 [-86, -60] | 114, -2 [-6, +2], 1 / 3 | 1 / 70 / 1 | 1 / 0 / 34 / 36 | -1 (102) |
| x.md.p1 | 38 / 43 / 5 / 106 | 4.3 | -16 [-25, -7] | -34 [-46, -22] | +32 [+21, +44] | 76, +3 [+0, +7], 3 / 0 | 39 / 5 / 4 | 19 / 20 / 0 / 5 | +3 (72) |
| x.md.m1 | 73 / 84 / 5 / 30 | 5.7 | +19 [+8, +30] | +42 [+31, +53] | -44 [-56, -33] | 115, +0 [-7, +7], 6 / 6 | 0 / 42 / 0 | 0 / 0 / 19 / 23 | +4 (98) |
| x.rand.p1 | 52 / 57 / 6 / 77 | 4.9 | -2 [-8, +4] | -6 [-12, +0] | +3 [-2, +8] | 107, +2 [-3, +7], 4 / 2 | 8 / 2 / 3 | 4 / 4 / 0 / 2 | +1 (88) |
| x.rand.m1 | 52 / 65 / 3 / 72 | 4.3 | -2 [-6, +2] | +2 [-3, +8] | -2 [-8, +3] | 112, -2 [-5, +0], 0 / 2 | 3 / 5 / 1 | 1 / 2 / 1 / 4 | -2 (107) |

- C toward caution (x.w.p1): dAns -115 (-59.9%), dX +118 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 -109 [-123, -96]; vs x.rand.m1 -117 [-130, -104] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 0 questions answered both times (need at least 58): right-rate change undefined, W>R 0 / R>W 0, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> undefined (answers dropped: 54 right, 61 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m1): dAns +69 (35.9%), dX -73 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p1 +75 [+62, +88]; vs x.rand.m1 +67 [+54, +80] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change -1.8 pp [-5.4, +1.7], W>R 1 / R>W 3, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 46.2% (answers dropped: 1 right, 0 wrong; added: 34 right, 36 wrong)
KEY family=C judge=pro layer=20 A=ONE-WAY(x.w.m1) B=not_met A_effect_only=TWO-WAY T=+69 Aplus=+115 Aminus=+69

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 20

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|
| caut.wq.m1 | 42 / 52 / 1 / 97 | 86 / 98 / 7 / 1 | +44 [+33, +56] | +90 [+76, +104] | -96 [-109, -82] | 93, +2, 2 / 0 | 1 / 91 / 1 | +2 (84) |
| caut_E1.wq.m1 | 49 / 52 / 0 / 91 | 93 / 89 / 6 / 4 | +44 [+31, +57] | +81 [+68, +94] | -87 [-100, -74] | 100, +0, 3 / 3 | 1 / 82 / 1 | +0 (93) |
| conf.iso.p1 | 87 / 97 / 5 / 3 | 87 / 97 / 5 / 3 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 184, +0, 1 / 1 | 0 / 0 / 0 | +0 (181) |
| conf.wq.p1 | 87 / 97 / 5 / 3 | 44 / 51 / 5 / 92 | -43 [-56, -30] | -89 [-103, -75] | +89 [+75, +103] | 95, +1, 6 / 5 | 89 / 0 / 4 | +0 (72) |
| conf_E1.wq.p1 | 95 / 89 / 4 / 4 | 68 / 54 / 2 / 68 | -27 [-41, -13] | -62 [-75, -50] | +64 [+51, +77] | 122, +9, 13 / 4 | 62 / 0 / 2 | +5 (103) |

=== 2026-11-13 15:55:37 [qwen3-8b C tables gpunone] tables done ===

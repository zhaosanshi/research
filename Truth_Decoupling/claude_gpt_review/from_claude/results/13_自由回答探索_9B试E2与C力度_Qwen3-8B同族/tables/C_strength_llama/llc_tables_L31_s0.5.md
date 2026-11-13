TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.5 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 31

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 72 / 79 / 1 / 40 | 3.7 | -6 [-12, +0] | -15 [-24, -7] | +17 [+10, +25] | 149, +0 [-3, +3], 1 / 1 | 17 / 2 / 0 | 7 / 10 / 1 / 1 | +0 (144) |
| x.w.m0.5 | 85 / 90 / 4 / 13 | 4.5 | +7 [+2, +13] | +9 [+4, +15] | -10 [-17, -4] | 166, +2 [-2, +6], 3 / 1 | 0 / 9 / 0 | 0 / 0 / 5 / 4 | +1 (156) |
| x.md.p0.5 | 73 / 82 / 2 / 35 | 4.1 | -5 [-11, +1] | -11 [-18, -5] | +12 [+6, +19] | 154, -2 [-6, +2], 1 / 3 | 12 / 1 / 0 | 4 / 8 / 1 / 0 | -3 (145) |
| x.md.m0.5 | 82 / 91 / 3 / 16 | 4.4 | +4 [-2, +10] | +7 [+2, +13] | -7 [-13, -2] | 165, +2 [-3, +7], 4 / 2 | 1 / 8 / 0 | 1 / 0 / 3 / 5 | +2 (152) |
| x.rand.p0.5 | 78 / 88 / 3 / 23 | 4.1 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 166, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (163) |
| x.rand.m0.5 | 80 / 87 / 2 / 23 | 4.1 | +2 [+0, +5] | +1 [+0, +3] | +0 [+0, +0] | 166, +1 [+0, +3], 1 / 0 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +1 (163) |

- C toward caution (x.w.p0.5): dAns -15 (-7.8%), dX +17 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -15 [-24, -7]; vs x.rand.m0.5 -16 [-24, -8] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 149 questions answered both times (need at least 83): right-rate change +0.0 pp [-2.0, +2.0], W>R 1 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 47.0% -> 47.7% (answers dropped: 7 right, 10 wrong; added: 1 right, 1 wrong)
- C toward confident (x.w.m0.5): dAns +9 (4.7%), dX -10 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +9 [+4, +15]; vs x.rand.m0.5 +8 [+2, +15] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 166 questions answered both times (need at least 83): right-rate change +1.2 pp [-1.2, +3.7], W>R 3 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 48.6% (answers dropped: 0 right, 0 wrong; added: 5 right, 4 wrong)
KEY family=C judge=allenai layer=31 A=TWO-WAY B=ONE-WAY(x.w.p0.5) A_effect_only=TWO-WAY T=+9 Aplus=+15 Aminus=+9 EXPLORATORY strength=0.5

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 31

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 31

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 59 / 88 / 6 / 39 | 3.7 | -5 [-10, +0] | -15 [-23, -8] | +15 [+8, +23] | 147, -1 [-4, +2], 1 / 2 | 15 / 0 / 1 | 4 / 11 / 0 / 0 | +1 (142) |
| x.w.m0.5 | 67 / 101 / 8 / 16 | 4.5 | +3 [-1, +8] | +6 [+2, +11] | -8 [-14, -3] | 162, +1 [-2, +5], 2 / 1 | 0 / 6 / 0 | 0 / 0 / 2 / 4 | +2 (152) |
| x.md.p0.5 | 58 / 90 / 8 / 36 | 4.1 | -6 [-12, -1] | -14 [-21, -8] | +12 [+6, +19] | 148, -1 [-5, +2], 1 / 2 | 14 / 0 / 2 | 5 / 9 / 0 / 0 | +1 (138) |
| x.md.m0.5 | 65 / 104 / 7 / 16 | 4.4 | +1 [-5, +8] | +7 [+2, +13] | -8 [-14, -3] | 162, -2 [-8, +3], 3 / 5 | 0 / 7 / 0 | 0 / 0 / 3 / 4 | -1 (150) |
| x.rand.p0.5 | 66 / 97 / 5 / 24 | 4.1 | +2 [-2, +6] | +1 [+0, +3] | +0 [+0, +0] | 162, +1 [-2, +5], 2 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (158) |
| x.rand.m0.5 | 67 / 96 / 5 / 24 | 4.1 | +3 [+0, +7] | +1 [+0, +3] | +0 [+0, +0] | 162, +2 [+0, +5], 2 / 0 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (159) |

- C toward caution (x.w.p0.5): dAns -15 (-7.8%), dX +15 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -16 [-24, -9]; vs x.rand.m0.5 -16 [-24, -9] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 147 questions answered both times (need at least 81): right-rate change -0.7 pp [-3.0, +1.4], W>R 1 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 40.1% (answers dropped: 4 right, 11 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +6 (3.1%), dX -8 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +5 [+1, +10]; vs x.rand.m0.5 +5 [+1, +10] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 162 questions answered both times (need at least 81): right-rate change +0.6 pp [-1.3, +2.9], W>R 2 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 39.5% -> 39.9% (answers dropped: 0 right, 0 wrong; added: 2 right, 4 wrong)
KEY family=C judge=pro layer=31 A=TWO-WAY B=ONE-WAY(x.w.m0.5) A_effect_only=TWO-WAY T=+6 Aplus=+15 Aminus=+6 EXPLORATORY strength=0.5

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 31

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 13:05:04 [llama3.1-8b run=llama3.1-8b C tables gpunone e_on=E1] tables done ===

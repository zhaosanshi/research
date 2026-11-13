TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.5 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 19

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 67 / 87 / 2 / 36 | 4.0 | -11 [-18, -4] | -12 [-19, -5] | +13 [+7, +20] | 153, -5 [-10, -1], 0 / 5 | 13 / 1 / 0 | 7 / 6 / 1 / 0 | -5 (146) |
| x.w.m0.5 | 84 / 91 / 2 / 15 | 4.1 | +6 [-1, +13] | +9 [+4, +15] | -8 [-14, -3] | 166, +1 [-5, +7], 5 / 4 | 0 / 9 / 0 | 0 / 0 / 5 / 4 | +1 (161) |
| x.md.p0.5 | 74 / 82 / 2 / 34 | 4.3 | -4 [-11, +3] | -10 [-17, -4] | +11 [+5, +18] | 155, +2 [-3, +8], 5 / 3 | 11 / 1 / 0 | 6 / 5 / 0 / 1 | +0 (147) |
| x.md.m0.5 | 83 / 92 / 2 / 15 | 4.1 | +5 [-2, +13] | +9 [+4, +15] | -8 [-14, -3] | 166, +0 [-6, +6], 5 / 5 | 0 / 9 / 0 | 0 / 0 / 5 / 4 | -1 (159) |
| x.rand.p0.5 | 74 / 92 / 2 / 24 | 4.0 | -4 [-8, -1] | +0 [-3, +3] | +1 [+0, +3] | 165, -4 [-8, -1], 0 / 4 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | -3 (162) |
| x.rand.m0.5 | 80 / 86 / 3 / 23 | 4.0 | +2 [-2, +6] | +0 [+0, +0] | +0 [+0, +0] | 166, +2 [-2, +6], 3 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +2 (163) |

- C toward caution (x.w.p0.5): dAns -12 (-6.2%), dX +13 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -12 [-19, -6]; vs x.rand.m0.5 -12 [-19, -5] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 153 questions answered both times (need at least 83): right-rate change -3.3 pp [-6.3, -0.7], W>R 0 / R>W 5, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 43.5% (answers dropped: 7 right, 6 wrong; added: 1 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +9 (4.7%), dX -8 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +9 [+4, +15]; vs x.rand.m0.5 +9 [+4, +15] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 166 questions answered both times (need at least 83): right-rate change +0.6 pp [-3.0, +4.1], W>R 5 / R>W 4, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 48.0% (answers dropped: 0 right, 0 wrong; added: 5 right, 4 wrong)
KEY family=C judge=allenai layer=19 A=ONE-WAY(x.w.m0.5) B=not_met A_effect_only=TWO-WAY T=+9 Aplus=+12 Aminus=+9 EXPLORATORY strength=0.5

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 19

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 19

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 58 / 92 / 6 / 36 | 4.0 | -6 [-13, +0] | -12 [-20, -5] | +12 [+6, +19] | 149, -2 [-7, +3], 2 / 4 | 13 / 1 / 2 | 5 / 8 / 1 / 0 | +0 (142) |
| x.w.m0.5 | 66 / 100 / 9 / 17 | 4.1 | +2 [-3, +8] | +4 [+1, +8] | -7 [-13, -2] | 162, +1 [-4, +6], 4 / 3 | 0 / 4 / 0 | 0 / 0 / 1 / 3 | +1 (157) |
| x.md.p0.5 | 62 / 91 / 5 / 34 | 4.3 | -2 [-8, +4] | -9 [-16, -3] | +10 [+4, +16] | 152, -1 [-6, +4], 3 / 4 | 10 / 1 / 0 | 2 / 8 / 1 / 0 | +2 (143) |
| x.md.m0.5 | 65 / 101 / 9 / 17 | 4.1 | +1 [-5, +7] | +4 [+1, +8] | -7 [-13, -2] | 162, +0 [-6, +5], 4 / 4 | 0 / 4 / 0 | 0 / 0 / 1 / 3 | +0 (155) |
| x.rand.p0.5 | 65 / 97 / 5 / 25 | 4.0 | +1 [-3, +6] | +0 [-3, +3] | +1 [+0, +3] | 161, +0 [-4, +4], 2 / 2 | 1 / 1 / 0 | 0 / 1 / 1 / 0 | +1 (158) |
| x.rand.m0.5 | 68 / 95 / 5 / 24 | 4.0 | +4 [+1, +8] | +1 [+0, +3] | +0 [+0, +0] | 162, +3 [+0, +7], 3 / 0 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +3 (159) |

- C toward caution (x.w.p0.5): dAns -12 (-6.2%), dX +12 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -12 [-19, -6]; vs x.rand.m0.5 -13 [-20, -7] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 149 questions answered both times (need at least 81): right-rate change -1.3 pp [-4.7, +2.0], W>R 2 / R>W 4, answered -> hedge 2 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 38.7% (answers dropped: 5 right, 8 wrong; added: 1 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +4 (2.1%), dX -7 | A: beyond the tolerance, abstentions move the same way: False | B: and both intervals on that side: False
    random controls: vs x.rand.p0.5 +4 [+0, +9]; vs x.rand.m0.5 +3 [-1, +8] | A: larger than each (point): True | B: each interval on that side: False
    retention on the 162 questions answered both times (need at least 81): right-rate change +0.6 pp [-2.5, +3.8], W>R 4 / R>W 3, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 39.8% (answers dropped: 0 right, 0 wrong; added: 1 right, 3 wrong)
KEY family=C judge=pro layer=19 A=ONE-WAY(x.w.p0.5) B=not_met A_effect_only=ONE-WAY(x.w.p0.5) T=+4 Aplus=+12 Aminus=+4 EXPLORATORY strength=0.5

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 19

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 13:04:38 [llama3.1-8b run=llama3.1-8b C tables gpunone e_on=E1] tables done ===

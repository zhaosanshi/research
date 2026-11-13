TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.25 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 19

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 72 / 87 / 2 / 31 | 4.0 | -6 [-11, -2] | -7 [-13, -2] | +8 [+3, +14] | 158, -2 [-5, +0], 0 / 2 | 8 / 1 / 0 | 4 / 4 / 0 / 1 | -2 (155) |
| x.w.m0.25 | 82 / 87 / 3 / 20 | 4.1 | +4 [+0, +9] | +3 [+0, +7] | -3 [-7, +0] | 166, +3 [-1, +7], 4 / 1 | 0 / 3 / 0 | 0 / 0 / 1 / 2 | +4 (162) |
| x.md.p0.25 | 76 / 85 / 2 / 29 | 4.1 | -2 [-6, +2] | -5 [-10, +0] | +6 [+2, +11] | 160, +0 [-3, +3], 1 / 1 | 6 / 1 / 0 | 2 / 4 / 0 / 1 | -1 (157) |
| x.md.m0.25 | 80 / 88 / 4 / 20 | 4.0 | +2 [-3, +8] | +2 [-2, +6] | -3 [-7, +0] | 165, +2 [-3, +7], 4 / 2 | 1 / 3 / 1 | 1 / 0 / 1 / 2 | +2 (162) |
| x.rand.p0.25 | 77 / 89 / 3 / 23 | 4.0 | -1 [-3, +0] | +0 [+0, +0] | +0 [+0, +0] | 166, -1 [-3, +0], 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (164) |
| x.rand.m0.25 | 79 / 87 / 3 / 23 | 4.0 | +1 [-2, +5] | +0 [+0, +0] | +0 [+0, +0] | 166, +1 [-2, +5], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (164) |

- C toward caution (x.w.p0.25): dAns -7 (-3.6%), dX +8 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -7 [-13, -2]; vs x.rand.m0.25 -7 [-13, -2] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 158 questions answered both times (need at least 83): right-rate change -1.3 pp [-3.2, +0.0], W>R 0 / R>W 2, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 45.3% (answers dropped: 4 right, 4 wrong; added: 0 right, 1 wrong)
- C toward confident (x.w.m0.25): dAns +3 (1.6%), dX -3 | A: beyond the tolerance, abstentions move the same way: False | B: and both intervals on that side: False
    random controls: vs x.rand.p0.25 +3 [+0, +7]; vs x.rand.m0.25 +3 [+0, +7] | A: larger than each (point): True | B: each interval on that side: False
    retention on the 166 questions answered both times (need at least 83): right-rate change +1.8 pp [-0.6, +4.5], W>R 4 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 48.5% (answers dropped: 0 right, 0 wrong; added: 1 right, 2 wrong)
KEY family=C judge=allenai layer=19 A=ONE-WAY(x.w.p0.25) B=not_met A_effect_only=ONE-WAY(x.w.p0.25) T=+3 Aplus=+7 Aminus=+3 EXPLORATORY strength=0.25

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 19

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 19

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 66 / 90 / 5 / 31 | 4.0 | +2 [-2, +6] | -6 [-12, -1] | +7 [+3, +13] | 155, +2 [+0, +5], 2 / 0 | 7 / 1 / 0 | 1 / 6 / 1 / 0 | +2 (152) |
| x.w.m0.25 | 66 / 97 / 8 / 21 | 4.1 | +2 [+0, +5] | +1 [+0, +3] | -3 [-7, +0] | 162, +2 [+0, +5], 2 / 0 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +2 (158) |
| x.md.p0.25 | 65 / 92 / 5 / 30 | 4.1 | +1 [-2, +5] | -5 [-10, +0] | +6 [+2, +11] | 156, +0 [-3, +3], 1 / 1 | 6 / 1 / 0 | 0 / 6 / 1 / 0 | +1 (153) |
| x.md.m0.25 | 65 / 98 / 8 / 21 | 4.0 | +1 [-2, +5] | +1 [+0, +3] | -3 [-7, +0] | 162, +1 [-2, +5], 2 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +1 (159) |
| x.rand.p0.25 | 67 / 96 / 5 / 24 | 4.0 | +3 [+0, +7] | +1 [+0, +3] | +0 [+0, +0] | 162, +2 [+0, +5], 2 / 0 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +2 (160) |
| x.rand.m0.25 | 66 / 96 / 6 / 24 | 4.0 | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | 162, +2 [+0, +5], 2 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (160) |

- C toward caution (x.w.p0.25): dAns -6 (-3.1%), dX +7 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -7 [-13, -3]; vs x.rand.m0.25 -6 [-12, -1] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 155 questions answered both times (need at least 81): right-rate change +1.3 pp [+0.0, +3.3], W>R 2 / R>W 0, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 42.3% (answers dropped: 1 right, 6 wrong; added: 1 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +1 (0.5%), dX -3 | A: beyond the tolerance, abstentions move the same way: False | B: and both intervals on that side: False
    random controls: vs x.rand.p0.25 +0 [-3, +3]; vs x.rand.m0.25 +1 [+0, +3] | A: larger than each (point): False | B: each interval on that side: False
    retention on the 162 questions answered both times (need at least 81): right-rate change +1.2 pp [+0.0, +3.1], W>R 2 / R>W 0, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 40.5% (answers dropped: 0 right, 0 wrong; added: 0 right, 1 wrong)
KEY family=C judge=pro layer=19 A=ONE-WAY(x.w.p0.25) B=not_met A_effect_only=ONE-WAY(x.w.p0.25) T=+1 Aplus=+6 Aminus=+1 EXPLORATORY strength=0.25

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 19

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 13:04:41 [llama3.1-8b run=llama3.1-8b C tables gpunone e_on=E1] tables done ===

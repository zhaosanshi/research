TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.25 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 98 / 53 / 0 / 41 | 3.2 | -5 [-11, +1] | -15 [-23, -8] | +16 [+9, +24] | 151, +2 [+0, +5], 2 / 0 | 15 / 0 / 0 | 7 / 8 / 0 / 0 | +2 (150) |
| x.w.m0.25 | 106 / 65 / 1 / 20 | 3.3 | +3 [-2, +9] | +5 [+1, +10] | -5 [-10, -1] | 166, +2 [-3, +7], 4 / 2 | 0 / 5 / 0 | 0 / 0 / 1 / 4 | +4 (161) |
| x.md.p0.25 | 104 / 61 / 1 / 26 | 3.2 | +1 [-2, +5] | -1 [-4, +2] | +1 [-2, +4] | 164, +1 [-2, +5], 2 / 1 | 2 / 1 / 0 | 0 / 2 / 0 / 1 | +1 (162) |
| x.md.m0.25 | 101 / 63 / 2 / 26 | 3.3 | -2 [-7, +3] | -2 [-5, +0] | +1 [+0, +3] | 164, +0 [-4, +4], 2 / 2 | 2 / 0 / 1 | 2 / 0 / 0 / 0 | +2 (160) |
| x.rand.p0.25 | 104 / 62 / 1 / 25 | 3.3 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 166, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (165) |
| x.rand.m0.25 | 103 / 63 / 1 / 25 | 3.2 | +0 [-3, +3] | +0 [-3, +3] | +0 [-3, +3] | 165, +1 [+0, +3], 1 / 0 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +1 (164) |

- C toward caution (x.w.p0.25): dAns -15 (-7.8%), dX +16 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -15 [-23, -8]; vs x.rand.m0.25 -15 [-23, -8] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 151 questions answered both times (need at least 83): right-rate change +1.3 pp [+0.0, +3.4], W>R 2 / R>W 0, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 64.9% (answers dropped: 7 right, 8 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +5 (2.6%), dX -5 | A: beyond the tolerance, abstentions move the same way: False | B: and both intervals on that side: False
    random controls: vs x.rand.p0.25 +5 [+1, +10]; vs x.rand.m0.25 +5 [+1, +10] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 166 questions answered both times (need at least 83): right-rate change +1.2 pp [-1.7, +4.2], W>R 4 / R>W 2, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 62.0% (answers dropped: 0 right, 0 wrong; added: 1 right, 4 wrong)
KEY family=C judge=allenai layer=12 A=ONE-WAY(x.w.p0.25) B=not_met A_effect_only=ONE-WAY(x.w.p0.25) T=+5 Aplus=+15 Aminus=+5 EXPLORATORY strength=0.25

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 91 / 55 / 5 / 41 | 3.2 | -8 [-15, -2] | -17 [-25, -10] | +16 [+9, +24] | 146, +0 [-4, +4], 2 / 2 | 17 / 0 / 1 | 8 / 9 / 0 / 0 | +0 (146) |
| x.w.m0.25 | 97 / 72 / 3 / 20 | 3.3 | -2 [-7, +3] | +6 [+2, +11] | -5 [-10, -1] | 163, -3 [-8, +1], 1 / 4 | 0 / 6 / 0 | 0 / 0 / 1 / 5 | +0 (158) |
| x.md.p0.25 | 99 / 63 / 4 / 26 | 3.2 | +0 [-4, +4] | -1 [-4, +2] | +1 [-2, +4] | 161, +0 [-4, +4], 2 / 2 | 2 / 1 / 0 | 0 / 2 / 0 / 1 | +0 (159) |
| x.md.m0.25 | 94 / 68 / 4 / 26 | 3.3 | -5 [-11, +0] | -1 [-3, +0] | +1 [+0, +3] | 162, -4 [-9, +0], 1 / 5 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | -1 (157) |
| x.rand.p0.25 | 100 / 63 / 4 / 25 | 3.3 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 163, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (162) |
| x.rand.m0.25 | 99 / 64 / 4 / 25 | 3.2 | +0 [-3, +3] | +0 [-3, +3] | +0 [-3, +3] | 162, +1 [+0, +3], 1 / 0 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +1 (161) |

- C toward caution (x.w.p0.25): dAns -17 (-8.9%), dX +16 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -17 [-25, -10]; vs x.rand.m0.25 -17 [-25, -10] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 146 questions answered both times (need at least 82): right-rate change +0.0 pp [-2.7, +2.8], W>R 2 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 60.7% -> 62.3% (answers dropped: 8 right, 9 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +6 (3.1%), dX -5 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +6 [+2, +11]; vs x.rand.m0.25 +6 [+2, +11] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 163 questions answered both times (need at least 82): right-rate change -1.8 pp [-4.6, +0.6], W>R 1 / R>W 4, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 57.4% (answers dropped: 0 right, 0 wrong; added: 1 right, 5 wrong)
KEY family=C judge=pro layer=12 A=TWO-WAY B=ONE-WAY(x.w.p0.25) A_effect_only=TWO-WAY T=+6 Aplus=+17 Aminus=+6 EXPLORATORY strength=0.25

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 09:48:15 [qwen3.5-9b C tables gpunone e_on=E1] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.5 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 90 / 44 / 0 / 58 | 3.0 | -13 [-22, -5] | -32 [-42, -22] | +33 [+23, +43] | 134, +3 [+0, +7], 3 / 0 | 32 / 0 / 0 | 16 / 16 / 0 / 0 | +3 (131) |
| x.w.m0.5 | 113 / 72 / 1 / 6 | 3.6 | +10 [+3, +18] | +19 [+11, +28] | -19 [-28, -11] | 166, +1 [-3, +5], 3 / 2 | 0 / 19 / 0 | 0 / 0 / 9 / 10 | +2 (154) |
| x.md.p0.5 | 101 / 63 / 1 / 27 | 3.3 | -2 [-7, +3] | -2 [-5, +0] | +2 [+0, +5] | 164, -2 [-7, +3], 2 / 4 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | -2 (153) |
| x.md.m0.5 | 104 / 60 / 2 / 26 | 3.2 | +1 [-5, +7] | -2 [-6, +2] | +1 [-2, +4] | 163, +3 [-2, +8], 5 / 2 | 3 / 1 / 1 | 2 / 1 / 0 / 1 | +3 (159) |
| x.rand.p0.5 | 105 / 58 / 1 / 28 | 3.3 | +2 [+0, +5] | -3 [-7, +0] | +3 [+0, +7] | 163, +2 [+0, +5], 2 / 0 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | +2 (161) |
| x.rand.m0.5 | 98 / 65 / 3 / 26 | 3.3 | -5 [-10, +0] | -3 [-8, +1] | +1 [-2, +5] | 162, -2 [-6, +2], 1 / 3 | 4 / 1 / 2 | 3 / 1 / 0 / 1 | +1 (159) |

- C toward caution (x.w.p0.5): dAns -32 (-16.7%), dX +33 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -29 [-39, -20]; vs x.rand.m0.5 -29 [-40, -19] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 134 questions answered both times (need at least 83): right-rate change +2.2 pp [+0.0, +5.1], W>R 3 / R>W 0, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 67.2% (answers dropped: 16 right, 16 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +19 (9.9%), dX -19 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +22 [+14, +31]; vs x.rand.m0.5 +22 [+14, +31] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 166 questions answered both times (need at least 83): right-rate change +0.6 pp [-1.9, +3.2], W>R 3 / R>W 2, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 61.1% (answers dropped: 0 right, 0 wrong; added: 9 right, 10 wrong)
KEY family=C judge=allenai layer=12 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+19 Aplus=+32 Aminus=+19 EXPLORATORY strength=0.5

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 82 / 47 / 5 / 58 | 3.0 | -17 [-26, -8] | -34 [-45, -24] | +33 [+23, +43] | 129, +0 [-5, +5], 3 / 3 | 34 / 0 / 1 | 17 / 17 / 0 / 0 | +0 (127) |
| x.w.m0.5 | 100 / 83 / 3 / 6 | 3.6 | +1 [-7, +9] | +20 [+12, +29] | -19 [-28, -11] | 163, -6 [-12, -1], 1 / 7 | 0 / 20 / 0 | 0 / 0 / 7 / 13 | -1 (151) |
| x.md.p0.5 | 100 / 62 / 3 / 27 | 3.3 | +1 [-4, +6] | -1 [-5, +2] | +2 [+0, +5] | 161, +0 [-5, +5], 3 / 3 | 2 / 1 / 0 | 0 / 2 / 1 / 0 | -1 (150) |
| x.md.m0.5 | 94 / 69 / 3 / 26 | 3.2 | -5 [-11, +1] | +0 [-4, +4] | +1 [-2, +4] | 161, -5 [-11, +0], 1 / 6 | 2 / 2 / 0 | 1 / 1 / 1 / 1 | -3 (156) |
| x.rand.p0.5 | 99 / 61 / 4 / 28 | 3.3 | +0 [-3, +3] | -3 [-7, +0] | +3 [+0, +7] | 160, +0 [-3, +3], 1 / 1 | 3 / 0 / 0 | 0 / 3 / 0 / 0 | +0 (158) |
| x.rand.m0.5 | 96 / 66 / 4 / 26 | 3.3 | -3 [-7, +0] | -1 [-5, +2] | +1 [-2, +5] | 161, -2 [-5, +0], 0 / 2 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +0 (156) |

- C toward caution (x.w.p0.5): dAns -34 (-17.7%), dX +33 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -31 [-41, -21]; vs x.rand.m0.5 -33 [-43, -23] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 129 questions answered both times (need at least 82): right-rate change +0.0 pp [-3.8, +3.8], W>R 3 / R>W 3, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 63.6% (answers dropped: 17 right, 17 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +20 (10.4%), dX -19 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +23 [+15, +32]; vs x.rand.m0.5 +21 [+13, +30] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 163 questions answered both times (need at least 82): right-rate change -3.7 pp [-7.3, -0.6], W>R 1 / R>W 7, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 54.6% (answers dropped: 0 right, 0 wrong; added: 7 right, 13 wrong)
KEY family=C judge=pro layer=12 A=ONE-WAY(x.w.p0.5) B=not_met A_effect_only=TWO-WAY T=+20 Aplus=+34 Aminus=+20 EXPLORATORY strength=0.5

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 09:48:11 [qwen3.5-9b C tables gpunone e_on=E1] tables done ===

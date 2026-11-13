TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.5 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 90 / 44 / 1 / 57 | 3.0 | -13 [-22, -4] | -32 [-42, -22] | +32 [+22, +42] | 134, +4 [+1, +8], 4 / 0 | 32 / 0 / 1 | 17 / 15 / 0 / 0 | +4 (133) |
| x.w.m0.5 | 111 / 71 / 3 / 7 | 3.7 | +8 [+1, +16] | +16 [+8, +24] | -18 [-26, -11] | 165, +3 [-2, +8], 5 / 2 | 1 / 17 / 1 | 1 / 0 / 6 / 11 | +0 (148) |
| x.md.p0.5 | 108 / 53 / 1 / 30 | 3.2 | +5 [-1, +11] | -5 [-10, -1] | +5 [+1, +10] | 161, +7 [+2, +12], 7 / 0 | 5 / 0 / 1 | 2 / 3 / 0 / 0 | +4 (154) |
| x.md.m0.5 | 100 / 71 / 2 / 19 | 3.5 | -3 [-10, +4] | +5 [+0, +10] | -6 [-11, -2] | 165, -2 [-9, +5], 5 / 7 | 1 / 6 / 1 | 1 / 0 / 0 / 6 | -2 (151) |
| x.rand.p0.5 | 102 / 61 / 1 / 28 | 3.1 | -1 [-5, +2] | -3 [-7, +0] | +3 [+0, +7] | 163, +0 [-3, +3], 1 / 1 | 3 / 0 / 0 | 1 / 2 / 0 / 0 | +0 (163) |
| x.rand.m0.5 | 106 / 60 / 2 / 24 | 3.5 | +3 [-2, +8] | +0 [-3, +3] | -1 [-3, +0] | 165, +4 [+0, +9], 5 / 1 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +3 (155) |

- C toward caution (x.w.p0.5): dAns -32 (-16.7%), dX +32 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -29 [-39, -20]; vs x.rand.m0.5 -32 [-43, -22] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 134 questions answered both times (need at least 83): right-rate change +3.0 pp [+0.7, +6.2], W>R 4 / R>W 0, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 67.2% (answers dropped: 17 right, 15 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +16 (8.3%), dX -18 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +19 [+11, +28]; vs x.rand.m0.5 +16 [+9, +24] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 165 questions answered both times (need at least 83): right-rate change +1.8 pp [-1.2, +5.1], W>R 5 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 61.0% (answers dropped: 1 right, 0 wrong; added: 6 right, 11 wrong)
KEY family=C judge=allenai layer=16 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+16 Aplus=+32 Aminus=+16 EXPLORATORY strength=0.5

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 82 / 48 / 4 / 58 | 3.0 | -17 [-26, -8] | -33 [-43, -23] | +33 [+23, +44] | 130, +0 [-5, +5], 3 / 3 | 33 / 0 / 1 | 17 / 16 / 0 / 0 | +0 (130) |
| x.w.m0.5 | 99 / 84 / 2 / 7 | 3.7 | +0 [-7, +7] | +20 [+12, +29] | -18 [-26, -11] | 163, -5 [-10, +0], 1 / 6 | 0 / 20 / 0 | 0 / 0 / 5 / 15 | -2 (145) |
| x.md.p0.5 | 100 / 57 / 5 / 30 | 3.2 | +1 [-6, +7] | -6 [-11, -2] | +5 [+1, +10] | 157, +3 [-3, +9], 6 / 3 | 6 / 0 / 1 | 2 / 4 / 0 / 0 | +1 (151) |
| x.md.m0.5 | 91 / 80 / 2 / 19 | 3.5 | -8 [-14, -2] | +8 [+3, +14] | -6 [-11, -2] | 163, -9 [-15, -4], 0 / 9 | 0 / 8 / 0 | 0 / 0 / 1 / 7 | -6 (148) |
| x.rand.p0.5 | 95 / 66 / 3 / 28 | 3.1 | -4 [-10, +1] | -2 [-6, +2] | +3 [+0, +7] | 160, -4 [-9, +0], 1 / 5 | 3 / 1 / 0 | 1 / 2 / 1 / 0 | -4 (160) |
| x.rand.m0.5 | 96 / 68 / 4 / 24 | 3.5 | -3 [-9, +3] | +1 [-2, +5] | -1 [-3, +0] | 162, -3 [-8, +2], 2 / 5 | 1 / 2 / 1 | 1 / 0 / 1 / 1 | +0 (152) |

- C toward caution (x.w.p0.5): dAns -33 (-17.2%), dX +33 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -31 [-41, -21]; vs x.rand.m0.5 -34 [-45, -24] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 130 questions answered both times (need at least 82): right-rate change +0.0 pp [-3.8, +3.8], W>R 3 / R>W 3, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 63.1% (answers dropped: 17 right, 16 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +20 (10.4%), dX -18 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +22 [+13, +31]; vs x.rand.m0.5 +19 [+11, +28] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 163 questions answered both times (need at least 82): right-rate change -3.1 pp [-6.3, +0.0], W>R 1 / R>W 6, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 54.1% (answers dropped: 0 right, 0 wrong; added: 5 right, 15 wrong)
KEY family=C judge=pro layer=16 A=ONE-WAY(x.w.p0.5) B=not_met A_effect_only=TWO-WAY T=+20 Aplus=+33 Aminus=+20 EXPLORATORY strength=0.5

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 09:51:41 [qwen3.5-9b C tables gpunone e_on=E1] tables done ===

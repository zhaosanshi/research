TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.25 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 97 / 53 / 0 / 42 | 3.1 | -6 [-13, +0] | -16 [-24, -9] | +17 [+10, +25] | 150, +2 [+0, +5], 2 / 0 | 16 / 0 / 0 | 8 / 8 / 0 / 0 | +2 (149) |
| x.w.m0.25 | 106 / 64 / 2 / 20 | 3.5 | +3 [-2, +8] | +4 [+0, +9] | -5 [-10, -1] | 165, +2 [-2, +6], 3 / 1 | 1 / 5 / 1 | 1 / 0 / 2 / 3 | +2 (157) |
| x.md.p0.25 | 104 / 57 / 2 / 29 | 3.1 | +1 [-3, +5] | -5 [-10, -1] | +4 [+1, +8] | 161, +3 [+0, +7], 3 / 0 | 5 / 0 / 1 | 2 / 3 / 0 / 0 | +2 (158) |
| x.md.m0.25 | 104 / 63 / 2 / 23 | 3.6 | +1 [-5, +7] | +1 [-2, +5] | -2 [-5, +0] | 165, +2 [-3, +8], 5 / 3 | 1 / 2 / 1 | 1 / 0 / 0 / 2 | +2 (155) |
| x.rand.p0.25 | 102 / 63 / 1 / 26 | 3.2 | -1 [-3, +0] | -1 [-3, +0] | +1 [+0, +3] | 165, -1 [-3, +0], 0 / 1 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -1 (163) |
| x.rand.m0.25 | 105 / 60 / 2 / 25 | 3.3 | +2 [-2, +6] | -1 [-3, +0] | +0 [+0, +0] | 165, +3 [+0, +7], 3 / 0 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +3 (163) |

- C toward caution (x.w.p0.25): dAns -16 (-8.3%), dX +17 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -15 [-23, -8]; vs x.rand.m0.25 -15 [-23, -8] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 150 questions answered both times (need at least 83): right-rate change +1.3 pp [+0.0, +3.4], W>R 2 / R>W 0, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 64.7% (answers dropped: 8 right, 8 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +4 (2.1%), dX -5 | A: beyond the tolerance, abstentions move the same way: False | B: and both intervals on that side: False
    random controls: vs x.rand.p0.25 +5 [+0, +10]; vs x.rand.m0.25 +5 [+1, +10] | A: larger than each (point): True | B: each interval on that side: False
    retention on the 165 questions answered both times (need at least 83): right-rate change +1.2 pp [-1.2, +3.7], W>R 3 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 62.4% (answers dropped: 1 right, 0 wrong; added: 2 right, 3 wrong)
KEY family=C judge=allenai layer=16 A=ONE-WAY(x.w.p0.25) B=not_met A_effect_only=ONE-WAY(x.w.p0.25) T=+4 Aplus=+16 Aminus=+4 EXPLORATORY strength=0.25

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 16

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 90 / 55 / 5 / 42 | 3.1 | -9 [-16, -2] | -18 [-26, -11] | +17 [+10, +25] | 145, +0 [-4, +4], 2 / 2 | 18 / 0 / 1 | 9 / 9 / 0 / 0 | +0 (145) |
| x.w.m0.25 | 97 / 72 / 3 / 20 | 3.5 | -2 [-7, +3] | +6 [+2, +11] | -5 [-10, -1] | 163, -4 [-8, -1], 0 / 4 | 0 / 6 / 0 | 0 / 0 / 2 / 4 | -1 (154) |
| x.md.p0.25 | 97 / 61 / 5 / 29 | 3.1 | -2 [-8, +3] | -5 [-10, -1] | +4 [+1, +8] | 158, +0 [-5, +5], 3 / 3 | 5 / 0 / 1 | 2 / 3 / 0 / 0 | -1 (156) |
| x.md.m0.25 | 93 / 72 / 4 / 23 | 3.6 | -6 [-11, -2] | +2 [+0, +5] | -2 [-5, +0] | 163, -6 [-11, -2], 0 / 6 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -2 (152) |
| x.rand.p0.25 | 95 / 67 / 4 / 26 | 3.2 | -4 [-8, -1] | -1 [-3, +0] | +1 [+0, +3] | 162, -4 [-8, -1], 0 / 4 | 1 / 0 / 0 | 0 / 1 / 0 / 0 | -4 (160) |
| x.rand.m0.25 | 98 / 64 / 5 / 25 | 3.3 | -1 [-5, +2] | -1 [-3, +0] | +0 [+0, +0] | 162, +0 [-3, +3], 1 / 1 | 1 / 0 / 1 | 1 / 0 / 0 / 0 | +0 (160) |

- C toward caution (x.w.p0.25): dAns -18 (-9.4%), dX +17 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -17 [-25, -10]; vs x.rand.m0.25 -17 [-25, -10] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 145 questions answered both times (need at least 82): right-rate change +0.0 pp [-2.7, +2.8], W>R 2 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 60.7% -> 62.1% (answers dropped: 9 right, 9 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +6 (3.1%), dX -5 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +7 [+3, +12]; vs x.rand.m0.25 +7 [+2, +12] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 163 questions answered both times (need at least 82): right-rate change -2.5 pp [-5.0, -0.6], W>R 0 / R>W 4, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 57.4% (answers dropped: 0 right, 0 wrong; added: 2 right, 4 wrong)
KEY family=C judge=pro layer=16 A=TWO-WAY B=ONE-WAY(x.w.p0.25) A_effect_only=TWO-WAY T=+6 Aplus=+18 Aminus=+6 EXPLORATORY strength=0.25

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 16

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 09:51:45 [qwen3.5-9b C tables gpunone e_on=E1] tables done ===

TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.25 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 98 / 55 / 0 / 39 | 3.2 | -5 [-11, +1] | -13 [-20, -7] | +14 [+7, +21] | 153, +2 [+0, +5], 2 / 0 | 13 / 0 / 0 | 7 / 6 / 0 / 0 | +2 (152) |
| x.w.m0.25 | 105 / 65 / 2 / 20 | 3.5 | +2 [-3, +8] | +4 [+0, +9] | -5 [-10, -1] | 165, +0 [-4, +4], 2 / 2 | 1 / 5 / 1 | 1 / 0 / 3 / 2 | +1 (158) |
| x.md.p0.25 | 104 / 59 / 2 / 27 | 3.1 | +1 [-3, +5] | -3 [-8, +1] | +2 [-2, +6] | 162, +3 [+0, +7], 3 / 0 | 4 / 1 / 1 | 2 / 2 / 0 / 1 | +2 (160) |
| x.md.m0.25 | 103 / 63 / 2 / 24 | 3.6 | +0 [-5, +5] | +0 [-3, +3] | -1 [-3, +0] | 165, +1 [-3, +6], 3 / 2 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +1 (156) |
| x.rand.p0.25 | 102 / 64 / 2 / 24 | 3.2 | -1 [-3, +0] | +0 [-3, +3] | -1 [-3, +0] | 165, +0 [+0, +0], 0 / 0 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (165) |
| x.rand.m0.25 | 104 / 62 / 1 / 25 | 3.3 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 166, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (165) |

- C toward caution (x.w.p0.25): dAns -13 (-6.8%), dX +14 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -13 [-21, -6]; vs x.rand.m0.25 -13 [-20, -7] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 153 questions answered both times (need at least 83): right-rate change +1.3 pp [+0.0, +3.3], W>R 2 / R>W 0, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 64.1% (answers dropped: 7 right, 6 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +4 (2.1%), dX -5 | A: beyond the tolerance, abstentions move the same way: False | B: and both intervals on that side: False
    random controls: vs x.rand.p0.25 +4 [+1, +8]; vs x.rand.m0.25 +4 [+0, +9] | A: larger than each (point): True | B: each interval on that side: False
    retention on the 165 questions answered both times (need at least 83): right-rate change +0.0 pp [-2.4, +2.4], W>R 2 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 62.0% -> 61.8% (answers dropped: 1 right, 0 wrong; added: 3 right, 2 wrong)
KEY family=C judge=allenai layer=17 A=ONE-WAY(x.w.p0.25) B=not_met A_effect_only=ONE-WAY(x.w.p0.25) T=+4 Aplus=+13 Aminus=+4 EXPLORATORY strength=0.25

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 92 / 56 / 5 / 39 | 3.2 | -7 [-14, +0] | -15 [-22, -8] | +14 [+7, +21] | 148, +1 [-3, +5], 3 / 2 | 15 / 0 / 1 | 8 / 7 / 0 / 0 | +1 (148) |
| x.w.m0.25 | 97 / 72 / 3 / 20 | 3.5 | -2 [-7, +3] | +6 [+2, +11] | -5 [-10, -1] | 163, -4 [-8, -1], 0 / 4 | 0 / 6 / 0 | 0 / 0 / 2 / 4 | +0 (155) |
| x.md.p0.25 | 98 / 63 / 4 / 27 | 3.1 | -1 [-6, +4] | -2 [-6, +2] | +2 [-2, +6] | 160, +0 [-5, +5], 3 / 3 | 3 / 1 / 0 | 1 / 2 / 0 / 1 | -1 (157) |
| x.md.m0.25 | 93 / 71 / 4 / 24 | 3.6 | -6 [-11, -2] | +1 [+0, +3] | -1 [-3, +0] | 163, -6 [-11, -2], 0 / 6 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -2 (153) |
| x.rand.p0.25 | 99 / 65 / 4 / 24 | 3.2 | +0 [+0, +0] | +1 [+0, +3] | -1 [-3, +0] | 163, +0 [+0, +0], 0 / 0 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +0 (162) |
| x.rand.m0.25 | 100 / 63 / 4 / 25 | 3.3 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 163, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (162) |

- C toward caution (x.w.p0.25): dAns -15 (-7.8%), dX +14 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -16 [-24, -9]; vs x.rand.m0.25 -15 [-22, -8] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 148 questions answered both times (need at least 82): right-rate change +0.7 pp [-2.1, +3.6], W>R 3 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 62.2% (answers dropped: 8 right, 7 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +6 (3.1%), dX -5 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +5 [+1, +10]; vs x.rand.m0.25 +6 [+2, +11] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 163 questions answered both times (need at least 82): right-rate change -2.5 pp [-5.0, -0.6], W>R 0 / R>W 4, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 57.4% (answers dropped: 0 right, 0 wrong; added: 2 right, 4 wrong)
KEY family=C judge=pro layer=17 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+6 Aplus=+15 Aminus=+6 EXPLORATORY strength=0.25

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 09:56:15 [qwen3.5-9b C tables gpunone e_on=E1] tables done ===

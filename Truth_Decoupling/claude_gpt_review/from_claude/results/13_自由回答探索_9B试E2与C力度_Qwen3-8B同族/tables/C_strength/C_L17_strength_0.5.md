TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3.5-9b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.5 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 103 / 63 / 1 / 25; rule abstentions 25; mean words 3.2

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 87 / 48 / 2 / 55 | 3.1 | -16 [-24, -8] | -31 [-41, -21] | +30 [+21, +40] | 135, +1 [+0, +3], 1 / 0 | 31 / 0 / 2 | 17 / 14 / 0 / 0 | +1 (133) |
| x.w.m0.5 | 112 / 70 / 3 / 7 | 3.8 | +9 [+2, +17] | +16 [+8, +24] | -18 [-26, -11] | 165, +3 [-2, +8], 5 / 2 | 1 / 17 / 1 | 1 / 0 / 7 / 10 | +2 (149) |
| x.md.p0.5 | 105 / 57 / 3 / 27 | 3.0 | +2 [-3, +8] | -4 [-9, +0] | +2 [-2, +6] | 161, +5 [+1, +10], 5 / 0 | 5 / 1 / 2 | 3 / 2 / 0 / 1 | +5 (159) |
| x.md.m0.5 | 105 / 62 / 2 / 23 | 3.9 | +2 [-4, +8] | +1 [-3, +6] | -2 [-6, +2] | 164, +2 [-3, +8], 5 / 3 | 2 / 3 / 1 | 1 / 1 / 1 / 2 | +1 (146) |
| x.rand.p0.5 | 101 / 65 / 2 / 24 | 3.3 | -2 [-5, +0] | +0 [-3, +3] | -1 [-3, +0] | 165, -1 [-3, +0], 0 / 1 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (162) |
| x.rand.m0.5 | 105 / 59 / 1 / 27 | 3.2 | +2 [+0, +5] | -2 [-5, +0] | +2 [+0, +5] | 164, +2 [+0, +5], 2 / 0 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | +2 (164) |

- C toward caution (x.w.p0.5): dAns -31 (-16.1%), dX +30 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -31 [-42, -21]; vs x.rand.m0.5 -29 [-39, -20] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 135 questions answered both times (need at least 83): right-rate change +0.7 pp [+0.0, +2.3], W>R 1 / R>W 0, answered -> hedge 2 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 62.0% -> 64.4% (answers dropped: 17 right, 14 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +16 (8.3%), dX -18 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +16 [+9, +24]; vs x.rand.m0.5 +18 [+10, +27] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 165 questions answered both times (need at least 83): right-rate change +1.8 pp [-1.2, +5.0], W>R 5 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 62.0% -> 61.5% (answers dropped: 1 right, 0 wrong; added: 7 right, 10 wrong)
KEY family=C judge=allenai layer=17 A=TWO-WAY B=ONE-WAY(x.w.p0.5) A_effect_only=TWO-WAY T=+16 Aplus=+31 Aminus=+16 EXPLORATORY strength=0.5

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 99 / 64 / 4 / 25; rule abstentions 25; mean words 3.2

## layer 17

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 82 / 50 / 5 / 55 | 3.1 | -17 [-26, -8] | -31 [-41, -22] | +30 [+21, +40] | 132, -1 [-5, +3], 2 / 3 | 31 / 0 / 1 | 16 / 15 / 0 / 0 | -1 (130) |
| x.w.m0.5 | 99 / 83 / 3 / 7 | 3.8 | +0 [-7, +7] | +19 [+11, +28] | -18 [-26, -11] | 163, -6 [-12, -1], 1 / 7 | 0 / 19 / 0 | 0 / 0 / 6 / 13 | -2 (146) |
| x.md.p0.5 | 99 / 63 / 3 / 27 | 3.0 | +0 [-6, +6] | -1 [-6, +3] | +2 [-2, +6] | 160, +0 [-5, +6], 4 / 4 | 3 / 2 / 0 | 1 / 2 / 1 / 1 | +0 (157) |
| x.md.m0.5 | 92 / 72 / 5 / 23 | 3.9 | -7 [-13, -2] | +1 [-3, +5] | -2 [-6, +2] | 161, -7 [-13, -2], 0 / 7 | 2 / 3 / 1 | 0 / 2 / 0 / 3 | -2 (142) |
| x.rand.p0.5 | 97 / 67 / 4 / 24 | 3.3 | -2 [-5, +0] | +1 [+0, +3] | -1 [-3, +0] | 163, -2 [-5, +0], 0 / 2 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -1 (159) |
| x.rand.m0.5 | 100 / 61 / 4 / 27 | 3.2 | +1 [+0, +3] | -2 [-5, +0] | +2 [+0, +5] | 161, +1 [+0, +3], 1 / 0 | 2 / 0 / 0 | 0 / 2 / 0 / 0 | +1 (161) |

- C toward caution (x.w.p0.5): dAns -31 (-16.1%), dX +30 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -32 [-42, -22]; vs x.rand.m0.5 -29 [-39, -20] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 132 questions answered both times (need at least 82): right-rate change -0.8 pp [-4.2, +2.4], W>R 2 / R>W 3, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 62.1% (answers dropped: 16 right, 15 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +19 (9.9%), dX -18 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +18 [+10, +26]; vs x.rand.m0.5 +21 [+13, +30] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 163 questions answered both times (need at least 82): right-rate change -3.7 pp [-7.2, -0.6], W>R 1 / R>W 7, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 60.7% -> 54.4% (answers dropped: 0 right, 0 wrong; added: 6 right, 13 wrong)
KEY family=C judge=pro layer=17 A=ONE-WAY(x.w.p0.5) B=not_met A_effect_only=TWO-WAY T=+19 Aplus=+31 Aminus=+19 EXPLORATORY strength=0.5

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 17

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 09:56:11 [qwen3.5-9b C tables gpunone e_on=E1] tables done ===

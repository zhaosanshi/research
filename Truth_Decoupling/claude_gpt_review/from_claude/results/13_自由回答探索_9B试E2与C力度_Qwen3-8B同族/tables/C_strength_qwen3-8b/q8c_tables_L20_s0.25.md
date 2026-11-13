TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.25 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 43 / 30 / 1 / 118 | 4.0 | -18 [-27, -10] | -44 [-56, -33] | +45 [+34, +57] | 73, +1 [-2, +4], 2 / 1 | 44 / 0 / 0 | 19 / 25 / 0 / 0 | +0 (68) |
| x.w.m0.25 | 75 / 75 / 3 / 39 | 4.9 | +14 [+6, +22] | +33 [+23, +44] | -34 [-44, -24] | 116, -1 [-5, +2], 1 / 2 | 1 / 34 / 1 | 0 / 1 / 15 / 19 | +0 (112) |
| x.md.p0.25 | 59 / 48 / 2 / 83 | 4.6 | -2 [-8, +4] | -10 [-16, -5] | +10 [+5, +16] | 107, +3 [-1, +8], 4 / 1 | 10 / 0 / 0 | 5 / 5 / 0 / 0 | +2 (98) |
| x.md.m0.25 | 65 / 59 / 2 / 66 | 5.0 | +4 [-1, +10] | +7 [+2, +12] | -7 [-12, -2] | 117, +1 [-3, +5], 3 / 2 | 0 / 7 / 0 | 0 / 0 / 3 / 4 | +1 (109) |
| x.rand.p0.25 | 63 / 53 / 2 / 74 | 4.9 | +2 [-3, +7] | -1 [-4, +2] | +1 [-2, +4] | 115, +3 [-1, +8], 4 / 1 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +1 (106) |
| x.rand.m0.25 | 60 / 56 / 2 / 74 | 4.6 | -1 [-5, +2] | -1 [-3, +0] | +1 [+0, +3] | 116, +0 [-3, +3], 1 / 1 | 1 / 0 / 0 | 1 / 0 / 0 / 0 | +1 (113) |

- C toward caution (x.w.p0.25): dAns -44 (-22.9%), dX +45 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -43 [-55, -32]; vs x.rand.m0.25 -43 [-54, -32] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 73 questions answered both times (need at least 59): right-rate change +1.4 pp [-3.0, +6.2], W>R 2 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 58.9% (answers dropped: 19 right, 25 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +33 (17.2%), dX -34 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +34 [+24, +45]; vs x.rand.m0.25 +34 [+24, +45] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 116 questions answered both times (need at least 59): right-rate change -0.9 pp [-4.0, +1.8], W>R 1 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 50.0% (answers dropped: 0 right, 1 wrong; added: 15 right, 19 wrong)
KEY family=C judge=allenai layer=20 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+33 Aplus=+44 Aminus=+33 EXPLORATORY strength=0.25

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 20

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 38 / 34 / 1 / 119 | 4.0 | -16 [-24, -8] | -43 [-54, -32] | +45 [+34, +57] | 72, +0 [-3, +3], 1 / 1 | 43 / 0 / 0 | 16 / 27 / 0 / 0 | +0 (67) |
| x.w.m0.25 | 67 / 80 / 6 / 39 | 4.9 | +13 [+6, +21] | +32 [+22, +43] | -35 [-46, -25] | 114, +0 [-3, +3], 1 / 1 | 1 / 33 / 1 | 1 / 0 / 14 / 19 | +0 (110) |
| x.md.p0.25 | 51 / 53 / 4 / 84 | 4.6 | -3 [-8, +2] | -11 [-18, -4] | +10 [+5, +16] | 103, +1 [-2, +4], 2 / 1 | 12 / 1 / 3 | 4 / 8 / 0 / 1 | +0 (96) |
| x.md.m0.25 | 59 / 63 / 3 / 67 | 5.0 | +5 [+0, +10] | +7 [+2, +12] | -7 [-12, -2] | 115, +1 [-2, +5], 2 / 1 | 0 / 7 / 0 | 0 / 0 / 4 / 3 | +2 (107) |
| x.rand.p0.25 | 54 / 58 / 5 / 75 | 4.9 | +0 [-4, +4] | -3 [-8, +1] | +1 [-2, +4] | 111, +2 [+0, +5], 2 / 0 | 4 / 1 / 2 | 2 / 2 / 0 / 1 | +0 (102) |
| x.rand.m0.25 | 55 / 60 / 3 / 74 | 4.6 | +1 [+0, +3] | +0 [+0, +0] | +0 [-3, +3] | 115, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (112) |

- C toward caution (x.w.p0.25): dAns -43 (-22.4%), dX +45 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -40 [-52, -29]; vs x.rand.m0.25 -43 [-54, -32] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 72 questions answered both times (need at least 58): right-rate change +0.0 pp [-4.0, +4.1], W>R 1 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 52.8% (answers dropped: 16 right, 27 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +32 (16.7%), dX -35 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +35 [+25, +46]; vs x.rand.m0.25 +32 [+22, +43] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change +0.0 pp [-2.6, +2.6], W>R 1 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 47.0% -> 45.6% (answers dropped: 1 right, 0 wrong; added: 14 right, 19 wrong)
KEY family=C judge=pro layer=20 A=TWO-WAY B=ONE-WAY(x.w.m0.25) A_effect_only=TWO-WAY T=+32 Aplus=+43 Aminus=+32 EXPLORATORY strength=0.25

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 20

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 15:36:39 [qwen3-8b C tables gpunone e_on=E1] tables done ===

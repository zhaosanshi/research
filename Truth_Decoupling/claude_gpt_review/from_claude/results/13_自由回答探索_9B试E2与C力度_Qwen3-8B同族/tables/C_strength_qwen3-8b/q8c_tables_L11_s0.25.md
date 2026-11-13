TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.25 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 58 / 48 / 2 / 84 | 4.3 | -3 [-9, +3] | -11 [-18, -5] | +11 [+5, +18] | 106, +2 [-3, +7], 4 / 2 | 11 / 0 / 0 | 5 / 6 / 0 / 0 | +2 (103) |
| x.w.m0.25 | 64 / 60 / 3 / 65 | 4.7 | +3 [-1, +8] | +7 [+2, +13] | -8 [-14, -3] | 116, -1 [-3, +0], 0 / 1 | 1 / 8 / 1 | 0 / 1 / 4 / 4 | +0 (110) |
| x.md.p0.25 | 63 / 54 / 2 / 73 | 4.6 | +2 [-2, +6] | +0 [-3, +3] | +0 [-3, +3] | 116, +3 [+0, +7], 3 / 0 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +3 (113) |
| x.md.m0.25 | 60 / 57 / 2 / 73 | 4.7 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 117, -1 [-5, +2], 1 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (113) |
| x.rand.p0.25 | 60 / 58 / 2 / 72 | 4.6 | -1 [-3, +0] | +1 [+0, +3] | -1 [-3, +0] | 117, -1 [-3, +0], 0 / 1 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | -1 (117) |
| x.rand.m0.25 | 62 / 55 / 2 / 73 | 4.7 | +1 [-2, +4] | +0 [+0, +0] | +0 [+0, +0] | 117, +1 [-2, +4], 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (116) |

- C toward caution (x.w.p0.25): dAns -11 (-5.7%), dX +11 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -12 [-19, -6]; vs x.rand.m0.25 -11 [-18, -5] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 106 questions answered both times (need at least 59): right-rate change +1.9 pp [-2.7, +6.5], W>R 4 / R>W 2, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 54.7% (answers dropped: 5 right, 6 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +7 (3.6%), dX -8 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +6 [+1, +12]; vs x.rand.m0.25 +7 [+2, +13] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 116 questions answered both times (need at least 59): right-rate change -0.9 pp [-2.8, +0.0], W>R 0 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 52.1% -> 51.6% (answers dropped: 0 right, 1 wrong; added: 4 right, 4 wrong)
KEY family=C judge=allenai layer=11 A=TWO-WAY B=ONE-WAY(x.w.m0.25) A_effect_only=TWO-WAY T=+7 Aplus=+11 Aminus=+7 EXPLORATORY strength=0.25

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 52 / 51 / 5 / 84 | 4.3 | -2 [-7, +3] | -12 [-19, -6] | +10 [+4, +17] | 103, +1 [-2, +4], 2 / 1 | 12 / 0 / 2 | 3 / 9 / 0 / 0 | +1 (100) |
| x.w.m0.25 | 57 / 65 / 4 / 66 | 4.7 | +3 [-1, +8] | +7 [+2, +13] | -8 [-14, -3] | 114, +0 [+0, +0], 0 / 0 | 1 / 8 / 1 | 1 / 0 / 4 / 4 | +0 (108) |
| x.md.p0.25 | 54 / 60 / 5 / 73 | 4.6 | +0 [-3, +3] | -1 [-5, +2] | -1 [-5, +2] | 113, +1 [+0, +3], 1 / 0 | 2 / 1 / 1 | 1 / 1 / 0 / 1 | +1 (110) |
| x.md.m0.25 | 53 / 62 / 3 / 74 | 4.7 | -1 [-5, +2] | +0 [+0, +0] | +0 [+0, +0] | 115, -1 [-5, +2], 1 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (111) |
| x.rand.p0.25 | 53 / 62 / 4 / 73 | 4.6 | -1 [-3, +0] | +0 [-3, +3] | -1 [-3, +0] | 114, +0 [+0, +0], 0 / 0 | 1 / 1 / 1 | 1 / 0 / 0 / 1 | +0 (114) |
| x.rand.m0.25 | 55 / 60 / 3 / 74 | 4.7 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 115, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (114) |

- C toward caution (x.w.p0.25): dAns -12 (-6.2%), dX +10 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -12 [-19, -5]; vs x.rand.m0.25 -12 [-19, -6] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 103 questions answered both times (need at least 58): right-rate change +1.0 pp [-2.1, +4.4], W>R 2 / R>W 1, answered -> hedge 2 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 50.5% (answers dropped: 3 right, 9 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +7 (3.6%), dX -8 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +7 [+2, +12]; vs x.rand.m0.25 +7 [+2, +13] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change +0.0 pp [-2.6, +2.6], W>R 0 / R>W 0, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 47.0% -> 46.7% (answers dropped: 1 right, 0 wrong; added: 4 right, 4 wrong)
KEY family=C judge=pro layer=11 A=TWO-WAY B=ONE-WAY(x.w.m0.25) A_effect_only=TWO-WAY T=+7 Aplus=+12 Aminus=+7 EXPLORATORY strength=0.25

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 15:56:11 [qwen3-8b C tables gpunone e_on=E1] tables done ===

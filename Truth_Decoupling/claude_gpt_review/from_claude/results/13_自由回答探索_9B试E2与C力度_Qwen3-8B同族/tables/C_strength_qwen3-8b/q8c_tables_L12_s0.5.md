TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.5 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 49 / 43 / 2 / 98 | 4.2 | -12 [-21, -4] | -25 [-34, -16] | +25 [+16, +34] | 92, +1 [-4, +6], 4 / 3 | 25 / 0 / 0 | 13 / 12 / 0 / 0 | +2 (86) |
| x.w.m0.5 | 79 / 69 / 3 / 41 | 5.0 | +18 [+10, +27] | +31 [+21, +41] | -32 [-42, -22] | 116, +1 [-2, +5], 2 / 1 | 1 / 32 / 1 | 0 / 1 / 17 / 15 | +2 (106) |
| x.md.p0.5 | 62 / 55 / 2 / 73 | 4.5 | +1 [-3, +5] | +0 [-3, +3] | +0 [-3, +3] | 116, +2 [-2, +6], 3 / 1 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +1 (111) |
| x.md.m0.5 | 59 / 59 / 2 / 72 | 4.6 | -2 [-7, +3] | +1 [+0, +3] | -1 [-3, +0] | 117, -3 [-8, +1], 1 / 4 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | -2 (110) |
| x.rand.p0.5 | 61 / 56 / 2 / 73 | 4.6 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (115) |
| x.rand.m0.5 | 61 / 55 / 2 / 74 | 4.7 | +0 [-4, +4] | -1 [-5, +2] | +1 [-2, +5] | 115, +1 [-2, +4], 2 / 1 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +1 (109) |

- C toward caution (x.w.p0.5): dAns -25 (-13.0%), dX +25 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -25 [-34, -16]; vs x.rand.m0.5 -24 [-33, -16] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 92 questions answered both times (need at least 59): right-rate change +1.1 pp [-4.4, +6.8], W>R 4 / R>W 3, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 53.3% (answers dropped: 13 right, 12 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +31 (16.1%), dX -32 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +31 [+21, +41]; vs x.rand.m0.5 +32 [+22, +43] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 116 questions answered both times (need at least 59): right-rate change +0.9 pp [-1.8, +3.9], W>R 2 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 53.4% (answers dropped: 0 right, 1 wrong; added: 17 right, 15 wrong)
KEY family=C judge=allenai layer=12 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+25 Aplus=+25 Aminus=+31 EXPLORATORY strength=0.5

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 44 / 46 / 4 / 98 | 4.2 | -10 [-18, -3] | -25 [-35, -16] | +24 [+15, +34] | 90, +1 [-2, +4], 2 / 1 | 25 / 0 / 2 | 11 / 14 / 0 / 0 | +1 (84) |
| x.w.m0.5 | 68 / 75 / 7 / 42 | 5.0 | +14 [+7, +22] | +28 [+18, +38] | -32 [-42, -22] | 114, +2 [+0, +5], 2 / 0 | 1 / 29 / 1 | 1 / 0 / 13 / 16 | +2 (104) |
| x.md.p0.5 | 51 / 61 / 7 / 73 | 4.5 | -3 [-8, +1] | -3 [-8, +1] | -1 [-5, +2] | 111, +0 [-3, +3], 1 / 1 | 4 / 1 / 3 | 3 / 1 / 0 / 1 | +0 (106) |
| x.md.m0.5 | 55 / 62 / 2 / 73 | 4.6 | +1 [-2, +5] | +2 [+0, +5] | -1 [-3, +0] | 115, +0 [-3, +3], 1 / 1 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +0 (109) |
| x.rand.p0.5 | 55 / 60 / 3 / 74 | 4.6 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 115, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (113) |
| x.rand.m0.5 | 52 / 61 / 5 / 74 | 4.7 | -2 [-6, +2] | -2 [-6, +2] | +0 [-4, +4] | 112, +0 [-3, +3], 1 / 1 | 3 / 1 / 1 | 2 / 1 / 0 / 1 | +0 (106) |

- C toward caution (x.w.p0.5): dAns -25 (-13.0%), dX +24 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -25 [-35, -16]; vs x.rand.m0.5 -23 [-33, -14] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 90 questions answered both times (need at least 58): right-rate change +1.1 pp [-2.4, +5.0], W>R 2 / R>W 1, answered -> hedge 2 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 48.9% (answers dropped: 11 right, 14 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +28 (14.6%), dX -32 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +28 [+18, +38]; vs x.rand.m0.5 +30 [+20, +40] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change +1.8 pp [+0.0, +4.5], W>R 2 / R>W 0, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 47.6% (answers dropped: 1 right, 0 wrong; added: 13 right, 16 wrong)
KEY family=C judge=pro layer=12 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+25 Aplus=+25 Aminus=+28 EXPLORATORY strength=0.5

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 15:49:53 [qwen3-8b C tables gpunone e_on=E1] tables done ===

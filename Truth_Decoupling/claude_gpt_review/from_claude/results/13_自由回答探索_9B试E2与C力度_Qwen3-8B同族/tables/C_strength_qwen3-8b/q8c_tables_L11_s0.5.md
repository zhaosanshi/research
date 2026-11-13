TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.5 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 51 / 42 / 2 / 97 | 4.1 | -10 [-18, -2] | -24 [-33, -15] | +24 [+15, +33] | 93, +1 [-4, +6], 4 / 3 | 24 / 0 / 0 | 11 / 13 / 0 / 0 | +0 (90) |
| x.w.m0.5 | 75 / 67 / 3 / 47 | 4.7 | +14 [+6, +22] | +25 [+16, +35] | -26 [-35, -17] | 116, +0 [-4, +4], 2 / 2 | 1 / 26 / 1 | 0 / 1 / 14 / 12 | +0 (109) |
| x.md.p0.5 | 63 / 53 / 2 / 74 | 4.5 | +2 [-2, +6] | -1 [-5, +2] | +1 [-2, +5] | 115, +3 [+0, +7], 3 / 0 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +2 (110) |
| x.md.m0.5 | 59 / 58 / 2 / 73 | 4.7 | -2 [-7, +3] | +0 [-3, +3] | +0 [-3, +3] | 116, -1 [-5, +3], 2 / 3 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +0 (109) |
| x.rand.p0.5 | 63 / 54 / 2 / 73 | 4.7 | +2 [+0, +5] | +0 [-3, +3] | +0 [-3, +3] | 116, +2 [+0, +5], 2 / 0 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | +2 (109) |
| x.rand.m0.5 | 61 / 56 / 2 / 73 | 4.7 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (111) |

- C toward caution (x.w.p0.5): dAns -24 (-12.5%), dX +24 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -24 [-33, -15]; vs x.rand.m0.5 -24 [-33, -15] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 93 questions answered both times (need at least 59): right-rate change +1.1 pp [-4.4, +6.7], W>R 4 / R>W 3, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 54.8% (answers dropped: 11 right, 13 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +25 (13.0%), dX -26 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +25 [+16, +35]; vs x.rand.m0.5 +25 [+16, +35] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 116 questions answered both times (need at least 59): right-rate change +0.0 pp [-3.4, +3.4], W>R 2 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 52.8% (answers dropped: 0 right, 1 wrong; added: 14 right, 12 wrong)
KEY family=C judge=allenai layer=11 A=TWO-WAY B=not_met A_effect_only=TWO-WAY T=+24 Aplus=+24 Aminus=+25 EXPLORATORY strength=0.5

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 11

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 46 / 46 / 3 / 97 | 4.1 | -8 [-15, -2] | -23 [-32, -14] | +23 [+14, +32] | 92, +1 [-2, +4], 2 / 1 | 23 / 0 / 1 | 9 / 14 / 0 / 0 | +1 (89) |
| x.w.m0.5 | 66 / 74 / 4 / 48 | 4.7 | +12 [+5, +20] | +25 [+16, +35] | -26 [-35, -17] | 114, +0 [-3, +3], 1 / 1 | 1 / 26 / 1 | 1 / 0 / 13 / 13 | -1 (107) |
| x.md.p0.5 | 55 / 59 / 4 / 74 | 4.5 | +1 [-2, +5] | -1 [-5, +2] | +0 [-4, +4] | 113, +2 [+0, +5], 2 / 0 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +1 (108) |
| x.md.m0.5 | 53 / 64 / 1 / 74 | 4.7 | -1 [-5, +2] | +2 [+0, +5] | +0 [-3, +3] | 115, -1 [-5, +2], 1 / 2 | 0 / 2 / 0 | 0 / 0 / 0 / 2 | -1 (109) |
| x.rand.p0.5 | 54 / 60 / 5 / 73 | 4.7 | +0 [-4, +4] | -1 [-5, +2] | -1 [-4, +2] | 113, +1 [-2, +5], 2 / 1 | 2 / 1 / 1 | 1 / 1 / 0 / 1 | +1 (106) |
| x.rand.m0.5 | 55 / 61 / 2 / 74 | 4.7 | +1 [+0, +3] | +1 [+0, +3] | +0 [+0, +0] | 115, +1 [+0, +3], 1 / 0 | 0 / 1 / 0 | 0 / 0 / 0 / 1 | +1 (110) |

- C toward caution (x.w.p0.5): dAns -23 (-12.0%), dX +23 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -22 [-31, -13]; vs x.rand.m0.5 -24 [-33, -15] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 92 questions answered both times (need at least 58): right-rate change +1.1 pp [-2.3, +4.9], W>R 2 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 50.0% (answers dropped: 9 right, 14 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +25 (13.0%), dX -26 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +26 [+17, +35]; vs x.rand.m0.5 +24 [+14, +34] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change +0.0 pp [-2.6, +2.6], W>R 1 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 47.0% -> 47.1% (answers dropped: 1 right, 0 wrong; added: 13 right, 13 wrong)
KEY family=C judge=pro layer=11 A=TWO-WAY B=ONE-WAY(x.w.m0.5) A_effect_only=TWO-WAY T=+23 Aplus=+23 Aminus=+25 EXPLORATORY strength=0.5

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 11

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 15:56:07 [qwen3-8b C tables gpunone e_on=E1] tables done ===

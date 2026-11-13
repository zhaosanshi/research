TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.5 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 20 / 13 / 0 / 159 | 3.2 | -41 [-52, -30] | -84 [-97, -71] | +86 [+73, +100] | 33, +0 [+0, +0], 0 / 0 | 84 / 0 / 0 | 41 / 43 / 0 / 0 | +0 (33) |
| x.w.m0.5 | 88 / 86 / 3 / 15 | 5.3 | +27 [+16, +38] | +57 [+44, +70] | -58 [-70, -46] | 116, -3 [-8, +1], 1 / 4 | 1 / 58 / 1 | 0 / 1 / 30 / 28 | +0 (107) |
| x.md.p0.5 | 51 / 42 / 2 / 97 | 4.3 | -10 [-17, -3] | -24 [-33, -16] | +24 [+16, +33] | 93, +0 [-4, +4], 2 / 2 | 24 / 0 / 0 | 10 / 14 / 0 / 0 | +1 (87) |
| x.md.m0.5 | 73 / 68 / 2 / 49 | 5.1 | +12 [+3, +21] | +24 [+15, +33] | -24 [-33, -15] | 117, +0 [-6, +6], 5 / 5 | 0 / 24 / 0 | 0 / 0 / 12 / 12 | +2 (105) |
| x.rand.p0.5 | 64 / 52 / 2 / 74 | 5.0 | +3 [-2, +8] | -1 [-4, +2] | +1 [-2, +4] | 115, +4 [+0, +9], 5 / 1 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +2 (101) |
| x.rand.m0.5 | 60 / 56 / 2 / 74 | 4.5 | -1 [-5, +2] | -1 [-4, +2] | +1 [-2, +4] | 115, +0 [-3, +3], 1 / 1 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +1 (111) |

- C toward caution (x.w.p0.5): dAns -84 (-43.8%), dX +86 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -83 [-97, -70]; vs x.rand.m0.5 -83 [-96, -70] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 33 questions answered both times (need at least 59): right-rate change +0.0 pp [-8.7, +8.7], W>R 0 / R>W 0, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 60.6% (answers dropped: 41 right, 43 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +57 (29.7%), dX -58 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +58 [+45, +71]; vs x.rand.m0.5 +58 [+45, +71] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 116 questions answered both times (need at least 59): right-rate change -2.6 pp [-6.6, +0.9], W>R 1 / R>W 4, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 50.6% (answers dropped: 0 right, 1 wrong; added: 30 right, 28 wrong)
KEY family=C judge=allenai layer=20 A=ONE-WAY(x.w.m0.5) B=not_met A_effect_only=TWO-WAY T=+57 Aplus=+84 Aminus=+57 EXPLORATORY strength=0.5

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 20

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.5, strength 0.5 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 20

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.5 | 20 / 13 / 0 / 159 | 3.2 | -34 [-45, -24] | -82 [-95, -69] | +85 [+72, +99] | 33, +0 [+0, +0], 0 / 0 | 82 / 0 / 0 | 34 / 48 / 0 / 0 | +0 (33) |
| x.w.m0.5 | 79 / 91 / 7 / 15 | 5.3 | +25 [+15, +35] | +55 [+43, +68] | -59 [-71, -47] | 114, +0 [-3, +3], 1 / 1 | 1 / 56 / 1 | 1 / 0 / 26 / 30 | +1 (106) |
| x.md.p0.5 | 45 / 45 / 4 / 98 | 4.3 | -9 [-16, -3] | -25 [-35, -16] | +24 [+16, +33] | 89, +1 [+0, +3], 1 / 0 | 26 / 1 / 3 | 10 / 16 / 0 / 1 | +1 (84) |
| x.md.m0.5 | 65 / 73 / 4 / 50 | 5.1 | +11 [+3, +20] | +23 [+15, +32] | -24 [-33, -15] | 115, +0 [-6, +6], 4 / 4 | 0 / 23 / 0 | 0 / 0 / 11 / 12 | +2 (103) |
| x.rand.p0.5 | 54 / 58 / 5 / 75 | 5.0 | +0 [-5, +5] | -3 [-8, +1] | +1 [-2, +4] | 111, +2 [-2, +6], 3 / 1 | 4 / 1 / 2 | 2 / 2 / 0 / 1 | +0 (97) |
| x.rand.m0.5 | 55 / 60 / 3 / 74 | 4.5 | +1 [+0, +3] | +0 [-3, +3] | +0 [-4, +4] | 114, +1 [+0, +3], 1 / 0 | 1 / 1 / 0 | 0 / 1 / 0 / 1 | +1 (110) |

- C toward caution (x.w.p0.5): dAns -82 (-42.7%), dX +85 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 -79 [-93, -66]; vs x.rand.m0.5 -82 [-95, -69] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 33 questions answered both times (need at least 58): right-rate change +0.0 pp [-8.7, +8.7], W>R 0 / R>W 0, answered -> hedge 0 | A: point inside: False | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 60.6% (answers dropped: 34 right, 48 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +55 (28.6%), dX -59 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.5 +58 [+46, +70]; vs x.rand.m0.5 +55 [+43, +68] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change +0.0 pp [-2.6, +2.6], W>R 1 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 47.0% -> 46.5% (answers dropped: 1 right, 0 wrong; added: 26 right, 30 wrong)
KEY family=C judge=pro layer=20 A=ONE-WAY(x.w.m0.5) B=ONE-WAY(x.w.m0.5) A_effect_only=TWO-WAY T=+55 Aplus=+82 Aminus=+55 EXPLORATORY strength=0.5

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 20

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 15:36:36 [qwen3-8b C tables gpunone e_on=E1] tables done ===

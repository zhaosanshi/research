TABLES run=/workspace/Truth_Decoupling/results_e44_factorial_das/llama3.1-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.25 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0

## layer 31

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 74 / 83 / 2 / 33 | 3.9 | -4 [-9, +0] | -9 [-16, -3] | +10 [+5, +16] | 156, +1 [+0, +3], 1 / 0 | 10 / 1 / 0 | 5 / 5 / 0 / 1 | +1 (154) |
| x.w.m0.25 | 82 / 87 / 4 / 19 | 4.5 | +4 [+0, +9] | +3 [+0, +7] | -4 [-8, -1] | 166, +2 [-2, +6], 3 / 1 | 0 / 3 / 0 | 0 / 0 / 2 / 1 | +1 (159) |
| x.md.p0.25 | 74 / 86 / 2 / 30 | 4.1 | -4 [-9, +0] | -6 [-12, -1] | +7 [+2, +12] | 159, -2 [-6, +2], 1 / 3 | 7 / 1 / 0 | 2 / 5 / 0 / 1 | -3 (154) |
| x.md.m0.25 | 80 / 88 / 4 / 20 | 4.3 | +2 [-2, +6] | +2 [+0, +5] | -3 [-7, +0] | 166, +1 [-2, +5], 2 / 1 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +1 (158) |
| x.rand.p0.25 | 76 / 90 / 3 / 23 | 4.1 | -2 [-5, +0] | +0 [+0, +0] | +0 [+0, +0] | 166, -2 [-5, +0], 0 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -2 (164) |
| x.rand.m0.25 | 78 / 88 / 3 / 23 | 4.0 | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | 166, +0 [+0, +0], 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (165) |

- C toward caution (x.w.p0.25): dAns -9 (-4.7%), dX +10 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -9 [-16, -3]; vs x.rand.m0.25 -9 [-16, -3] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 156 questions answered both times (need at least 83): right-rate change +0.6 pp [+0.0, +2.0], W>R 1 / R>W 0, answered -> hedge 0 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 47.0% -> 47.1% (answers dropped: 5 right, 5 wrong; added: 0 right, 1 wrong)
- C toward confident (x.w.m0.25): dAns +3 (1.6%), dX -4 | A: beyond the tolerance, abstentions move the same way: False | B: and both intervals on that side: False
    random controls: vs x.rand.p0.25 +3 [+0, +7]; vs x.rand.m0.25 +3 [+0, +7] | A: larger than each (point): True | B: each interval on that side: False
    retention on the 166 questions answered both times (need at least 83): right-rate change +1.2 pp [-1.2, +3.7], W>R 3 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 48.5% (answers dropped: 0 right, 0 wrong; added: 2 right, 1 wrong)
KEY family=C judge=allenai layer=31 A=ONE-WAY(x.w.p0.25) B=ONE-WAY(x.w.p0.25) A_effect_only=ONE-WAY(x.w.p0.25) T=+3 Aplus=+9 Aminus=+3 EXPLORATORY strength=0.25

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 31

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0

## layer 31

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 61 / 91 / 7 / 33 | 3.9 | -3 [-7, +0] | -10 [-16, -4] | +9 [+4, +15] | 152, -1 [-3, +0], 0 / 1 | 10 / 0 / 1 | 2 / 8 / 0 / 0 | +0 (149) |
| x.w.m0.25 | 66 / 98 / 7 / 21 | 4.5 | +2 [-2, +6] | +2 [+0, +5] | -3 [-7, +0] | 162, +1 [-2, +5], 2 / 1 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +2 (154) |
| x.md.p0.25 | 63 / 91 / 7 / 31 | 4.1 | -1 [-5, +2] | -8 [-14, -3] | +7 [+2, +12] | 154, +1 [+0, +3], 1 / 0 | 8 / 0 / 1 | 2 / 6 / 0 / 0 | +1 (148) |
| x.md.m0.25 | 66 / 98 / 7 / 21 | 4.3 | +2 [-2, +6] | +2 [+0, +5] | -3 [-7, +0] | 162, +1 [-2, +5], 2 / 1 | 0 / 2 / 0 | 0 / 0 / 1 / 1 | +2 (154) |
| x.rand.p0.25 | 64 / 99 / 5 / 24 | 4.1 | +0 [-3, +3] | +1 [+0, +3] | +0 [+0, +0] | 162, -1 [-3, +0], 0 / 1 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (159) |
| x.rand.m0.25 | 65 / 98 / 5 / 24 | 4.0 | +1 [+0, +3] | +1 [+0, +3] | +0 [+0, +0] | 162, +0 [+0, +0], 0 / 0 | 0 / 1 / 0 | 0 / 0 / 1 / 0 | +0 (161) |

- C toward caution (x.w.p0.25): dAns -10 (-5.2%), dX +9 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -11 [-18, -5]; vs x.rand.m0.25 -11 [-18, -5] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 152 questions answered both times (need at least 81): right-rate change -0.7 pp [-2.1, +0.0], W>R 0 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 39.5% -> 40.1% (answers dropped: 2 right, 8 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +2 (1.0%), dX -3 | A: beyond the tolerance, abstentions move the same way: False | B: and both intervals on that side: False
    random controls: vs x.rand.p0.25 +1 [+0, +3]; vs x.rand.m0.25 +1 [+0, +3] | A: larger than each (point): True | B: each interval on that side: False
    retention on the 162 questions answered both times (need at least 81): right-rate change +0.6 pp [-1.3, +2.9], W>R 2 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 39.5% -> 40.2% (answers dropped: 0 right, 0 wrong; added: 1 right, 1 wrong)
KEY family=C judge=pro layer=31 A=ONE-WAY(x.w.p0.25) B=ONE-WAY(x.w.p0.25) A_effect_only=ONE-WAY(x.w.p0.25) T=+2 Aplus=+10 Aminus=+2 EXPLORATORY strength=0.25

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 31

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 13:05:07 [llama3.1-8b run=llama3.1-8b C tables gpunone e_on=E1] tables done ===

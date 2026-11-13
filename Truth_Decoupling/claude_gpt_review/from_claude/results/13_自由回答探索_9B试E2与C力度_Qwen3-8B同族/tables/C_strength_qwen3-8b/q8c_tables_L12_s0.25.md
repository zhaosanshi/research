TABLES run=/workspace/Truth_Decoupling/results_newmodels_ff_20261112/e44/qwen3-8b e_on=E1 work=judge_tune_proS_C_L<layer> strength=0.25 judges=allenai pro
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 61 / 56 / 2 / 73; rule abstentions 73; mean words 4.8

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 57 / 48 / 2 / 85 | 4.4 | -4 [-10, +2] | -12 [-19, -6] | +12 [+6, +19] | 105, +1 [-3, +5], 3 / 2 | 12 / 0 / 0 | 5 / 7 / 0 / 0 | +1 (102) |
| x.w.m0.25 | 64 / 62 / 2 / 64 | 4.7 | +3 [-1, +8] | +9 [+4, +15] | -9 [-15, -4] | 117, +0 [-3, +3], 1 / 1 | 0 / 9 / 0 | 0 / 0 / 3 / 6 | +1 (110) |
| x.md.p0.25 | 62 / 55 / 2 / 73 | 4.5 | +1 [-3, +5] | +0 [-3, +3] | +0 [-3, +3] | 116, +2 [-2, +6], 3 / 1 | 1 / 1 / 0 | 1 / 0 / 0 / 1 | +2 (112) |
| x.md.m0.25 | 61 / 56 / 2 / 73 | 4.6 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (112) |
| x.rand.p0.25 | 61 / 56 / 2 / 73 | 4.6 | +0 [-4, +4] | +0 [+0, +0] | +0 [+0, +0] | 117, +0 [-4, +4], 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | -1 (116) |
| x.rand.m0.25 | 60 / 56 / 2 / 74 | 4.7 | -1 [-5, +2] | -1 [-5, +2] | +1 [-2, +5] | 115, +0 [-3, +3], 1 / 1 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +0 (113) |

- C toward caution (x.w.p0.25): dAns -12 (-6.2%), dX +12 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -12 [-19, -6]; vs x.rand.m0.25 -11 [-18, -5] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 105 questions answered both times (need at least 59): right-rate change +1.0 pp [-3.1, +5.1], W>R 3 / R>W 2, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 52.1% -> 54.3% (answers dropped: 5 right, 7 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +9 (4.7%), dX -9 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +9 [+4, +15]; vs x.rand.m0.25 +10 [+4, +16] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 117 questions answered both times (need at least 59): right-rate change +0.0 pp [-2.5, +2.5], W>R 1 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 52.1% -> 50.8% (answers dropped: 0 right, 0 wrong; added: 3 right, 6 wrong)
KEY family=C judge=allenai layer=12 A=TWO-WAY B=ONE-WAY(x.w.m0.25) A_effect_only=TWO-WAY T=+9 Aplus=+12 Aminus=+9 EXPLORATORY strength=0.25

# C arms on other prompts, judge allenai (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p0.25, strength 0.25 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.

baseline x: right / wrong / hedge-or-uninformative / abstained = 54 / 61 / 3 / 74; rule abstentions 73; mean words 4.8

## layer 12

| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |
|---|---|---|---|---|---|---|---|---|---|
| x.w.p0.25 | 52 / 51 / 4 / 85 | 4.4 | -2 [-7, +3] | -12 [-19, -6] | +11 [+4, +18] | 103, +1 [-2, +4], 2 / 1 | 12 / 0 / 1 | 3 / 9 / 0 / 0 | +1 (100) |
| x.w.m0.25 | 57 / 66 / 4 / 65 | 4.7 | +3 [-1, +8] | +8 [+2, +14] | -9 [-15, -4] | 114, +1 [+0, +3], 1 / 0 | 1 / 9 / 1 | 1 / 0 / 3 / 6 | +1 (107) |
| x.md.p0.25 | 51 / 61 / 7 / 73 | 4.5 | -3 [-7, +1] | -3 [-8, +1] | -1 [-5, +2] | 111, +0 [-3, +3], 1 / 1 | 4 / 1 / 3 | 3 / 1 / 0 / 1 | +0 (107) |
| x.md.m0.25 | 54 / 61 / 3 / 74 | 4.6 | +0 [-3, +3] | +0 [+0, +0] | +0 [+0, +0] | 115, +0 [-3, +3], 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +0 (110) |
| x.rand.p0.25 | 55 / 60 / 3 / 74 | 4.6 | +1 [+0, +3] | +0 [+0, +0] | +0 [+0, +0] | 115, +1 [+0, +3], 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | +1 (114) |
| x.rand.m0.25 | 53 / 61 / 4 / 74 | 4.7 | -1 [-5, +2] | -1 [-5, +2] | +0 [-4, +4] | 113, +0 [-3, +3], 1 / 1 | 2 / 1 / 0 | 1 / 1 / 0 / 1 | +0 (111) |

- C toward caution (x.w.p0.25): dAns -12 (-6.2%), dX +11 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 -12 [-19, -6]; vs x.rand.m0.25 -11 [-18, -5] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 103 questions answered both times (need at least 58): right-rate change +1.0 pp [-2.1, +4.3], W>R 2 / R>W 1, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 50.5% (answers dropped: 3 right, 9 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.25): dAns +8 (4.2%), dX -9 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    random controls: vs x.rand.p0.25 +8 [+2, +14]; vs x.rand.m0.25 +9 [+3, +16] | A: larger than each (point): True | B: each interval on that side: True
    retention on the 114 questions answered both times (need at least 58): right-rate change +0.9 pp [+0.0, +2.8], W>R 1 / R>W 0, answered -> hedge 1 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 47.0% -> 46.3% (answers dropped: 1 right, 0 wrong; added: 3 right, 6 wrong)
KEY family=C judge=pro layer=12 A=TWO-WAY B=ONE-WAY(x.w.m0.25) A_effect_only=TWO-WAY T=+8 Aplus=+12 Aminus=+8 EXPLORATORY strength=0.25

# C arms on other prompts, judge pro (each arm against its own base prompt, no write)

## layer 12

| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |
|---|---|---|---|---|---|---|---|---|

=== 2026-11-13 15:49:57 [qwen3-8b C tables gpunone e_on=E1] tables done ===

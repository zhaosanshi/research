# E conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
baseline x: right / wrong / hedge-or-uninformative / abstained = 78 / 88 / 3 / 23; rule abstentions 23; mean words 4.0
| x.w.p2 | 102 / 49 / 0 / 41 | 7.7 | +24 [+12, +36] | -15 [-25, -6] | +18 [+9, +27] | 146, +27 [+16, +38], 31 / 4 | 20 / 5 / 0 | 6 / 14 / 3 / 2 | +11 (87) |
- F, complete-sentence background: arm x_F.w.p1 not judged
- main effect: dR +24 (12.5% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: none judged; MISSING x.rand.p2, x.rand.m2 | A: above each (point): None | B: each interval above 0: None
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: None | B: dR interval above 0 and answered-rate interval inside: None
- retention on x: answered-rate change -7.8 pp [-13.0, -3.1]; moved out / in 20 / 5 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +11 on 87 questions; mean words 4.0 -> 7.7
KEY family=E judge=allenai layer=10 A=not_met B=not_met dR=+24 dR_F=NA dAns=-15 dAns_F=NA rand_margin=NA EXPLORATORY base=x main=x.w.p2 strength=2
# E conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
EXPLORATORY: base prompt x, main arm x.w.p2, strength 2 — outside the frozen criteria (which use base x, arm x.w.p1, strength 1).
baseline x: right / wrong / hedge-or-uninformative / abstained = 64 / 98 / 6 / 24; rule abstentions 23; mean words 4.0
| x.w.p2 | 85 / 67 / 1 / 39 | 7.7 | +21 [+9, +34] | -10 [-20, +0] | +15 [+7, +24] | 144, +15 [+4, +26], 24 / 9 | 18 / 8 / 0 | 2 / 16 / 8 / 0 | +8 (87) |
- F, complete-sentence background: arm x_F.w.p1 not judged
- main effect: dR +21 (10.9% of the questions) | A: beyond the tolerance: True | B: and interval above 0: True
- random controls: none judged; MISSING x.rand.p2, x.rand.m2 | A: above each (point): None | B: each interval above 0: None
- F | A: dR on x_F positive and its answered-rate change inside the tolerance: None | B: dR interval above 0 and answered-rate interval inside: None
- retention on x: answered-rate change -5.2 pp [-10.4, +0.0]; moved out / in 18 / 8 | A: point inside: False | B: whole interval inside: False
- diagnostic: G_same +8 on 87 questions; mean words 4.0 -> 7.7
KEY family=E judge=pro layer=10 A=not_met B=not_met dR=+21 dR_F=NA dAns=-10 dAns_F=NA rand_margin=NA EXPLORATORY base=x main=x.w.p2 strength=2
--- C strength 0.5, L31
# C conditions under judge allenai (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
- C toward caution (x.w.p0.5): dAns -15 (-7.8%), dX +17 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    retention on the 149 questions answered both times (need at least 83): right-rate change +0.0 pp [-2.0, +2.0], W>R 1 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 47.0% -> 47.7% (answers dropped: 7 right, 10 wrong; added: 1 right, 1 wrong)
- C toward confident (x.w.m0.5): dAns +9 (4.7%), dX -10 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    retention on the 166 questions answered both times (need at least 83): right-rate change +1.2 pp [-1.2, +3.7], W>R 3 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 47.0% -> 48.6% (answers dropped: 0 right, 0 wrong; added: 5 right, 4 wrong)
KEY family=C judge=allenai layer=31 A=incomplete B=incomplete A_effect_only=incomplete T=+9 Aplus=+15 Aminus=+9 EXPLORATORY strength=0.5
# C conditions under judge pro (192 questions; tolerance 3%; paired bootstrap 10000 resamples, seed 20261111; against baseline_x)
- C toward caution (x.w.p0.5): dAns -15 (-7.8%), dX +15 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    retention on the 147 questions answered both times (need at least 81): right-rate change -0.7 pp [-3.0, +1.4], W>R 1 / R>W 2, answered -> hedge 1 | A: point inside: True | B: whole interval inside: False
    reported, not a condition: right-rate among ALL answered 39.5% -> 40.1% (answers dropped: 4 right, 11 wrong; added: 0 right, 0 wrong)
- C toward confident (x.w.m0.5): dAns +6 (3.1%), dX -8 | A: beyond the tolerance, abstentions move the same way: True | B: and both intervals on that side: True
    retention on the 162 questions answered both times (need at least 81): right-rate change +0.6 pp [-1.3, +2.9], W>R 2 / R>W 1, answered -> hedge 0 | A: point inside: True | B: whole interval inside: True
    reported, not a condition: right-rate among ALL answered 39.5% -> 39.9% (answers dropped: 0 right, 0 wrong; added: 2 right, 4 wrong)
KEY family=C judge=pro layer=31 A=incomplete B=incomplete A_effect_only=incomplete T=+6 Aplus=+15 Aminus=+6 EXPLORATORY strength=0.5

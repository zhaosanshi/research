# Side branch MD (same as the main run, initialised at the mean-difference direction) — family E, judge allenai, 192 questions: right-count change (answered-count change), each arm against its own base prompt

No write: caut 82 / 82 / 1 / 27; caut_E1 90 / 68 / 2 / 32; conf 88 / 101 / 1 / 2; conf_E1 108 / 83 / 1 / 0; x 78 / 88 / 3 / 23  (right / wrong / hedge / abstained)

| layer | conf.wq.p1 | conf_E1.wq.m1 | conf.iso.p1 | caut.wq.p1 | caut_E1.wq.m1 | x.w.p1 | x.w.m1 | x.md.p1 | x.rand.p1 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | -3 (+0) | -6 (+0) | +0 (+0) | -1 (-3) | -1 (+1) | +3 (-3) | +2 (+1) | +1 (+0) | +0 (+1) |
| 10 | +15 (+1) | -21 (-1) | +0 (+0) | +6 (-8) | -15 (+6) | +9 (-10) | +1 (+7) | +4 (+0) | +0 (+3) |
| 16 | +2 (-1) | -14 (+1) | -1 (+0) | +5 (-4) | -5 (+6) | +0 (-7) | +6 (+5) | +3 (-7) | -1 (+0) |

Mean over the layers present: conf.wq.p1 +4.7 (+0.0); conf_E1.wq.m1 -13.7 (+0.0); conf.iso.p1 -0.3 (+0.0); caut.wq.p1 +3.3 (-5.0); caut_E1.wq.m1 -7.0 (+4.3); x.w.p1 +4.0 (-6.7); x.w.m1 +3.0 (+4.3); x.md.p1 +2.7 (-2.3); x.rand.p1 -0.3 (+1.3)

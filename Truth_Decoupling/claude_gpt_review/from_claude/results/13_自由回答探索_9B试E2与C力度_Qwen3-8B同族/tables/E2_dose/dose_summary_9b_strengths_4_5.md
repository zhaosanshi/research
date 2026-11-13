## judge pro: must-answer base (conf), change in right answers against conf without any write

Prompt = conf with the E2 sentence, no write. Direction = conf, no sentence, the learned E direction at the given strength (strength 1 = the frozen size of the write). Each cell: dR [95%] / dAns / G_same (n) / mean words. Random = the smaller margin of the direction over the two random directions (both signs) of the same strength, a point estimate (the per-layer cells tables carry the intervals); mean difference = dR of the driver's md arm of the same strength, which writes strength x |mean difference| (not size-matched to the learned write, strength x gap).

| model | layer | conf alone: words | prompt + E2 | direction 4 | direction 5 | margin over random (4 / 5) | mean difference (4 / 5) |
|---|---|---|---|---|---|---|---|
| 9B | 11 | 3.1 | +22 [+11, +33] / -1 / +20 (174) / 2.8 | +31 [+16, +46] / -7 / +30 (162) / 2.5 | +32 [+17, +47] / -8 / +35 (161) / 2.5 | +23 / +25 | -5 / — |
| 9B | 14 | 3.1 | +22 [+11, +33] / -1 / +20 (174) / 2.8 | +29 [+15, +44] / -7 / +25 (166) / 2.1 | +27 [+12, +42] / -8 / +26 (166) / 1.9 | +20 / +20 | -6 / — |
| 9B | 15 | 3.1 | +22 [+11, +33] / -1 / +20 (174) / 2.8 | +29 [+14, +44] / -6 / +25 (167) / 2.1 | +28 [+13, +43] / -2 / +22 (167) / 2.2 | +25 / +21 | -5 / — |

## judge allenai: must-answer base (conf), change in right answers against conf without any write

Prompt = conf with the E2 sentence, no write. Direction = conf, no sentence, the learned E direction at the given strength (strength 1 = the frozen size of the write). Each cell: dR [95%] / dAns / G_same (n) / mean words. Random = the smaller margin of the direction over the two random directions (both signs) of the same strength, a point estimate (the per-layer cells tables carry the intervals); mean difference = dR of the driver's md arm of the same strength, which writes strength x |mean difference| (not size-matched to the learned write, strength x gap).

| model | layer | conf alone: words | prompt + E2 | direction 4 | direction 5 | margin over random (4 / 5) | mean difference (4 / 5) |
|---|---|---|---|---|---|---|---|
| 9B | 11 | 3.1 | +24 [+14, +34] / +2 / +18 (175) / 2.8 | +38 [+24, +52] / +1 / +28 (167) / 2.5 | +36 [+22, +50] / +1 / +27 (169) / 2.5 | +29 / +26 | +4 / — |
| 9B | 14 | 3.1 | +24 [+14, +34] / +2 / +18 (175) / 2.8 | +36 [+23, +50] / +0 / +27 (173) / 2.1 | +33 [+17, +48] / -1 / +28 (175) / 1.9 | +30 / +28 | -2 / — |
| 9B | 15 | 3.1 | +24 [+14, +34] / +2 / +18 (175) / 2.8 | +33 [+18, +48] / +0 / +26 (174) / 2.1 | +26 [+10, +42] / +1 / +19 (170) / 2.2 | +29 / +19 | -2 / — |


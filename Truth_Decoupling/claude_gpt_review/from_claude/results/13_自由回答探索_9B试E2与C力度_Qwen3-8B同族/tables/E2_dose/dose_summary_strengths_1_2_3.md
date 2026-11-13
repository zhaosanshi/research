## judge pro: must-answer base (conf), change in right answers against conf without any write

Prompt = conf with the E2 sentence, no write. Direction = conf, no sentence, the learned E direction at the given strength (strength 1 = the frozen size of the write). Each cell: dR [95%] / dAns / G_same (n) / mean words. Random = the smaller margin of the direction over the two random directions (both signs) of the same strength, a point estimate (the per-layer cells tables carry the intervals); mean difference = dR of the driver's md arm of the same strength, which writes strength x |mean difference| (not size-matched to the learned write, strength x gap).

| model | layer | conf alone: words | prompt + E2 | direction 1 | direction 2 | direction 3 | margin over random (1 / 2 / 3) | mean difference (1 / 2 / 3) |
|---|---|---|---|---|---|---|---|---|
| Llama | 10 | 3.4 | +28 [+14, +42] / +2 / +13 (148) / 4.0 | +25 [+13, +37] / -1 / +14 (148) / 4.1 | +43 [+28, +58] / -4 / +21 (117) / 4.7 | +37 [+18, +55] / +1 / +20 (82) / 5.7 | +24 / +43 / +38 | -1 / -2 / -3 |
| Llama | 11 | 3.4 | +28 [+14, +42] / +2 / +13 (148) / 4.0 | +29 [+17, +41] / -2 / +17 (151) / 4.0 | +39 [+23, +55] / +0 / +22 (119) / 4.8 | +38 [+20, +56] / +2 / +24 (82) / 6.1 | +27 / +35 / +32 | +3 / +2 / +1 |
| 9B | 11 | 3.1 | +22 [+11, +33] / -1 / +20 (174) / 2.8 | +17 [+8, +27] / +0 / +16 (176) / 2.9 | +28 [+16, +40] / -1 / +25 (177) / 2.5 | +33 [+19, +47] / -7 / +30 (162) / 2.8 | +15 / +25 / +28 | -1 / -1 / +1 |
| 9B | 14 | 3.1 | +22 [+11, +33] / -1 / +20 (174) / 2.8 | +18 [+10, +27] / +0 / +16 (177) / 2.8 | +28 [+17, +40] / -1 / +26 (179) / 2.4 | +30 [+16, +44] / -4 / +27 (171) / 2.3 | +14 / +21 / +22 | +1 / -2 / -7 |
| 9B | 15 | 3.1 | +22 [+11, +33] / -1 / +20 (174) / 2.8 | +18 [+10, +27] / +0 / +16 (175) / 2.9 | +26 [+15, +38] / -3 / +26 (175) / 2.5 | +30 [+16, +44] / -4 / +28 (166) / 2.5 | +15 / +25 / +28 | +10 / +8 / -3 |

## judge allenai: must-answer base (conf), change in right answers against conf without any write

Prompt = conf with the E2 sentence, no write. Direction = conf, no sentence, the learned E direction at the given strength (strength 1 = the frozen size of the write). Each cell: dR [95%] / dAns / G_same (n) / mean words. Random = the smaller margin of the direction over the two random directions (both signs) of the same strength, a point estimate (the per-layer cells tables carry the intervals); mean difference = dR of the driver's md arm of the same strength, which writes strength x |mean difference| (not size-matched to the learned write, strength x gap).

| model | layer | conf alone: words | prompt + E2 | direction 1 | direction 2 | direction 3 | margin over random (1 / 2 / 3) | mean difference (1 / 2 / 3) |
|---|---|---|---|---|---|---|---|---|
| Llama | 10 | 3.4 | +29 [+15, +42] / +2 / +15 (154) / 4.0 | +30 [+18, +42] / +2 / +15 (155) / 4.1 | +51 [+36, +65] / +0 / +18 (122) / 4.7 | +58 [+40, +75] / +1 / +23 (88) / 5.7 | +28 / +49 / +59 | +0 / -3 / -6 |
| Llama | 11 | 3.4 | +29 [+15, +42] / +2 / +15 (154) / 4.0 | +29 [+17, +41] / +0 / +19 (158) / 4.0 | +44 [+28, +60] / -2 / +20 (121) / 4.8 | +60 [+43, +77] / -3 / +24 (85) / 6.1 | +31 / +40 / +58 | +0 / -6 / -3 |
| 9B | 11 | 3.1 | +24 [+14, +34] / +2 / +18 (175) / 2.8 | +17 [+8, +26] / -1 / +17 (175) / 2.9 | +25 [+15, +36] / +0 / +22 (177) / 2.5 | +39 [+25, +53] / +1 / +27 (165) / 2.8 | +14 / +20 / +33 | +2 / +0 / +6 |
| 9B | 14 | 3.1 | +24 [+14, +34] / +2 / +18 (175) / 2.8 | +17 [+9, +26] / +0 / +16 (177) / 2.8 | +24 [+14, +35] / +1 / +21 (180) / 2.4 | +32 [+20, +44] / +1 / +23 (174) / 2.3 | +16 / +21 / +27 | +4 / +2 / -2 |
| 9B | 15 | 3.1 | +24 [+14, +34] / +2 / +18 (175) / 2.8 | +18 [+9, +27] / +0 / +16 (175) / 2.9 | +24 [+13, +36] / +1 / +21 (178) / 2.5 | +34 [+22, +47] / +1 / +25 (170) / 2.5 | +13 / +20 / +30 | +12 / +4 / +3 |


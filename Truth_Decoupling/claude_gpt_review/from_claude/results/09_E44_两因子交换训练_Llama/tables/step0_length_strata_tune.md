stratum 'not longer' = the answer under b has at most 1 word more than under a; 'longer' = at least 2 words more.
dR = right-count change; the interval resamples ALL questions (questions outside the stratum contribute 0).

| pair a -> b | judge | all: dR | not longer: n, dR [95%], W>R / R>W | longer: n, dR [95%], W>R / R>W | same text: n |
|---|---|---|---|---|---|
| x -> x_F | allenai | +24 | 37, +0 [-6, +5], 3 / 1 | 155, +24 [+12, +36], 28 / 6 | 17 |
| x -> x_F | pro | +16 | 37, +0 [-4, +4], 1 / 1 | 155, +16 [+7, +26], 17 / 5 | 17 |
| x -> x_E1 | allenai | +11 | 162, +8 [-1, +17], 12 / 3 | 30, +3 [-1, +7], 4 / 1 | 93 |
| x -> x_E1 | pro | +14 | 162, +7 [-1, +15], 11 / 2 | 30, +7 [+1, +14], 6 / 1 | 93 |
| conf -> conf_E1 | allenai | +20 | 159, +9 [+2, +16], 10 / 2 | 33, +11 [+4, +18], 12 / 1 | 87 |
| conf -> conf_E1 | pro | +21 | 159, +9 [+1, +17], 10 / 4 | 33, +12 [+4, +21], 15 / 2 | 87 |
| conf -> conf_E2 | allenai | +29 | 158, +17 [+6, +28], 24 / 9 | 34, +12 [+5, +20], 14 / 2 | 55 |
| conf -> conf_E2 | pro | +28 | 158, +18 [+6, +30], 23 / 10 | 34, +10 [+2, +18], 14 / 3 | 55 |
| conf_M -> conf_E1 | allenai | +44 | 147, +22 [+13, +32], 22 / 1 | 45, +22 [+13, +31], 22 / 1 | 65 |
| conf_M -> conf_E1 | pro | +33 | 147, +12 [+3, +22], 16 / 4 | 45, +21 [+12, +31], 23 / 2 | 65 |
| caut -> caut_E1 | allenai | +8 | 158, +7 [-1, +15], 10 / 3 | 34, +1 [-5, +7], 5 / 4 | 100 |
| caut -> caut_E1 | pro | +12 | 158, +6 [+0, +13], 8 / 2 | 34, +6 [-1, +13], 7 / 3 | 100 |

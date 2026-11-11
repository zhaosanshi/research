# E+C against selection by the model's own confidence at the same N (from_claude/39)

## judge pro

### E+C with C -0.25: N = 168 non-abstained answers (PRIMARY, pre-registered)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E+C -0.25 (C decides what to abstain on)** | — | **93 / 68 / 4 / 3** | 57.8% | 9.1 | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 186 (103) | 97 / 64 / 4 / 3 | 60.2% | 10.7 | -4 [-13, +5] | +4 [-4, +12] | 93.0 | 103 |
| E+C -0.5, then selection (same prompt) [sum_logp] | 186 (103) | 95 / 65 / 5 / 3 | 59.4% | 7.5 | -2 [-12, +8] | +3 [-5, +11] | 93.0 | 103 |
| E+C -0.5, then selection (same prompt) [first_logp] | 186 (103) | 94 / 66 / 5 / 3 | 58.8% | 10.2 | -1 [-10, +8] | +2 [-5, +9] | 93.0 | 103 |
| E+C -1, then selection (same prompt) | 188 (101) | 95 / 63 / 6 / 4 | 60.1% | 11.8 | -2 [-12, +7] | +5 [-5, +15] | 90.3 | 101 |
| E+C -1, then selection (same prompt) [sum_logp] | 188 (101) | 95 / 64 / 5 / 4 | 59.7% | 8.1 | -2 [-12, +7] | +4 [-5, +13] | 90.3 | 101 |
| E+C -1, then selection (same prompt) [first_logp] | 188 (101) | 92 / 65 / 7 / 4 | 58.6% | 10.8 | +1 [-9, +11] | +3 [-6, +12] | 90.3 | 101 |
| E + selection (must-answer prompt) | 192 (106) | 98 / 61 / 6 / 3 | 61.6% | 5.6 | -5 [-17, +7] | +7 [-4, +18] | 92.8 | 106 |
| E + selection (must-answer prompt) [sum_logp] | 192 (106) | 97 / 62 / 6 / 3 | 61.0% | 4.2 | -4 [-17, +8] | +6 [-5, +16] | 92.8 | 106 |
| E + selection (must-answer prompt) [first_logp] | 192 (106) | 98 / 61 / 6 / 3 | 61.6% | 5.0 | -5 [-17, +7] | +7 [-3, +17] | 92.8 | 106 |
| E + selection (plain prompt) | 151 (83) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 173 (66) | 65 / 95 / 6 / 2 | 40.6% | 4.7 | +28 [+15, +42] | -27 [-40, -14] | 64.1 | 66 |
| C -0.5 + selection | 179 (67) | 65 / 93 / 7 / 3 | 41.1% | 4.6 | +28 [+15, +42] | -25 [-39, -11] | 62.9 | 67 |
| C -1 + selection | 185 (73) | 71 / 90 / 6 / 1 | 44.1% | 4.2 | +22 [+8, +36] | -22 [-36, -8] | 66.3 | 73 |
| no write + selection (must-answer prompt) | 190 (70) | 65 / 97 / 6 / 0 | 40.1% | 3.5 | +28 [+14, +42] | -29 [-43, -15] | 61.9 | 70 |
| no write + selection (plain prompt) | 169 (64) | 63 / 98 / 6 / 1 | 39.1% | 4.1 | +30 [+16, +44] | -30 [-43, -17] | 63.6 | 64 |

### E+C with C -0.5: N = 186 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E+C -0.5 (C decides what to abstain on)** | — | **103 / 75 / 5 / 3** | 57.9% | 10.0 | — | — | — | — |
| E+C -1, then selection (same prompt) | 188 (101) | 100 / 75 / 7 / 4 | 57.1% | 11.0 | +3 [-2, +8] | +0 [-6, +5] | 99.9 | 101 |
| E+C -1, then selection (same prompt) [sum_logp] | 188 (101) | 101 / 74 / 7 / 4 | 57.7% | 10.6 | +2 [-3, +7] | +1 [-5, +7] | 99.9 | 101 |
| E+C -1, then selection (same prompt) [first_logp] | 188 (101) | 101 / 74 / 7 / 4 | 57.7% | 10.8 | +2 [-3, +7] | +1 [-5, +7] | 99.9 | 101 |
| E + selection (must-answer prompt) | 192 (106) | 104 / 73 / 6 / 3 | 58.8% | 5.4 | -1 [-11, +10] | +2 [-7, +11] | 102.7 | 106 |
| E + selection (must-answer prompt) [sum_logp] | 192 (106) | 106 / 71 / 6 / 3 | 59.9% | 4.6 | -3 [-13, +7] | +4 [-6, +14] | 102.7 | 106 |
| E + selection (must-answer prompt) [first_logp] | 192 (106) | 106 / 70 / 6 / 4 | 60.2% | 5.2 | -3 [-13, +7] | +5 [-5, +15] | 102.7 | 106 |
| E + selection (plain prompt) | 151 (83) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 173 (66) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 179 (67) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 185 (73) | fewer candidates than N | — | — | — | — | — | — |
| no write + selection (must-answer prompt) | 190 (70) | 70 / 109 / 7 / 0 | 39.1% | 3.4 | +33 [+19, +47] | -34 [-48, -21] | 68.5 | 70 |
| no write + selection (plain prompt) | 169 (64) | fewer candidates than N | — | — | — | — | — | — |

### E+C with C -1: N = 188 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E+C -1 (C decides what to abstain on)** | — | **101 / 76 / 7 / 4** | 57.1% | 10.9 | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 186 (103) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 186 (103) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 186 (103) | fewer candidates than N | — | — | — | — | — | — |
| E + selection (must-answer prompt) | 192 (106) | 105 / 73 / 6 / 4 | 59.0% | 5.3 | -4 [-15, +7] | +3 [-7, +13] | 103.8 | 106 |
| E + selection (must-answer prompt) [sum_logp] | 192 (106) | 106 / 72 / 6 / 4 | 59.6% | 4.9 | -5 [-16, +6] | +4 [-6, +14] | 103.8 | 106 |
| E + selection (must-answer prompt) [first_logp] | 192 (106) | 106 / 72 / 6 / 4 | 59.6% | 5.3 | -5 [-16, +6] | +4 [-6, +14] | 103.8 | 106 |
| E + selection (plain prompt) | 151 (83) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 173 (66) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 179 (67) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 185 (73) | fewer candidates than N | — | — | — | — | — | — |
| no write + selection (must-answer prompt) | 190 (70) | 70 / 111 / 7 / 0 | 38.7% | 3.4 | +31 [+17, +45] | -35 [-49, -21] | 69.3 | 70 |
| no write + selection (plain prompt) | 169 (64) | fewer candidates than N | — | — | — | — | — | — |


## judge allenai

### E+C with C -0.25: N = 168 non-abstained answers (PRIMARY, pre-registered)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E+C -0.25 (C decides what to abstain on)** | — | **118 / 50 / 0 / 0** | 70.2% | 9.1 | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 186 (129) | 120 / 48 / 0 / 0 | 71.4% | 10.7 | -2 [-11, +7] | +2 [-5, +9] | 116.5 | 129 |
| E+C -0.5, then selection (same prompt) [sum_logp] | 186 (129) | 116 / 52 / 0 / 0 | 69.0% | 7.5 | +2 [-8, +12] | -2 [-7, +3] | 116.5 | 129 |
| E+C -0.5, then selection (same prompt) [first_logp] | 186 (129) | 118 / 50 / 0 / 0 | 70.2% | 10.2 | +0 [-9, +9] | +0 [-4, +4] | 116.5 | 129 |
| E+C -1, then selection (same prompt) | 188 (133) | 124 / 44 / 0 / 0 | 73.8% | 11.8 | -6 [-17, +5] | +6 [-3, +15] | 118.9 | 133 |
| E+C -1, then selection (same prompt) [sum_logp] | 188 (133) | 119 / 49 / 0 / 0 | 70.8% | 8.1 | -1 [-11, +9] | +1 [-5, +7] | 118.9 | 133 |
| E+C -1, then selection (same prompt) [first_logp] | 188 (133) | 119 / 49 / 0 / 0 | 70.8% | 10.8 | -1 [-13, +10] | +1 [-7, +8] | 118.9 | 133 |
| E + selection (must-answer prompt) | 192 (127) | 115 / 53 / 0 / 0 | 68.5% | 5.6 | +3 [-10, +15] | -3 [-14, +9] | 111.1 | 127 |
| E + selection (must-answer prompt) [sum_logp] | 192 (127) | 113 / 55 / 0 / 0 | 67.3% | 4.2 | +5 [-8, +18] | -5 [-15, +5] | 111.1 | 127 |
| E + selection (must-answer prompt) [first_logp] | 192 (127) | 115 / 53 / 0 / 0 | 68.5% | 5.0 | +3 [-10, +16] | -3 [-13, +7] | 111.1 | 127 |
| E + selection (plain prompt) | 151 (102) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 173 (82) | 81 / 84 / 3 / 0 | 49.1% | 4.7 | +37 [+25, +50] | -34 [-47, -22] | 79.6 | 82 |
| C -0.5 + selection | 179 (85) | 80 / 86 / 2 / 0 | 48.2% | 4.6 | +38 [+25, +51] | -36 [-50, -23] | 79.8 | 85 |
| C -1 + selection | 185 (86) | 80 / 86 / 2 / 0 | 48.2% | 4.2 | +38 [+24, +52] | -36 [-50, -23] | 78.1 | 86 |
| no write + selection (must-answer prompt) | 190 (88) | 80 / 88 / 0 / 0 | 47.6% | 3.5 | +38 [+24, +52] | -38 [-52, -24] | 77.8 | 88 |
| no write + selection (plain prompt) | 169 (78) | 77 / 88 / 3 / 0 | 46.7% | 4.1 | +41 [+29, +54] | -38 [-51, -25] | 77.5 | 78 |

### E+C with C -0.5: N = 186 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E+C -0.5 (C decides what to abstain on)** | — | **129 / 57 / 0 / 0** | 69.4% | 10.0 | — | — | — | — |
| E+C -1, then selection (same prompt) | 188 (133) | 133 / 53 / 0 / 0 | 71.5% | 11.0 | -4 [-11, +3] | +4 [-3, +11] | 131.6 | 133 |
| E+C -1, then selection (same prompt) [sum_logp] | 188 (133) | 132 / 54 / 0 / 0 | 71.0% | 10.6 | -3 [-10, +4] | +3 [-3, +9] | 131.6 | 133 |
| E+C -1, then selection (same prompt) [first_logp] | 188 (133) | 132 / 54 / 0 / 0 | 71.0% | 10.8 | -3 [-10, +4] | +3 [-3, +10] | 131.6 | 133 |
| E + selection (must-answer prompt) | 192 (127) | 123 / 63 / 0 / 0 | 66.1% | 5.4 | +6 [-4, +16] | -6 [-16, +3] | 123.0 | 127 |
| E + selection (must-answer prompt) [sum_logp] | 192 (127) | 124 / 62 / 0 / 0 | 66.7% | 4.6 | +5 [-5, +15] | -5 [-14, +4] | 123.0 | 127 |
| E + selection (must-answer prompt) [first_logp] | 192 (127) | 126 / 60 / 0 / 0 | 67.7% | 5.2 | +3 [-7, +13] | -3 [-13, +7] | 123.0 | 127 |
| E + selection (plain prompt) | 151 (102) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 173 (82) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 179 (85) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 185 (86) | fewer candidates than N | — | — | — | — | — | — |
| no write + selection (must-answer prompt) | 190 (88) | 86 / 99 / 1 / 0 | 46.5% | 3.4 | +43 [+29, +57] | -42 [-56, -28] | 86.1 | 88 |
| no write + selection (plain prompt) | 169 (78) | fewer candidates than N | — | — | — | — | — | — |

### E+C with C -1: N = 188 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E+C -1 (C decides what to abstain on)** | — | **133 / 55 / 0 / 0** | 70.7% | 10.9 | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 186 (129) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 186 (129) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 186 (129) | fewer candidates than N | — | — | — | — | — | — |
| E + selection (must-answer prompt) | 192 (127) | 125 / 63 / 0 / 0 | 66.5% | 5.3 | +8 [-3, +19] | -8 [-18, +2] | 124.4 | 127 |
| E + selection (must-answer prompt) [sum_logp] | 192 (127) | 125 / 63 / 0 / 0 | 66.5% | 4.9 | +8 [-3, +19] | -8 [-18, +2] | 124.4 | 127 |
| E + selection (must-answer prompt) [first_logp] | 192 (127) | 126 / 62 / 0 / 0 | 67.0% | 5.3 | +7 [-4, +17] | -7 [-17, +3] | 124.4 | 127 |
| E + selection (plain prompt) | 151 (102) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 173 (82) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 179 (85) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 185 (86) | fewer candidates than N | — | — | — | — | — | — |
| no write + selection (must-answer prompt) | 190 (88) | 86 / 101 / 1 / 0 | 46.0% | 3.4 | +47 [+33, +61] | -46 [-61, -32] | 87.1 | 88 |
| no write + selection (plain prompt) | 169 (78) | fewer candidates than N | — | — | — | — | — | — |

## pre-registered readings (39 section 2), both judges together

- C -0.25 | E+C -0.5, then selection (same prompt): leans to the comparator (both point estimates < 0; intervals include 0)
- C -0.25 | E+C -1, then selection (same prompt): leans to the comparator (both point estimates < 0; intervals include 0)
- C -0.25 | E + selection (must-answer prompt): cannot tell
- C -0.5 | E+C -1, then selection (same prompt): cannot tell
- C -0.5 | E + selection (must-answer prompt): cannot tell
- C -1 | E + selection (must-answer prompt): cannot tell

# E+C against selection by the model's own confidence at the same N (from_claude/39 procedure; from_claude/44)

## judge pro

### E +1 with C -0.25: N = 166 non-abstained answers (PRIMARY, chosen by rule abstentions before judging)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +1, C -0.25 (C decides what to abstain on)** | — | **107 / 55 / 4 / 0** | 66.0% | 3.2 | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 174 (112) | 110 / 54 / 2 / 0 | 67.1% | 3.4 | -3 [-10, +3] | +1 [-5, +7] | 106.9 | 112 |
| E+C -0.5, then selection (same prompt) [sum_logp] | 174 (112) | 109 / 55 / 2 / 0 | 66.5% | 2.9 | -2 [-9, +5] | +0 [-7, +7] | 106.9 | 112 |
| E+C -0.5, then selection (same prompt) [first_logp] | 174 (112) | 109 / 56 / 1 / 0 | 66.1% | 3.2 | -2 [-8, +4] | -1 [-7, +5] | 106.9 | 112 |
| E+C -1, then selection (same prompt) | 186 (112) | 106 / 58 / 2 / 0 | 64.6% | 4.0 | +1 [-7, +9] | -3 [-11, +5] | 100.0 | 112 |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (112) | 103 / 61 / 2 / 0 | 62.8% | 2.9 | +4 [-5, +13] | -6 [-15, +3] | 100.0 | 112 |
| E+C -1, then selection (same prompt) [first_logp] | 186 (112) | 106 / 59 / 1 / 0 | 64.2% | 3.6 | +1 [-7, +9] | -4 [-13, +5] | 100.0 | 112 |
| E+C -2, then selection (same prompt) | 190 (106) | 102 / 61 / 2 / 1 | 62.6% | 5.0 | +5 [-5, +16] | -6 [-15, +3] | 92.6 | 106 |
| E+C -2, then selection (same prompt) [sum_logp] | 190 (106) | 96 / 67 / 2 / 1 | 58.9% | 3.0 | +11 [+0, +23] | -12 [-23, -1] | 92.6 | 106 |
| E+C -2, then selection (same prompt) [first_logp] | 190 (106) | 101 / 63 / 1 / 1 | 61.6% | 4.6 | +6 [-5, +17] | -8 [-18, +2] | 92.6 | 106 |
| E + selection (must-answer prompt) | 190 (118) | 109 / 54 / 1 / 2 | 66.9% | 3.0 | -2 [-10, +6] | +1 [-8, +10] | 103.1 | 118 |
| E + selection (must-answer prompt) [sum_logp] | 190 (118) | 108 / 56 / 1 / 1 | 65.9% | 2.2 | -1 [-12, +10] | -1 [-11, +9] | 103.1 | 118 |
| E + selection (must-answer prompt) [first_logp] | 190 (118) | 108 / 55 / 1 / 2 | 66.3% | 2.8 | -1 [-9, +7] | +0 [-7, +7] | 103.1 | 118 |
| E + selection (plain prompt) | 155 (102) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (97) | 96 / 67 / 3 / 0 | 58.9% | 3.7 | +11 [+2, +20] | -12 [-22, -3] | 93.6 | 97 |
| C -0.5 + selection | 185 (99) | 96 / 68 / 2 / 0 | 58.5% | 3.9 | +11 [+2, +21] | -13 [-23, -3] | 88.8 | 99 |
| C -1 + selection | 188 (96) | 91 / 73 / 1 / 1 | 55.5% | 4.4 | +16 [+6, +26] | -18 [-29, -7] | 84.8 | 96 |
| C -2 + selection | 190 (93) | 91 / 73 / 1 / 1 | 55.5% | 5.2 | +16 [+5, +28] | -18 [-30, -6] | 81.3 | 93 |
| no write + selection (must-answer prompt) | 190 (100) | 90 / 73 / 1 / 2 | 55.2% | 3.1 | +17 [+7, +27] | -18 [-29, -8] | 87.4 | 100 |
| no write + selection (plain prompt) | 167 (99) | 99 / 63 / 4 / 0 | 61.1% | 3.3 | +8 [+1, +16] | -8 [-17, +0] | 98.4 | 99 |

### E +1 with C -0.5: N = 174 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +1, C -0.5 (C decides what to abstain on)** | — | **112 / 60 / 2 / 0** | 65.1% | 3.3 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) | 186 (112) | 110 / 62 / 2 / 0 | 64.0% | 3.9 | +2 [-4, +8] | -2 [-10, +6] | 104.8 | 112 |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (112) | 108 / 64 / 2 / 0 | 62.8% | 3.1 | +4 [-2, +10] | -4 [-11, +3] | 104.8 | 112 |
| E+C -1, then selection (same prompt) [first_logp] | 186 (112) | 109 / 64 / 1 / 0 | 63.0% | 3.8 | +3 [-2, +8] | -4 [-10, +2] | 104.8 | 112 |
| E+C -2, then selection (same prompt) | 190 (106) | 103 / 68 / 2 / 1 | 60.2% | 4.9 | +9 [+0, +18] | -8 [-18, +2] | 97.1 | 106 |
| E+C -2, then selection (same prompt) [sum_logp] | 190 (106) | 100 / 71 / 2 / 1 | 58.5% | 3.2 | +12 [+3, +22] | -11 [-21, -1] | 97.1 | 106 |
| E+C -2, then selection (same prompt) [first_logp] | 190 (106) | 103 / 69 / 1 / 1 | 59.9% | 4.6 | +9 [+0, +18] | -9 [-18, +0] | 97.1 | 106 |
| E + selection (must-answer prompt) | 190 (118) | 112 / 59 / 1 / 2 | 65.5% | 2.9 | +0 [-8, +9] | +1 [-8, +10] | 108.1 | 118 |
| E + selection (must-answer prompt) [sum_logp] | 190 (118) | 111 / 61 / 1 / 1 | 64.5% | 2.4 | +1 [-9, +11] | -1 [-11, +9] | 108.1 | 118 |
| E + selection (must-answer prompt) [first_logp] | 190 (118) | 112 / 59 / 1 / 2 | 65.5% | 2.8 | +0 [-8, +8] | +1 [-7, +9] | 108.1 | 118 |
| E + selection (plain prompt) | 155 (102) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (97) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (99) | 97 / 75 / 2 / 0 | 56.4% | 3.8 | +15 [+7, +23] | -15 [-26, -4] | 93.1 | 99 |
| C -1 + selection | 188 (96) | 92 / 80 / 1 / 1 | 53.5% | 4.4 | +20 [+11, +30] | -20 [-31, -9] | 88.9 | 96 |
| C -2 + selection | 190 (93) | 93 / 79 / 1 / 1 | 54.1% | 5.1 | +19 [+8, +30] | -19 [-30, -8] | 85.2 | 93 |
| no write + selection (must-answer prompt) | 190 (100) | 95 / 76 / 1 / 2 | 55.6% | 3.1 | +17 [+8, +27] | -16 [-26, -6] | 91.6 | 100 |
| no write + selection (plain prompt) | 167 (99) | fewer candidates than N | — | — | — | — | — | — |

### E +1 with C -1: N = 186 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +1, C -1 (C decides what to abstain on)** | — | **112 / 72 / 2 / 0** | 60.9% | 3.8 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 174 (112) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 174 (112) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 174 (112) | fewer candidates than N | — | — | — | — | — | — |
| E+C -2, then selection (same prompt) | 190 (106) | 106 / 77 / 2 / 1 | 57.9% | 4.7 | +6 [-3, +15] | -5 [-14, +3] | 103.8 | 106 |
| E+C -2, then selection (same prompt) [sum_logp] | 190 (106) | 106 / 77 / 2 / 1 | 57.9% | 4.1 | +6 [-3, +15] | -5 [-14, +3] | 103.8 | 106 |
| E+C -2, then selection (same prompt) [first_logp] | 190 (106) | 106 / 78 / 1 / 1 | 57.6% | 4.6 | +6 [-3, +15] | -6 [-15, +3] | 103.8 | 106 |
| E + selection (must-answer prompt) | 190 (118) | 117 / 66 / 1 / 2 | 63.9% | 2.9 | -5 [-13, +3] | +6 [-2, +14] | 115.5 | 118 |
| E + selection (must-answer prompt) [sum_logp] | 190 (118) | 117 / 66 / 1 / 2 | 63.9% | 2.7 | -5 [-13, +4] | +6 [-2, +14] | 115.5 | 118 |
| E + selection (must-answer prompt) [first_logp] | 190 (118) | 117 / 66 / 1 / 2 | 63.9% | 2.8 | -5 [-13, +3] | +6 [-2, +14] | 115.5 | 118 |
| E + selection (plain prompt) | 155 (102) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (97) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (99) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 188 (96) | 96 / 88 / 1 / 1 | 52.2% | 4.2 | +16 [+7, +25] | -16 [-26, -7] | 95.0 | 96 |
| C -2 + selection | 190 (93) | 93 / 91 / 1 / 1 | 50.5% | 5.0 | +19 [+8, +30] | -19 [-30, -8] | 91.0 | 93 |
| no write + selection (must-answer prompt) | 190 (100) | 99 / 84 / 1 / 2 | 54.1% | 3.1 | +13 [+5, +22] | -12 [-22, -3] | 97.9 | 100 |
| no write + selection (plain prompt) | 167 (99) | fewer candidates than N | — | — | — | — | — | — |

### E +1 with C -2: N = 190 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +1, C -2 (C decides what to abstain on)** | — | **106 / 81 / 2 / 1** | 56.7% | 4.6 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 166 (107) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 174 (112) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 174 (112) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 174 (112) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) | 186 (112) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (112) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) [first_logp] | 186 (112) | fewer candidates than N | — | — | — | — | — | — |
| E + selection (must-answer prompt) | 190 (118) | 118 / 69 / 1 / 2 | 63.1% | 2.8 | -12 [-21, -3] | +12 [+3, +21] | 118.0 | 118 |
| E + selection (must-answer prompt) [sum_logp] | 190 (118) | 118 / 69 / 1 / 2 | 63.1% | 2.8 | -12 [-21, -3] | +12 [+3, +21] | 118.0 | 118 |
| E + selection (must-answer prompt) [first_logp] | 190 (118) | 118 / 69 / 1 / 2 | 63.1% | 2.8 | -12 [-21, -3] | +12 [+3, +21] | 118.0 | 118 |
| E + selection (plain prompt) | 155 (102) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (97) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (99) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 188 (96) | fewer candidates than N | — | — | — | — | — | — |
| C -2 + selection | 190 (93) | 93 / 95 / 1 / 1 | 49.5% | 5.0 | +13 [+4, +22] | -14 [-23, -5] | 93.0 | 93 |
| no write + selection (must-answer prompt) | 190 (100) | 100 / 87 / 1 / 2 | 53.5% | 3.1 | +6 [-4, +15] | -6 [-15, +3] | 100.0 | 100 |
| no write + selection (plain prompt) | 167 (99) | fewer candidates than N | — | — | — | — | — | — |

### E +2 with C -0.25: N = 150 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +2, C -0.25 (C decides what to abstain on)** | — | **106 / 42 / 2 / 0** | 71.6% | 3.0 | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 171 (118) | 109 / 38 / 3 / 0 | 74.1% | 3.0 | -3 [-11, +5] | +4 [-3, +11] | 103.5 | 118 |
| E+C -0.5, then selection (same prompt) [sum_logp] | 171 (118) | 108 / 39 / 3 / 0 | 73.5% | 2.3 | -2 [-10, +6] | +3 [-5, +11] | 103.5 | 118 |
| E+C -0.5, then selection (same prompt) [first_logp] | 171 (118) | 106 / 42 / 2 / 0 | 71.6% | 2.7 | +0 [-8, +8] | +0 [-7, +7] | 103.5 | 118 |
| E+C -1, then selection (same prompt) | 186 (120) | 104 / 43 / 3 / 0 | 70.7% | 3.2 | +2 [-9, +13] | -1 [-10, +8] | 96.8 | 120 |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (120) | 106 / 42 / 1 / 1 | 71.6% | 2.1 | +0 [-10, +10] | +0 [-10, +10] | 96.8 | 120 |
| E+C -1, then selection (same prompt) [first_logp] | 186 (120) | 103 / 46 / 1 / 0 | 69.1% | 2.7 | +3 [-7, +13] | -4 [-13, +5] | 96.8 | 120 |
| E+C -2, then selection (same prompt) | 188 (120) | 101 / 47 / 1 / 1 | 68.2% | 4.2 | +5 [-7, +17] | -5 [-16, +5] | 95.7 | 120 |
| E+C -2, then selection (same prompt) [sum_logp] | 188 (120) | 103 / 45 / 1 / 1 | 69.6% | 2.2 | +3 [-8, +14] | -3 [-15, +9] | 95.7 | 120 |
| E+C -2, then selection (same prompt) [first_logp] | 188 (120) | 103 / 45 / 1 / 1 | 69.6% | 3.4 | +3 [-8, +14] | -3 [-14, +8] | 95.7 | 120 |
| E + selection (must-answer prompt) | 190 (126) | 107 / 41 / 1 / 1 | 72.3% | 2.5 | -1 [-12, +10] | +1 [-8, +10] | 99.5 | 126 |
| E + selection (must-answer prompt) [sum_logp] | 190 (126) | 109 / 37 / 1 / 3 | 74.7% | 1.8 | -3 [-14, +7] | +5 [-5, +16] | 99.5 | 126 |
| E + selection (must-answer prompt) [first_logp] | 190 (126) | 108 / 40 / 1 / 1 | 73.0% | 2.3 | -2 [-12, +8] | +2 [-8, +12] | 99.5 | 126 |
| E + selection (plain prompt) | 142 (98) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (97) | 93 / 54 / 3 / 0 | 63.3% | 3.7 | +13 [+2, +24] | -12 [-23, -1] | 84.6 | 97 |
| C -0.5 + selection | 185 (99) | 90 / 58 / 2 / 0 | 60.8% | 4.0 | +16 [+5, +27] | -16 [-28, -4] | 80.3 | 99 |
| C -1 + selection | 188 (96) | 87 / 61 / 1 / 1 | 58.8% | 4.5 | +19 [+9, +30] | -19 [-31, -7] | 76.6 | 96 |
| C -2 + selection | 190 (93) | 86 / 62 / 1 / 1 | 58.1% | 5.3 | +20 [+7, +33] | -20 [-33, -7] | 73.4 | 93 |
| no write + selection (must-answer prompt) | 190 (100) | 85 / 63 / 1 / 1 | 57.4% | 3.2 | +21 [+9, +33] | -21 [-34, -9] | 78.9 | 100 |
| no write + selection (plain prompt) | 167 (99) | 95 / 51 / 4 / 0 | 65.1% | 3.4 | +11 [+0, +22] | -9 [-21, +3] | 88.9 | 99 |

### E +2 with C -0.5: N = 171 non-abstained answers (PRIMARY, chosen by rule abstentions before judging)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +2, C -0.5 (C decides what to abstain on)** | — | **118 / 50 / 3 / 0** | 70.2% | 3.0 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) | 186 (120) | 116 / 52 / 3 / 0 | 69.0% | 3.3 | +2 [-5, +9] | -2 [-10, +6] | 110.3 | 120 |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (120) | 115 / 53 / 2 / 1 | 68.5% | 2.6 | +3 [-5, +11] | -3 [-12, +6] | 110.3 | 120 |
| E+C -1, then selection (same prompt) [first_logp] | 186 (120) | 114 / 56 / 1 / 0 | 67.1% | 3.0 | +4 [-4, +12] | -6 [-13, +1] | 110.3 | 120 |
| E+C -2, then selection (same prompt) | 188 (120) | 112 / 57 / 1 / 1 | 66.3% | 4.2 | +6 [-3, +15] | -7 [-17, +3] | 109.1 | 120 |
| E+C -2, then selection (same prompt) [sum_logp] | 188 (120) | 113 / 56 / 1 / 1 | 66.9% | 2.7 | +5 [-4, +15] | -6 [-16, +4] | 109.1 | 120 |
| E+C -2, then selection (same prompt) [first_logp] | 188 (120) | 114 / 55 / 1 / 1 | 67.5% | 3.5 | +4 [-5, +13] | -5 [-15, +5] | 109.1 | 120 |
| E + selection (must-answer prompt) | 190 (126) | 118 / 48 / 2 / 3 | 71.1% | 2.6 | +0 [-8, +8] | +2 [-7, +11] | 113.4 | 126 |
| E + selection (must-answer prompt) [sum_logp] | 190 (126) | 119 / 48 / 1 / 3 | 71.3% | 2.1 | -1 [-10, +8] | +2 [-8, +12] | 113.4 | 126 |
| E + selection (must-answer prompt) [first_logp] | 190 (126) | 117 / 50 / 1 / 3 | 70.1% | 2.5 | +1 [-7, +9] | +0 [-8, +8] | 113.4 | 126 |
| E + selection (plain prompt) | 142 (98) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (97) | 97 / 71 / 3 / 0 | 57.7% | 3.6 | +21 [+11, +31] | -21 [-32, -10] | 96.4 | 97 |
| C -0.5 + selection | 185 (99) | 97 / 72 / 2 / 0 | 57.4% | 3.9 | +21 [+11, +32] | -22 [-34, -11] | 91.5 | 99 |
| C -1 + selection | 188 (96) | 92 / 77 / 1 / 1 | 54.4% | 4.4 | +26 [+15, +37] | -27 [-39, -16] | 87.3 | 96 |
| C -2 + selection | 190 (93) | 92 / 77 / 1 / 1 | 54.4% | 5.2 | +26 [+14, +39] | -27 [-39, -15] | 83.7 | 93 |
| no write + selection (must-answer prompt) | 190 (100) | 92 / 76 / 1 / 2 | 54.8% | 3.1 | +26 [+15, +37] | -26 [-38, -14] | 90.0 | 100 |
| no write + selection (plain prompt) | 167 (99) | fewer candidates than N | — | — | — | — | — | — |

### E +2 with C -1: N = 186 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +2, C -1 (C decides what to abstain on)** | — | **120 / 62 / 3 / 1** | 65.9% | 3.1 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 171 (118) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 171 (118) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 171 (118) | fewer candidates than N | — | — | — | — | — | — |
| E+C -2, then selection (same prompt) | 188 (120) | 120 / 64 / 1 / 1 | 65.2% | 4.0 | +0 [-8, +8] | -2 [-10, +6] | 118.7 | 120 |
| E+C -2, then selection (same prompt) [sum_logp] | 188 (120) | 120 / 64 / 1 / 1 | 65.2% | 3.7 | +0 [-8, +8] | -2 [-10, +6] | 118.7 | 120 |
| E+C -2, then selection (same prompt) [first_logp] | 188 (120) | 120 / 64 / 1 / 1 | 65.2% | 3.8 | +0 [-8, +8] | -2 [-10, +6] | 118.7 | 120 |
| E + selection (must-answer prompt) | 190 (126) | 124 / 56 / 2 / 4 | 68.9% | 2.5 | -4 [-13, +5] | +6 [-3, +15] | 123.3 | 126 |
| E + selection (must-answer prompt) [sum_logp] | 190 (126) | 124 / 56 / 2 / 4 | 68.9% | 2.4 | -4 [-13, +5] | +6 [-3, +15] | 123.3 | 126 |
| E + selection (must-answer prompt) [first_logp] | 190 (126) | 125 / 56 / 1 / 4 | 69.1% | 2.4 | -5 [-13, +3] | +6 [-3, +15] | 123.3 | 126 |
| E + selection (plain prompt) | 142 (98) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (97) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (99) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 188 (96) | 96 / 88 / 1 / 1 | 52.2% | 4.2 | +24 [+13, +36] | -26 [-38, -14] | 95.0 | 96 |
| C -2 + selection | 190 (93) | 93 / 91 / 1 / 1 | 50.5% | 5.0 | +27 [+15, +40] | -29 [-42, -17] | 91.0 | 93 |
| no write + selection (must-answer prompt) | 190 (100) | 99 / 84 / 1 / 2 | 54.1% | 3.1 | +21 [+10, +32] | -22 [-34, -10] | 97.9 | 100 |
| no write + selection (plain prompt) | 167 (99) | fewer candidates than N | — | — | — | — | — | — |

### E +2 with C -2: N = 188 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +2, C -2 (C decides what to abstain on)** | — | **120 / 66 / 1 / 1** | 64.5% | 4.0 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 150 (106) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 171 (118) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 171 (118) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 171 (118) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) | 186 (120) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (120) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) [first_logp] | 186 (120) | fewer candidates than N | — | — | — | — | — | — |
| E + selection (must-answer prompt) | 190 (126) | 126 / 56 / 2 / 4 | 69.2% | 2.5 | -6 [-14, +1] | +10 [+3, +18] | 124.7 | 126 |
| E + selection (must-answer prompt) [sum_logp] | 190 (126) | 126 / 56 / 2 / 4 | 69.2% | 2.4 | -6 [-14, +1] | +10 [+2, +18] | 124.7 | 126 |
| E + selection (must-answer prompt) [first_logp] | 190 (126) | 125 / 58 / 1 / 4 | 68.3% | 2.4 | -5 [-13, +2] | +8 [+0, +16] | 124.7 | 126 |
| E + selection (plain prompt) | 142 (98) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (97) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (99) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 188 (96) | 96 / 90 / 1 / 1 | 51.6% | 4.2 | +24 [+13, +35] | -24 [-35, -13] | 96.0 | 96 |
| C -2 + selection | 190 (93) | 93 / 93 / 1 / 1 | 50.0% | 5.0 | +27 [+16, +39] | -27 [-39, -16] | 92.0 | 93 |
| no write + selection (must-answer prompt) | 190 (100) | 99 / 86 / 1 / 2 | 53.5% | 3.1 | +21 [+10, +33] | -20 [-32, -9] | 98.9 | 100 |
| no write + selection (plain prompt) | 167 (99) | fewer candidates than N | — | — | — | — | — | — |


## judge allenai

### E +1 with C -0.25: N = 166 non-abstained answers (PRIMARY, chosen by rule abstentions before judging)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +1, C -0.25 (C decides what to abstain on)** | — | **117 / 49 / 0 / 0** | 70.5% | 3.2 | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 174 (120) | 118 / 48 / 0 / 0 | 71.1% | 3.4 | -1 [-5, +3] | +1 [-5, +7] | 114.5 | 120 |
| E+C -0.5, then selection (same prompt) [sum_logp] | 174 (120) | 114 / 52 / 0 / 0 | 68.7% | 2.9 | +3 [-3, +9] | -3 [-8, +2] | 114.5 | 120 |
| E+C -0.5, then selection (same prompt) [first_logp] | 174 (120) | 116 / 50 / 0 / 0 | 69.9% | 3.2 | +1 [-3, +5] | -1 [-6, +4] | 114.5 | 120 |
| E+C -1, then selection (same prompt) | 186 (126) | 119 / 47 / 0 / 0 | 71.7% | 4.0 | -2 [-10, +6] | +2 [-6, +10] | 112.5 | 126 |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (126) | 111 / 54 / 1 / 0 | 67.3% | 2.9 | +6 [-3, +15] | -5 [-14, +3] | 112.5 | 126 |
| E+C -1, then selection (same prompt) [first_logp] | 186 (126) | 118 / 48 / 0 / 0 | 71.1% | 3.6 | -1 [-9, +7] | +1 [-7, +9] | 112.5 | 126 |
| E+C -2, then selection (same prompt) | 190 (122) | 116 / 50 / 0 / 0 | 69.9% | 5.0 | +1 [-9, +11] | -1 [-9, +8] | 106.6 | 122 |
| E+C -2, then selection (same prompt) [sum_logp] | 190 (122) | 106 / 58 / 2 / 0 | 64.6% | 3.0 | +11 [+0, +22] | -9 [-19, +0] | 106.6 | 122 |
| E+C -2, then selection (same prompt) [first_logp] | 190 (122) | 115 / 51 / 0 / 0 | 69.3% | 4.6 | +2 [-8, +12] | -2 [-10, +6] | 106.6 | 122 |
| E + selection (must-answer prompt) | 190 (129) | 119 / 46 / 1 / 0 | 72.1% | 3.0 | -2 [-11, +7] | +3 [-6, +12] | 112.7 | 129 |
| E + selection (must-answer prompt) [sum_logp] | 190 (129) | 113 / 51 / 2 / 0 | 68.9% | 2.2 | +4 [-8, +16] | -2 [-12, +7] | 112.7 | 129 |
| E + selection (must-answer prompt) [first_logp] | 190 (129) | 118 / 46 / 2 / 0 | 72.0% | 2.8 | -1 [-10, +8] | +3 [-4, +11] | 112.7 | 129 |
| E + selection (plain prompt) | 155 (111) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (106) | 105 / 59 / 2 / 0 | 64.0% | 3.7 | +12 [+4, +21] | -10 [-19, -2] | 102.3 | 106 |
| C -0.5 + selection | 185 (111) | 107 / 57 / 2 / 0 | 65.2% | 3.9 | +10 [+1, +20] | -8 [-18, +2] | 99.6 | 111 |
| C -1 + selection | 188 (110) | 104 / 60 / 2 / 0 | 63.4% | 4.4 | +13 [+2, +24] | -11 [-22, +0] | 97.1 | 110 |
| C -2 + selection | 190 (108) | 101 / 63 / 2 / 0 | 61.6% | 5.2 | +16 [+4, +28] | -14 [-25, -3] | 94.4 | 108 |
| no write + selection (must-answer prompt) | 190 (111) | 101 / 63 / 2 / 0 | 61.6% | 3.1 | +16 [+5, +27] | -14 [-24, -4] | 97.0 | 111 |
| no write + selection (plain prompt) | 167 (103) | 103 / 62 / 1 / 0 | 62.4% | 3.3 | +14 [+7, +22] | -13 [-21, -6] | 102.4 | 103 |

### E +1 with C -0.5: N = 174 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +1, C -0.5 (C decides what to abstain on)** | — | **120 / 54 / 0 / 0** | 69.0% | 3.3 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) | 186 (126) | 123 / 51 / 0 / 0 | 70.7% | 3.9 | -3 [-11, +4] | +3 [-5, +11] | 117.9 | 126 |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (126) | 116 / 57 / 1 / 0 | 67.1% | 3.1 | +4 [-4, +12] | -3 [-10, +3] | 117.9 | 126 |
| E+C -1, then selection (same prompt) [first_logp] | 186 (126) | 122 / 52 / 0 / 0 | 70.1% | 3.8 | -2 [-9, +5] | +2 [-4, +8] | 117.9 | 126 |
| E+C -2, then selection (same prompt) | 190 (122) | 118 / 55 / 1 / 0 | 68.2% | 4.9 | +2 [-8, +12] | -1 [-10, +7] | 111.7 | 122 |
| E+C -2, then selection (same prompt) [sum_logp] | 190 (122) | 111 / 61 / 2 / 0 | 64.5% | 3.2 | +9 [-1, +19] | -7 [-15, +0] | 111.7 | 122 |
| E+C -2, then selection (same prompt) [first_logp] | 190 (122) | 117 / 56 / 1 / 0 | 67.6% | 4.6 | +3 [-6, +12] | -2 [-9, +5] | 111.7 | 122 |
| E + selection (must-answer prompt) | 190 (129) | 122 / 50 / 2 / 0 | 70.9% | 2.9 | -2 [-10, +6] | +4 [-4, +12] | 118.1 | 129 |
| E + selection (must-answer prompt) [sum_logp] | 190 (129) | 119 / 53 / 2 / 0 | 69.2% | 2.4 | +1 [-10, +12] | +1 [-7, +9] | 118.1 | 129 |
| E + selection (must-answer prompt) [first_logp] | 190 (129) | 122 / 49 / 3 / 0 | 71.3% | 2.8 | -2 [-10, +6] | +5 [-2, +12] | 118.1 | 129 |
| E + selection (plain prompt) | 155 (111) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (106) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (111) | 108 / 63 / 3 / 0 | 63.2% | 3.8 | +12 [+3, +21] | -9 [-19, +1] | 104.4 | 111 |
| C -1 + selection | 188 (110) | 106 / 66 / 2 / 0 | 61.6% | 4.4 | +14 [+3, +25] | -12 [-23, -2] | 101.8 | 110 |
| C -2 + selection | 190 (108) | 104 / 68 / 2 / 0 | 60.5% | 5.1 | +16 [+4, +28] | -14 [-25, -4] | 98.9 | 108 |
| no write + selection (must-answer prompt) | 190 (111) | 105 / 67 / 2 / 0 | 61.0% | 3.1 | +15 [+4, +26] | -13 [-23, -4] | 101.7 | 111 |
| no write + selection (plain prompt) | 167 (103) | fewer candidates than N | — | — | — | — | — | — |

### E +1 with C -1: N = 186 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +1, C -1 (C decides what to abstain on)** | — | **126 / 59 / 1 / 0** | 68.1% | 3.8 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 174 (120) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 174 (120) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 174 (120) | fewer candidates than N | — | — | — | — | — | — |
| E+C -2, then selection (same prompt) | 190 (122) | 122 / 62 / 2 / 0 | 66.3% | 4.7 | +4 [-4, +13] | -3 [-10, +4] | 119.4 | 122 |
| E+C -2, then selection (same prompt) [sum_logp] | 190 (122) | 119 / 65 / 2 / 0 | 64.7% | 4.1 | +7 [-2, +16] | -6 [-14, +1] | 119.4 | 122 |
| E+C -2, then selection (same prompt) [first_logp] | 190 (122) | 121 / 63 / 2 / 0 | 65.8% | 4.6 | +5 [-3, +13] | -4 [-12, +4] | 119.4 | 122 |
| E + selection (must-answer prompt) | 190 (129) | 128 / 55 / 3 / 0 | 69.9% | 2.9 | -2 [-9, +5] | +4 [-3, +11] | 126.3 | 129 |
| E + selection (must-answer prompt) [sum_logp] | 190 (129) | 127 / 56 / 3 / 0 | 69.4% | 2.7 | -1 [-9, +7] | +3 [-4, +10] | 126.3 | 129 |
| E + selection (must-answer prompt) [first_logp] | 190 (129) | 126 / 57 / 3 / 0 | 68.9% | 2.8 | +0 [-7, +7] | +2 [-5, +9] | 126.3 | 129 |
| E + selection (plain prompt) | 155 (111) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (106) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (111) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 188 (110) | 110 / 74 / 2 / 0 | 59.8% | 4.2 | +16 [+7, +25] | -15 [-24, -7] | 108.8 | 110 |
| C -2 + selection | 190 (108) | 107 / 77 / 2 / 0 | 58.2% | 5.0 | +19 [+9, +30] | -18 [-27, -9] | 105.7 | 108 |
| no write + selection (must-answer prompt) | 190 (111) | 110 / 74 / 2 / 0 | 59.8% | 3.1 | +16 [+7, +26] | -15 [-24, -7] | 108.7 | 111 |
| no write + selection (plain prompt) | 167 (103) | fewer candidates than N | — | — | — | — | — | — |

### E +1 with C -2: N = 190 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +1, C -2 (C decides what to abstain on)** | — | **122 / 66 / 2 / 0** | 64.9% | 4.6 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 166 (117) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 174 (120) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 174 (120) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 174 (120) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) | 186 (126) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (126) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) [first_logp] | 186 (126) | fewer candidates than N | — | — | — | — | — | — |
| E + selection (must-answer prompt) | 190 (129) | 129 / 58 / 3 / 0 | 69.0% | 2.8 | -7 [-15, +1] | +8 [+0, +16] | 129.0 | 129 |
| E + selection (must-answer prompt) [sum_logp] | 190 (129) | 129 / 58 / 3 / 0 | 69.0% | 2.8 | -7 [-15, +1] | +8 [+0, +16] | 129.0 | 129 |
| E + selection (must-answer prompt) [first_logp] | 190 (129) | 129 / 58 / 3 / 0 | 69.0% | 2.8 | -7 [-15, +1] | +8 [+0, +16] | 129.0 | 129 |
| E + selection (plain prompt) | 155 (111) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (106) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (111) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 188 (110) | fewer candidates than N | — | — | — | — | — | — |
| C -2 + selection | 190 (108) | 108 / 80 / 2 / 0 | 57.4% | 5.0 | +14 [+5, +23] | -14 [-24, -5] | 108.0 | 108 |
| no write + selection (must-answer prompt) | 190 (111) | 111 / 76 / 3 / 0 | 59.4% | 3.1 | +11 [+2, +20] | -10 [-19, -2] | 111.0 | 111 |
| no write + selection (plain prompt) | 167 (103) | fewer candidates than N | — | — | — | — | — | — |

### E +2 with C -0.25: N = 150 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +2, C -0.25 (C decides what to abstain on)** | — | **114 / 36 / 0 / 0** | 76.0% | 3.0 | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 171 (126) | 114 / 36 / 0 / 0 | 76.0% | 3.0 | +0 [-8, +8] | +0 [-6, +6] | 110.5 | 126 |
| E+C -0.5, then selection (same prompt) [sum_logp] | 171 (126) | 113 / 37 / 0 / 0 | 75.3% | 2.3 | +1 [-8, +10] | -1 [-8, +6] | 110.5 | 126 |
| E+C -0.5, then selection (same prompt) [first_logp] | 171 (126) | 113 / 37 / 0 / 0 | 75.3% | 2.7 | +1 [-7, +9] | -1 [-7, +5] | 110.5 | 126 |
| E+C -1, then selection (same prompt) | 186 (132) | 113 / 36 / 1 / 0 | 75.8% | 3.2 | +1 [-10, +12] | +0 [-8, +8] | 106.5 | 132 |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (132) | 109 / 40 / 1 / 0 | 73.2% | 2.1 | +5 [-6, +16] | -4 [-13, +5] | 106.5 | 132 |
| E+C -1, then selection (same prompt) [first_logp] | 186 (132) | 110 / 39 / 1 / 0 | 73.8% | 2.7 | +4 [-6, +14] | -3 [-10, +4] | 106.5 | 132 |
| E+C -2, then selection (same prompt) | 188 (130) | 111 / 38 / 1 / 0 | 74.5% | 4.2 | +3 [-9, +15] | -2 [-11, +7] | 103.7 | 130 |
| E+C -2, then selection (same prompt) [sum_logp] | 188 (130) | 106 / 43 / 1 / 0 | 71.1% | 2.2 | +8 [-4, +20] | -7 [-17, +3] | 103.7 | 130 |
| E+C -2, then selection (same prompt) [first_logp] | 188 (130) | 112 / 37 / 1 / 0 | 75.2% | 3.4 | +2 [-8, +12] | -1 [-10, +8] | 103.7 | 130 |
| E + selection (must-answer prompt) | 190 (135) | 113 / 36 / 1 / 0 | 75.8% | 2.5 | +1 [-10, +12] | +0 [-8, +8] | 106.6 | 135 |
| E + selection (must-answer prompt) [sum_logp] | 190 (135) | 114 / 35 / 1 / 0 | 76.5% | 1.8 | +0 [-12, +12] | +1 [-8, +10] | 106.6 | 135 |
| E + selection (must-answer prompt) [first_logp] | 190 (135) | 114 / 35 / 1 / 0 | 76.5% | 2.3 | +0 [-10, +10] | +1 [-7, +9] | 106.6 | 135 |
| E + selection (plain prompt) | 142 (109) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (106) | 101 / 48 / 1 / 0 | 67.8% | 3.7 | +13 [+3, +24] | -12 [-21, -3] | 92.4 | 106 |
| C -0.5 + selection | 185 (111) | 101 / 48 / 1 / 0 | 67.8% | 4.0 | +13 [+2, +24] | -12 [-22, -2] | 90.0 | 111 |
| C -1 + selection | 188 (110) | 101 / 47 / 2 / 0 | 68.2% | 4.5 | +13 [+2, +24] | -11 [-21, -2] | 87.8 | 110 |
| C -2 + selection | 190 (108) | 96 / 53 / 1 / 0 | 64.4% | 5.3 | +18 [+6, +31] | -17 [-28, -7] | 85.3 | 108 |
| no write + selection (must-answer prompt) | 190 (111) | 93 / 55 / 2 / 0 | 62.8% | 3.2 | +21 [+9, +33] | -19 [-30, -8] | 87.6 | 111 |
| no write + selection (plain prompt) | 167 (103) | 99 / 51 / 0 / 0 | 66.0% | 3.4 | +15 [+6, +24] | -15 [-25, -5] | 92.5 | 103 |

### E +2 with C -0.5: N = 171 non-abstained answers (PRIMARY, chosen by rule abstentions before judging)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +2, C -0.5 (C decides what to abstain on)** | — | **126 / 45 / 0 / 0** | 73.7% | 3.0 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) | 186 (132) | 128 / 42 / 1 / 0 | 75.3% | 3.3 | -2 [-10, +6] | +3 [-5, +10] | 121.4 | 132 |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (132) | 122 / 48 / 1 / 0 | 71.8% | 2.6 | +4 [-5, +13] | -3 [-11, +5] | 121.4 | 132 |
| E+C -1, then selection (same prompt) [first_logp] | 186 (132) | 125 / 45 / 1 / 0 | 73.5% | 3.0 | +1 [-7, +9] | +0 [-6, +6] | 121.4 | 132 |
| E+C -2, then selection (same prompt) | 188 (130) | 122 / 48 / 1 / 0 | 71.8% | 4.2 | +4 [-6, +14] | -3 [-11, +5] | 118.2 | 130 |
| E+C -2, then selection (same prompt) [sum_logp] | 188 (130) | 119 / 51 / 1 / 0 | 70.0% | 2.7 | +7 [-3, +17] | -6 [-15, +2] | 118.2 | 130 |
| E+C -2, then selection (same prompt) [first_logp] | 188 (130) | 124 / 46 / 1 / 0 | 72.9% | 3.5 | +2 [-7, +11] | -1 [-9, +6] | 118.2 | 130 |
| E + selection (must-answer prompt) | 190 (135) | 126 / 43 / 2 / 0 | 74.6% | 2.6 | +0 [-9, +9] | +2 [-5, +9] | 121.5 | 135 |
| E + selection (must-answer prompt) [sum_logp] | 190 (135) | 125 / 45 / 1 / 0 | 73.5% | 2.1 | +1 [-9, +12] | +0 [-9, +9] | 121.5 | 135 |
| E + selection (must-answer prompt) [first_logp] | 190 (135) | 126 / 43 / 2 / 0 | 74.6% | 2.5 | +0 [-8, +8] | +2 [-5, +10] | 121.5 | 135 |
| E + selection (plain prompt) | 142 (109) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (106) | 106 / 63 / 2 / 0 | 62.7% | 3.6 | +20 [+11, +30] | -18 [-28, -9] | 105.4 | 106 |
| C -0.5 + selection | 185 (111) | 108 / 60 / 3 / 0 | 64.3% | 3.9 | +18 [+8, +28] | -15 [-25, -5] | 102.6 | 111 |
| C -1 + selection | 188 (110) | 106 / 63 / 2 / 0 | 62.7% | 4.4 | +20 [+9, +32] | -18 [-29, -8] | 100.1 | 110 |
| C -2 + selection | 190 (108) | 103 / 66 / 2 / 0 | 60.9% | 5.2 | +23 [+11, +35] | -21 [-31, -11] | 97.2 | 108 |
| no write + selection (must-answer prompt) | 190 (111) | 102 / 67 / 2 / 0 | 60.4% | 3.1 | +24 [+13, +35] | -22 [-33, -11] | 99.9 | 111 |
| no write + selection (plain prompt) | 167 (103) | fewer candidates than N | — | — | — | — | — | — |

### E +2 with C -1: N = 186 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +2, C -1 (C decides what to abstain on)** | — | **132 / 53 / 1 / 0** | 71.4% | 3.1 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 171 (126) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 171 (126) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 171 (126) | fewer candidates than N | — | — | — | — | — | — |
| E+C -2, then selection (same prompt) | 188 (130) | 130 / 55 / 1 / 0 | 70.3% | 4.0 | +2 [-4, +8] | -2 [-9, +5] | 128.6 | 130 |
| E+C -2, then selection (same prompt) [sum_logp] | 188 (130) | 130 / 55 / 1 / 0 | 70.3% | 3.7 | +2 [-4, +8] | -2 [-8, +4] | 128.6 | 130 |
| E+C -2, then selection (same prompt) [first_logp] | 188 (130) | 130 / 55 / 1 / 0 | 70.3% | 3.8 | +2 [-4, +8] | -2 [-8, +4] | 128.6 | 130 |
| E + selection (must-answer prompt) | 190 (135) | 134 / 50 / 2 / 0 | 72.8% | 2.5 | -2 [-10, +6] | +3 [-5, +11] | 132.2 | 135 |
| E + selection (must-answer prompt) [sum_logp] | 190 (135) | 132 / 52 / 2 / 0 | 71.7% | 2.4 | +0 [-8, +8] | +1 [-7, +9] | 132.2 | 135 |
| E + selection (must-answer prompt) [first_logp] | 190 (135) | 133 / 51 / 2 / 0 | 72.3% | 2.4 | -1 [-9, +8] | +2 [-6, +10] | 132.2 | 135 |
| E + selection (plain prompt) | 142 (109) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (106) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (111) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 188 (110) | 110 / 74 / 2 / 0 | 59.8% | 4.2 | +22 [+10, +34] | -21 [-33, -10] | 108.8 | 110 |
| C -2 + selection | 190 (108) | 107 / 77 / 2 / 0 | 58.2% | 5.0 | +25 [+13, +37] | -24 [-36, -13] | 105.7 | 108 |
| no write + selection (must-answer prompt) | 190 (111) | 110 / 74 / 2 / 0 | 59.8% | 3.1 | +22 [+11, +34] | -21 [-33, -10] | 108.7 | 111 |
| no write + selection (plain prompt) | 167 (103) | fewer candidates than N | — | — | — | — | — | — |

### E +2 with C -2: N = 188 non-abstained answers (secondary)

| system | candidates (right among them) | R / W / H / X of the N answered | right among answered | mean words | E+C minus this: right [95%] | E+C minus this: wrong [95%] | random N: expected right | at most right (oracle) |
|---|---|---|---|---|---|---|---|---|
| **E +2, C -2 (C decides what to abstain on)** | — | **130 / 57 / 1 / 0** | 69.5% | 4.0 | — | — | — | — |
| E+C -0.25, then selection (same prompt) | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [sum_logp] | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.25, then selection (same prompt) [first_logp] | 150 (114) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) | 171 (126) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [sum_logp] | 171 (126) | fewer candidates than N | — | — | — | — | — | — |
| E+C -0.5, then selection (same prompt) [first_logp] | 171 (126) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) | 186 (132) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) [sum_logp] | 186 (132) | fewer candidates than N | — | — | — | — | — | — |
| E+C -1, then selection (same prompt) [first_logp] | 186 (132) | fewer candidates than N | — | — | — | — | — | — |
| E + selection (must-answer prompt) | 190 (135) | 135 / 51 / 2 / 0 | 72.6% | 2.5 | -5 [-13, +2] | +6 [-2, +14] | 133.6 | 135 |
| E + selection (must-answer prompt) [sum_logp] | 190 (135) | 133 / 53 / 2 / 0 | 71.5% | 2.4 | -3 [-11, +5] | +4 [-4, +12] | 133.6 | 135 |
| E + selection (must-answer prompt) [first_logp] | 190 (135) | 133 / 53 / 2 / 0 | 71.5% | 2.4 | -3 [-11, +5] | +4 [-4, +12] | 133.6 | 135 |
| E + selection (plain prompt) | 142 (109) | fewer candidates than N | — | — | — | — | — | — |
| C -0.25 + selection | 172 (106) | fewer candidates than N | — | — | — | — | — | — |
| C -0.5 + selection | 185 (111) | fewer candidates than N | — | — | — | — | — | — |
| C -1 + selection | 188 (110) | 110 / 76 / 2 / 0 | 59.1% | 4.2 | +20 [+9, +31] | -19 [-30, -8] | 110.0 | 110 |
| C -2 + selection | 190 (108) | 108 / 78 / 2 / 0 | 58.1% | 5.0 | +22 [+10, +34] | -21 [-33, -9] | 106.9 | 108 |
| no write + selection (must-answer prompt) | 190 (111) | 110 / 76 / 2 / 0 | 59.1% | 3.1 | +20 [+9, +32] | -19 [-30, -8] | 109.8 | 111 |
| no write + selection (plain prompt) | 167 (103) | fewer candidates than N | — | — | — | — | — | — |

## pre-registered readings (39 section 2.1), both judges together

- E +1, C -0.25 (primary) | E+C -0.5, then selection (same prompt): leans to the comparator (both point estimates < 0; intervals include 0)
- E +1, C -0.25 (primary) | E+C -1, then selection (same prompt): cannot tell
- E +1, C -0.25 (primary) | E+C -2, then selection (same prompt): leans to the combination (both point estimates > 0; intervals include 0)
- E +1, C -0.25 (primary) | E + selection (must-answer prompt): leans to the comparator (both point estimates < 0; intervals include 0)
- E +1, C -0.5 | E+C -1, then selection (same prompt): cannot tell
- E +1, C -0.5 | E+C -2, then selection (same prompt): leans to the combination (both point estimates > 0; intervals include 0)
- E +1, C -0.5 | E + selection (must-answer prompt): cannot tell
- E +1, C -1 | E+C -2, then selection (same prompt): leans to the combination (both point estimates > 0; intervals include 0)
- E +1, C -1 | E + selection (must-answer prompt): leans to the comparator (both point estimates < 0; intervals include 0)
- E +1, C -2 | E + selection (must-answer prompt): leans to the comparator (both point estimates < 0; intervals include 0)
- E +2, C -0.25 | E+C -0.5, then selection (same prompt): cannot tell
- E +2, C -0.25 | E+C -1, then selection (same prompt): leans to the combination (both point estimates > 0; intervals include 0)
- E +2, C -0.25 | E+C -2, then selection (same prompt): leans to the combination (both point estimates > 0; intervals include 0)
- E +2, C -0.25 | E + selection (must-answer prompt): cannot tell
- E +2, C -0.5 (primary) | E+C -1, then selection (same prompt): cannot tell
- E +2, C -0.5 (primary) | E+C -2, then selection (same prompt): leans to the combination (both point estimates > 0; intervals include 0)
- E +2, C -0.5 (primary) | E + selection (must-answer prompt): cannot tell
- E +2, C -1 | E+C -2, then selection (same prompt): cannot tell
- E +2, C -1 | E + selection (must-answer prompt): leans to the comparator (both point estimates < 0; intervals include 0)
- E +2, C -2 | E + selection (must-answer prompt): leans to the comparator (both point estimates < 0; intervals include 0)

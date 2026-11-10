## frozen prompts (tune, n=192): sentence-end / mean words / negation / refusals
- x: 187 / 4.0 / 39 / 23
- x_E: 186 / 4.4 / 62 / 31
- x_conf: 188 / 3.4 / 20 / 2
- x_caut: 183 / 4.0 / 42 / 27
- gold best answers used as E training targets (n=89): sentence-end 2, mean words 9.7, negation 41

## 1. E arms on tune: sentence-end / mean words / negation / refusals | target, endpoint gap, mean |write| at prompt content positions, at decode steps
| L | main | recover | md_main | random_main |
|---|---|---|---|---|
| 0 | 6 / 8.5 / 109 / 3 | -0.12, -0.00, 0.16, 0.14 | 188 / 3.2 / 55 / 31 | -0.12, -0.00, 0.16, 0.12 | 186 / 4.1 / 39 / 23 | -0.01, -0.00, 0.02, 0.02 | 186 / 4.2 / 41 / 26 | -0.12, -0.00, 0.16, 0.13 |
| 1 | 0 / 8.6 / 78 / 6 | -0.32, -0.00, 0.38, 0.37 | 189 / 3.6 / 61 / 35 | -0.32, -0.00, 0.38, 0.34 | 186 / 4.1 / 38 / 23 | +0.01, -0.00, 0.03, 0.03 | 186 / 4.0 / 40 / 21 | -0.32, -0.00, 0.38, 0.35 |
| 2 | 0 / 8.5 / 100 / 8 | -0.46, -0.00, 0.56, 0.53 | 188 / 3.6 / 56 / 31 | -0.46, -0.00, 0.55, 0.51 | 187 / 4.0 / 40 / 25 | -0.12, -0.00, 0.05, 0.04 | 184 / 4.1 / 39 / 23 | -0.46, -0.00, 0.56, 0.52 |
| 3 | 20 / 10.9 / 63 / 18 | +0.65, +0.01, 0.81, 0.76 | 184 / 3.6 / 60 / 34 | +0.64, +0.01, 0.79, 0.70 | 187 / 4.0 / 39 / 23 | +0.15, +0.01, 0.16, 0.12 | 187 / 3.9 / 37 / 23 | +0.65, +0.01, 0.81, 0.73 |
| 4 | 2 / 7.8 / 55 / 10 | -1.01, -0.02, 1.16, 1.13 | 189 / 3.4 / 51 / 28 | -0.99, -0.02, 1.14, 1.08 | 187 / 4.1 / 39 / 23 | +0.24, -0.02, 0.18, 0.12 | 186 / 4.0 / 39 / 25 | -1.01, -0.02, 1.16, 1.10 |
| 5 | 4 / 7.1 / 69 / 19 | -1.14, -0.02, 1.11, 1.18 | 188 / 3.7 / 58 / 40 | -1.12, -0.02, 1.09, 1.10 | 186 / 4.0 / 39 / 24 | -0.01, -0.02, 0.15, 0.10 | 186 / 3.9 / 36 / 22 | -1.14, -0.02, 1.11, 1.14 |
| 6 | 3 / 7.4 / 72 / 15 | +1.08, +0.02, 1.21, 1.20 | 187 / 4.3 / 55 / 31 | +1.06, +0.02, 1.18, 1.13 | 187 / 4.1 / 39 / 23 | -0.26, +0.02, 0.21, 0.12 | 187 / 3.8 / 34 / 18 | +1.08, +0.02, 1.21, 1.16 |
| 7 | 4 / 10.9 / 79 / 6 | -1.39, -0.04, 1.59, 1.53 | 185 / 4.3 / 58 / 35 | -1.35, -0.04, 1.55, 1.42 | 185 / 4.1 / 40 / 25 | +0.31, -0.04, 0.51, 0.47 | 183 / 4.4 / 42 / 27 | -1.39, -0.04, 1.59, 1.46 |
| 8 | 3 / 8.9 / 50 / 4 | +1.61, +0.07, 1.79, 1.62 | 185 / 3.9 / 61 / 32 | +1.54, +0.07, 1.71, 1.47 | 186 / 4.1 / 40 / 23 | +0.50, +0.07, 0.40, 0.39 | 183 / 4.3 / 39 / 23 | +1.61, +0.07, 1.79, 1.55 |
| 9 | 5 / 8.1 / 87 / 12 | -2.36, +0.06, 2.32, 2.32 | 185 / 3.9 / 44 / 23 | -2.42, +0.06, 2.38, 2.22 | 184 / 4.4 / 36 / 21 | -0.81, +0.06, 0.70, 0.66 | 183 / 4.0 / 55 / 40 | -2.36, +0.06, 2.32, 2.14 |
| 10 | 1 / 9.3 / 65 / 4 | +2.41, +0.16, 2.58, 2.52 | 189 / 3.8 / 58 / 37 | +2.25, +0.16, 2.39, 2.20 | 188 / 3.9 / 39 / 24 | +0.62, +0.16, 0.42, 0.40 | 186 / 4.3 / 23 / 10 | +2.41, +0.16, 2.58, 2.39 |
| 11 | 17 / 6.3 / 73 / 39 | -2.47, -0.25, 2.69, 2.44 | 190 / 3.8 / 71 / 49 | -2.22, -0.25, 2.38, 2.01 | 187 / 4.1 / 38 / 25 | +0.57, -0.25, 0.44, 0.38 | 185 / 3.9 / 35 / 21 | -2.47, -0.25, 2.69, 2.31 |
| 12 | 2 / 7.9 / 57 / 18 | +2.41, +0.32, 2.67, 2.29 | 190 / 4.0 / 57 / 31 | +2.09, +0.32, 2.30, 1.80 | 186 / 4.0 / 40 / 24 | +0.33, +0.32, 0.46, 0.48 | 187 / 3.7 / 37 / 26 | +2.41, +0.32, 2.67, 2.17 |
| 13 | 7 / 7.4 / 42 / 11 | -2.48, -0.42, 2.74, 2.49 | 191 / 3.8 / 56 / 25 | -2.06, -0.42, 2.26, 1.83 | 185 / 4.4 / 42 / 25 | +0.74, -0.42, 0.56, 0.77 | 185 / 4.2 / 39 / 25 | -2.48, -0.42, 2.74, 2.41 |
| 14 | 1 / 8.8 / 58 / 8 | -2.31, -0.47, 2.74, 2.30 | 186 / 4.3 / 64 / 28 | -1.84, -0.47, 2.17, 1.59 | 184 / 4.5 / 45 / 27 | +1.13, -0.47, 0.72, 1.03 | 190 / 3.7 / 45 / 31 | -2.31, -0.47, 2.74, 2.21 |
| 15 | 7 / 7.0 / 55 / 21 | -2.82, -0.52, 2.89, 2.73 | 178 / 5.4 / 75 / 35 | -2.31, -0.52, 2.30, 1.95 | 185 / 4.6 / 49 / 28 | +1.97, -0.52, 1.19, 1.46 | 185 / 4.4 / 46 / 26 | -2.82, -0.52, 2.89, 2.72 |
| 16 | 8 / 5.0 / 48 / 21 | +2.97, +0.48, 3.18, 2.77 | 186 / 4.8 / 71 / 36 | +2.48, +0.48, 2.62, 2.01 | 186 / 4.2 / 48 / 28 | +2.02, +0.48, 1.43, 1.62 | 188 / 4.4 / 38 / 22 | +2.97, +0.48, 3.18, 2.77 |
| 17 | 4 / 6.5 / 49 / 20 | +3.12, +0.55, 3.14, 3.07 | 177 / 5.5 / 67 / 31 | +2.57, +0.55, 2.47, 2.18 | 186 / 4.1 / 47 / 27 | +2.08, +0.55, 1.57, 1.66 | 187 / 4.2 / 40 / 23 | +3.12, +0.55, 3.14, 3.05 |
| 18 | 23 / 6.1 / 45 / 22 | -2.97, -0.51, 3.64, 2.97 | 184 / 5.5 / 73 / 34 | -2.45, -0.51, 3.04, 2.14 | 186 / 4.3 / 45 / 26 | +2.36, -0.51, 1.83, 1.82 | 181 / 3.9 / 38 / 24 | -2.97, -0.51, 3.64, 3.01 |
| 19 | 5 / 5.6 / 44 / 24 | -2.94, -0.47, 3.61, 3.13 | 183 / 5.2 / 70 / 33 | -2.47, -0.47, 3.06, 2.32 | 185 / 4.3 / 46 / 28 | +2.29, -0.47, 1.81, 1.67 | 187 / 3.8 / 40 / 26 | -2.94, -0.47, 3.61, 3.10 |
| 20 | 11 / 6.9 / 53 / 27 | +2.82, +0.56, 3.75, 3.28 | 184 / 5.0 / 74 / 37 | +2.27, +0.56, 3.11, 2.34 | 184 / 4.4 / 48 / 28 | +2.53, +0.56, 2.40, 2.10 | 190 / 3.9 / 37 / 23 | +2.82, +0.56, 3.75, 3.24 |
| 21 | 1 / 6.4 / 61 / 33 | +3.87, +0.50, 4.40, 3.70 | 175 / 5.3 / 73 / 36 | +3.37, +0.50, 3.78, 2.87 | 184 / 4.4 / 46 / 28 | +3.07, +0.50, 2.80, 2.47 | 185 / 3.8 / 39 / 24 | +3.87, +0.50, 4.40, 3.69 |
| 22 | 13 / 4.7 / 50 / 32 | +3.94, +0.41, 4.11, 3.70 | 184 / 4.4 / 70 / 40 | +3.53, +0.41, 3.55, 3.01 | 184 / 4.4 / 46 / 27 | +3.54, +0.41, 3.07, 2.81 | 187 / 3.9 / 41 / 27 | +3.94, +0.41, 4.11, 3.71 |
| 23 | 10 / 4.2 / 53 / 34 | -3.84, -0.53, 4.28, 3.75 | 188 / 4.3 / 74 / 45 | -3.31, -0.53, 3.58, 2.91 | 183 / 4.5 / 47 / 28 | +3.96, -0.53, 3.17, 2.94 | 186 / 4.1 / 41 / 24 | -3.84, -0.53, 4.28, 3.80 |
| 24 | 145 / 4.8 / 53 / 36 | +3.82, +0.49, 4.17, 4.07 | 187 / 4.8 / 76 / 43 | +3.33, +0.49, 3.51, 3.22 | 186 / 4.3 / 45 / 25 | +4.51, +0.49, 3.40, 3.17 | 189 / 4.0 / 39 / 22 | +3.82, +0.49, 4.17, 4.08 |
| 25 | 35 / 3.5 / 48 / 33 | +5.18, +0.56, 6.08, 5.54 | 174 / 4.8 / 72 / 39 | +4.61, +0.56, 5.39, 4.56 | 185 / 4.4 / 45 / 25 | +4.76, +0.56, 3.96, 3.40 | 184 / 4.3 / 42 / 25 | +5.18, +0.56, 6.08, 5.52 |
| 26 | 14 / 3.5 / 47 / 33 | +5.82, +0.58, 6.18, 5.68 | 172 / 4.9 / 71 / 37 | +5.24, +0.58, 5.41, 4.78 | 186 / 4.4 / 43 / 26 | +5.05, +0.58, 4.15, 3.57 | 186 / 4.3 / 38 / 24 | +5.82, +0.58, 6.18, 5.76 |
| 27 | 23 / 3.4 / 43 / 29 | +5.68, +0.90, 7.31, 6.91 | 163 / 4.9 / 67 / 38 | +4.78, +0.90, 6.25, 5.49 | 186 / 4.2 / 44 / 28 | +5.03, +0.90, 4.35, 3.75 | 188 / 4.0 / 38 / 23 | +5.68, +0.90, 7.31, 6.92 |
| 28 | 175 / 4.6 / 46 / 30 | -4.83, -0.95, 7.76, 7.49 | 181 / 4.9 / 64 / 34 | -3.88, -0.95, 6.69, 6.00 | 186 / 4.3 / 45 / 29 | +4.77, -0.95, 4.29, 3.58 | 187 / 4.1 / 40 / 24 | -4.83, -0.95, 7.76, 7.49 |
| 29 | 129 / 4.6 / 43 / 28 | -4.12, -1.13, 8.27, 7.57 | 162 / 5.0 / 61 / 30 | -2.98, -1.13, 7.05, 5.83 | 187 / 4.1 / 44 / 28 | +4.05, -1.13, 4.76, 4.00 | 187 / 4.0 / 38 / 23 | -4.12, -1.13, 8.27, 7.37 |
| 30 | 132 / 4.6 / 40 / 25 | -5.46, -1.22, 9.46, 8.50 | 172 / 4.9 / 62 / 31 | -4.24, -1.22, 8.18, 6.67 | 188 / 4.0 / 44 / 28 | +6.56, -1.22, 5.73, 5.19 | 188 / 3.9 / 39 / 23 | -5.46, -1.22, 9.46, 8.02 |
| 31 | 173 / 4.4 / 38 / 25 | +8.98, +1.56, 8.86, 6.86 | 178 / 5.7 / 63 / 37 | +7.42, +1.56, 7.08, 5.84 | 186 / 3.8 / 47 / 30 | +11.86, +1.56, 9.02, 7.76 | 186 / 4.1 / 39 / 25 | +8.98, +1.56, 8.86, 6.75 |

## 2. judged flips by how the text changed (AllenAI, rule refusals excluded). cell = n | wrong→right | right→wrong
| condition (vs its base) | same text, only surface | changed, not longer (< +2 words) | changed and longer (≥ +2 words) |
|---|---|---|---|
| baseline_x_E (vs baseline_x) | 93 | 0 | 0 | 69 | 12 | 3 | 30 | 4 | 1 |
| C_L0_caut_to_conf (vs baseline_x_caut) | 82 | 0 | 0 | 99 | 3 | 19 | 11 | 3 | 0 |
| C_L0_conf_to_caut (vs baseline_x_conf) | 97 | 0 | 0 | 78 | 5 | 8 | 17 | 2 | 2 |
| C_L0_main (vs baseline_x) | 80 | 0 | 0 | 94 | 4 | 14 | 18 | 4 | 0 |
| C_L0_md_main (vs baseline_x) | 183 | 0 | 0 | 6 | 1 | 0 | 3 | 0 | 0 |
| C_L0_random_main (vs baseline_x) | 144 | 0 | 0 | 40 | 3 | 6 | 8 | 2 | 1 |
| C_L0_to_conf (vs baseline_x) | 80 | 0 | 0 | 94 | 4 | 14 | 18 | 4 | 0 |
| C_L10_caut_to_conf (vs baseline_x_caut) | 76 | 0 | 0 | 96 | 4 | 13 | 20 | 4 | 2 |
| C_L10_conf_to_caut (vs baseline_x_conf) | 60 | 0 | 0 | 102 | 7 | 11 | 30 | 4 | 0 |
| C_L10_main (vs baseline_x) | 74 | 0 | 0 | 91 | 6 | 13 | 27 | 4 | 2 |
| C_L10_md_main (vs baseline_x) | 177 | 0 | 0 | 12 | 2 | 0 | 3 | 0 | 0 |
| C_L10_random_main (vs baseline_x) | 97 | 0 | 0 | 74 | 5 | 9 | 21 | 3 | 1 |
| C_L10_to_conf (vs baseline_x) | 75 | 0 | 0 | 89 | 6 | 13 | 28 | 4 | 2 |
| C_L11_caut_to_conf (vs baseline_x_caut) | 66 | 0 | 0 | 113 | 6 | 23 | 13 | 0 | 2 |
| C_L11_conf_to_caut (vs baseline_x_conf) | 61 | 0 | 0 | 111 | 5 | 11 | 20 | 1 | 4 |
| C_L11_main (vs baseline_x) | 67 | 0 | 0 | 103 | 8 | 15 | 22 | 0 | 1 |
| C_L11_md_main (vs baseline_x) | 175 | 0 | 0 | 15 | 1 | 1 | 2 | 0 | 0 |
| C_L11_random_main (vs baseline_x) | 109 | 0 | 0 | 72 | 6 | 13 | 11 | 2 | 1 |
| C_L11_to_conf (vs baseline_x) | 66 | 0 | 0 | 105 | 8 | 14 | 21 | 0 | 2 |
| C_L12_caut_to_conf (vs baseline_x_caut) | 74 | 0 | 0 | 95 | 6 | 15 | 23 | 4 | 1 |
| C_L12_conf_to_caut (vs baseline_x_conf) | 81 | 0 | 0 | 87 | 6 | 9 | 24 | 5 | 0 |
| C_L12_main (vs baseline_x) | 74 | 0 | 0 | 95 | 8 | 10 | 23 | 2 | 2 |
| C_L12_md_main (vs baseline_x) | 172 | 0 | 0 | 16 | 2 | 0 | 4 | 0 | 0 |
| C_L12_random_main (vs baseline_x) | 123 | 0 | 0 | 59 | 4 | 4 | 10 | 1 | 1 |
| C_L12_to_conf (vs baseline_x) | 70 | 0 | 0 | 99 | 10 | 10 | 23 | 3 | 1 |
| C_L13_caut_to_conf (vs baseline_x_caut) | 86 | 0 | 0 | 94 | 7 | 13 | 12 | 3 | 0 |
| C_L13_conf_to_caut (vs baseline_x_conf) | 77 | 0 | 0 | 109 | 10 | 9 | 6 | 0 | 0 |
| C_L13_main (vs baseline_x) | 94 | 0 | 0 | 83 | 6 | 8 | 15 | 4 | 1 |
| C_L13_md_main (vs baseline_x) | 160 | 0 | 0 | 28 | 3 | 2 | 4 | 1 | 0 |
| C_L13_random_main (vs baseline_x) | 117 | 0 | 0 | 62 | 6 | 9 | 13 | 3 | 0 |
| C_L13_to_conf (vs baseline_x) | 93 | 0 | 0 | 84 | 6 | 8 | 15 | 4 | 1 |
| C_L14_caut_to_conf (vs baseline_x_caut) | 80 | 0 | 0 | 103 | 5 | 18 | 9 | 2 | 1 |
| C_L14_conf_to_caut (vs baseline_x_conf) | 69 | 0 | 0 | 111 | 6 | 11 | 12 | 2 | 1 |
| C_L14_main (vs baseline_x) | 82 | 0 | 0 | 99 | 9 | 16 | 11 | 3 | 0 |
| C_L14_md_main (vs baseline_x) | 165 | 0 | 0 | 21 | 4 | 1 | 6 | 1 | 0 |
| C_L14_random_main (vs baseline_x) | 107 | 0 | 0 | 75 | 3 | 8 | 10 | 1 | 0 |
| C_L14_to_conf (vs baseline_x) | 84 | 0 | 0 | 97 | 9 | 16 | 11 | 3 | 0 |
| C_L15_caut_to_conf (vs baseline_x_caut) | 108 | 0 | 0 | 72 | 1 | 9 | 12 | 1 | 3 |
| C_L15_conf_to_caut (vs baseline_x_conf) | 99 | 0 | 0 | 74 | 1 | 6 | 19 | 1 | 4 |
| C_L15_main (vs baseline_x) | 103 | 0 | 0 | 71 | 5 | 6 | 18 | 2 | 1 |
| C_L15_md_main (vs baseline_x) | 158 | 0 | 0 | 28 | 4 | 2 | 6 | 3 | 0 |
| C_L15_random_main (vs baseline_x) | 122 | 0 | 0 | 54 | 5 | 8 | 16 | 3 | 0 |
| C_L15_to_conf (vs baseline_x) | 114 | 0 | 0 | 61 | 2 | 6 | 17 | 3 | 1 |
| C_L16_caut_to_conf (vs baseline_x_caut) | 95 | 0 | 0 | 79 | 3 | 15 | 18 | 2 | 2 |
| C_L16_conf_to_caut (vs baseline_x_conf) | 79 | 0 | 0 | 84 | 5 | 7 | 29 | 1 | 3 |
| C_L16_main (vs baseline_x) | 94 | 0 | 0 | 81 | 5 | 10 | 17 | 4 | 1 |
| C_L16_md_main (vs baseline_x) | 159 | 0 | 0 | 24 | 3 | 3 | 9 | 1 | 0 |
| C_L16_random_main (vs baseline_x) | 131 | 0 | 0 | 46 | 5 | 4 | 15 | 3 | 2 |
| C_L16_to_conf (vs baseline_x) | 97 | 0 | 0 | 77 | 4 | 10 | 18 | 4 | 1 |
| C_L17_caut_to_conf (vs baseline_x_caut) | 113 | 0 | 0 | 68 | 4 | 13 | 11 | 1 | 0 |
| C_L17_conf_to_caut (vs baseline_x_conf) | 85 | 0 | 0 | 88 | 7 | 7 | 19 | 1 | 2 |
| C_L17_main (vs baseline_x) | 103 | 0 | 0 | 73 | 3 | 9 | 16 | 5 | 1 |
| C_L17_md_main (vs baseline_x) | 151 | 0 | 0 | 31 | 4 | 5 | 10 | 3 | 0 |
| C_L17_random_main (vs baseline_x) | 132 | 0 | 0 | 49 | 6 | 5 | 11 | 1 | 1 |
| C_L17_to_conf (vs baseline_x) | 111 | 0 | 0 | 68 | 5 | 9 | 13 | 4 | 0 |
| C_L18_caut_to_conf (vs baseline_x_caut) | 106 | 0 | 0 | 77 | 5 | 14 | 9 | 2 | 0 |
| C_L18_conf_to_caut (vs baseline_x_conf) | 84 | 0 | 0 | 94 | 7 | 6 | 14 | 3 | 2 |
| C_L18_main (vs baseline_x) | 100 | 0 | 0 | 79 | 7 | 9 | 13 | 0 | 1 |
| C_L18_md_main (vs baseline_x) | 149 | 0 | 0 | 32 | 5 | 4 | 11 | 2 | 0 |
| C_L18_random_main (vs baseline_x) | 140 | 0 | 0 | 44 | 5 | 2 | 8 | 0 | 1 |
| C_L18_to_conf (vs baseline_x) | 105 | 0 | 0 | 75 | 6 | 9 | 12 | 0 | 0 |
| C_L19_caut_to_conf (vs baseline_x_caut) | 122 | 0 | 0 | 58 | 2 | 8 | 12 | 4 | 0 |
| C_L19_conf_to_caut (vs baseline_x_conf) | 83 | 0 | 0 | 93 | 9 | 6 | 16 | 3 | 1 |
| C_L19_main (vs baseline_x) | 110 | 0 | 0 | 72 | 6 | 5 | 10 | 2 | 0 |
| C_L19_md_main (vs baseline_x) | 153 | 0 | 0 | 28 | 2 | 4 | 11 | 3 | 1 |
| C_L19_random_main (vs baseline_x) | 134 | 0 | 0 | 50 | 3 | 5 | 8 | 1 | 0 |
| C_L19_to_conf (vs baseline_x) | 123 | 0 | 0 | 58 | 2 | 4 | 11 | 2 | 0 |
| C_L1_caut_to_conf (vs baseline_x_caut) | 72 | 0 | 0 | 94 | 9 | 16 | 26 | 3 | 3 |
| C_L1_conf_to_caut (vs baseline_x_conf) | 69 | 0 | 0 | 95 | 10 | 11 | 28 | 3 | 2 |
| C_L1_main (vs baseline_x) | 68 | 0 | 0 | 90 | 9 | 11 | 34 | 4 | 2 |
| C_L1_md_main (vs baseline_x) | 183 | 0 | 0 | 7 | 2 | 1 | 2 | 0 | 0 |
| C_L1_random_main (vs baseline_x) | 135 | 0 | 0 | 48 | 6 | 9 | 9 | 1 | 0 |
| C_L1_to_conf (vs baseline_x) | 68 | 0 | 0 | 90 | 8 | 11 | 34 | 4 | 1 |
| C_L20_caut_to_conf (vs baseline_x_caut) | 117 | 0 | 0 | 64 | 5 | 8 | 11 | 1 | 1 |
| C_L20_conf_to_caut (vs baseline_x_conf) | 95 | 0 | 0 | 77 | 5 | 6 | 20 | 0 | 0 |
| C_L20_main (vs baseline_x) | 110 | 0 | 0 | 71 | 6 | 9 | 11 | 1 | 1 |
| C_L20_md_main (vs baseline_x) | 152 | 0 | 0 | 31 | 3 | 5 | 9 | 3 | 0 |
| C_L20_random_main (vs baseline_x) | 142 | 0 | 0 | 40 | 4 | 5 | 10 | 4 | 0 |
| C_L20_to_conf (vs baseline_x) | 122 | 0 | 0 | 60 | 8 | 7 | 10 | 1 | 1 |
| C_L21_caut_to_conf (vs baseline_x_caut) | 116 | 0 | 0 | 64 | 1 | 11 | 12 | 2 | 1 |
| C_L21_conf_to_caut (vs baseline_x_conf) | 100 | 0 | 0 | 73 | 4 | 8 | 19 | 3 | 1 |
| C_L21_main (vs baseline_x) | 111 | 0 | 0 | 64 | 5 | 8 | 17 | 3 | 0 |
| C_L21_md_main (vs baseline_x) | 154 | 0 | 0 | 30 | 2 | 7 | 8 | 1 | 0 |
| C_L21_random_main (vs baseline_x) | 138 | 0 | 0 | 48 | 5 | 5 | 6 | 1 | 0 |
| C_L21_to_conf (vs baseline_x) | 116 | 0 | 0 | 58 | 4 | 10 | 18 | 1 | 1 |
| C_L22_caut_to_conf (vs baseline_x_caut) | 117 | 0 | 0 | 65 | 5 | 7 | 10 | 2 | 3 |
| C_L22_conf_to_caut (vs baseline_x_conf) | 88 | 0 | 0 | 85 | 6 | 7 | 19 | 2 | 1 |
| C_L22_main (vs baseline_x) | 107 | 0 | 0 | 74 | 5 | 10 | 11 | 4 | 0 |
| C_L22_md_main (vs baseline_x) | 154 | 0 | 0 | 31 | 3 | 5 | 7 | 2 | 0 |
| C_L22_random_main (vs baseline_x) | 135 | 0 | 0 | 48 | 5 | 5 | 9 | 1 | 1 |
| C_L22_to_conf (vs baseline_x) | 114 | 0 | 0 | 63 | 7 | 7 | 15 | 5 | 1 |
| C_L23_caut_to_conf (vs baseline_x_caut) | 127 | 0 | 0 | 55 | 2 | 6 | 10 | 2 | 1 |
| C_L23_conf_to_caut (vs baseline_x_conf) | 101 | 0 | 0 | 79 | 7 | 7 | 12 | 0 | 2 |
| C_L23_main (vs baseline_x) | 102 | 0 | 0 | 75 | 7 | 8 | 15 | 2 | 1 |
| C_L23_md_main (vs baseline_x) | 156 | 0 | 0 | 27 | 0 | 5 | 9 | 2 | 0 |
| C_L23_random_main (vs baseline_x) | 135 | 0 | 0 | 48 | 2 | 4 | 9 | 2 | 0 |
| C_L23_to_conf (vs baseline_x) | 121 | 0 | 0 | 56 | 5 | 8 | 15 | 3 | 1 |
| C_L24_caut_to_conf (vs baseline_x_caut) | 138 | 0 | 0 | 42 | 1 | 5 | 12 | 3 | 3 |
| C_L24_conf_to_caut (vs baseline_x_conf) | 111 | 0 | 0 | 66 | 4 | 6 | 15 | 2 | 2 |
| C_L24_main (vs baseline_x) | 116 | 0 | 0 | 63 | 6 | 5 | 13 | 3 | 1 |
| C_L24_md_main (vs baseline_x) | 157 | 0 | 0 | 27 | 1 | 5 | 8 | 1 | 1 |
| C_L24_random_main (vs baseline_x) | 138 | 0 | 0 | 46 | 3 | 5 | 8 | 2 | 1 |
| C_L24_to_conf (vs baseline_x) | 134 | 0 | 0 | 47 | 4 | 5 | 11 | 0 | 1 |
| C_L25_caut_to_conf (vs baseline_x_caut) | 139 | 0 | 0 | 43 | 1 | 4 | 10 | 2 | 2 |
| C_L25_conf_to_caut (vs baseline_x_conf) | 100 | 0 | 0 | 78 | 4 | 7 | 14 | 3 | 2 |
| C_L25_main (vs baseline_x) | 110 | 0 | 0 | 73 | 6 | 7 | 9 | 1 | 1 |
| C_L25_md_main (vs baseline_x) | 162 | 0 | 0 | 22 | 0 | 3 | 8 | 1 | 1 |
| C_L25_random_main (vs baseline_x) | 139 | 0 | 0 | 40 | 3 | 1 | 13 | 2 | 0 |
| C_L25_to_conf (vs baseline_x) | 129 | 0 | 0 | 50 | 4 | 3 | 13 | 1 | 1 |
| C_L26_caut_to_conf (vs baseline_x_caut) | 128 | 0 | 0 | 55 | 3 | 7 | 9 | 2 | 2 |
| C_L26_conf_to_caut (vs baseline_x_conf) | 101 | 0 | 0 | 75 | 6 | 6 | 16 | 3 | 2 |
| C_L26_main (vs baseline_x) | 98 | 0 | 0 | 82 | 5 | 9 | 12 | 2 | 1 |
| C_L26_md_main (vs baseline_x) | 152 | 0 | 0 | 31 | 3 | 4 | 9 | 2 | 0 |
| C_L26_random_main (vs baseline_x) | 144 | 0 | 0 | 36 | 5 | 3 | 12 | 2 | 0 |
| C_L26_to_conf (vs baseline_x) | 126 | 0 | 0 | 59 | 8 | 6 | 7 | 1 | 1 |
| C_L27_caut_to_conf (vs baseline_x_caut) | 134 | 0 | 0 | 47 | 2 | 5 | 11 | 1 | 1 |
| C_L27_conf_to_caut (vs baseline_x_conf) | 105 | 0 | 0 | 76 | 6 | 7 | 11 | 1 | 2 |
| C_L27_main (vs baseline_x) | 118 | 0 | 0 | 63 | 4 | 6 | 11 | 2 | 1 |
| C_L27_md_main (vs baseline_x) | 161 | 0 | 0 | 24 | 0 | 4 | 7 | 2 | 0 |
| C_L27_random_main (vs baseline_x) | 148 | 0 | 0 | 41 | 2 | 5 | 3 | 0 | 0 |
| C_L27_to_conf (vs baseline_x) | 134 | 0 | 0 | 47 | 5 | 5 | 11 | 3 | 0 |
| C_L28_caut_to_conf (vs baseline_x_caut) | 136 | 0 | 0 | 46 | 1 | 7 | 10 | 2 | 0 |
| C_L28_conf_to_caut (vs baseline_x_conf) | 110 | 0 | 0 | 68 | 6 | 7 | 14 | 0 | 2 |
| C_L28_main (vs baseline_x) | 116 | 0 | 0 | 64 | 4 | 7 | 12 | 3 | 1 |
| C_L28_md_main (vs baseline_x) | 161 | 0 | 0 | 25 | 1 | 3 | 6 | 2 | 1 |
| C_L28_random_main (vs baseline_x) | 162 | 0 | 0 | 25 | 4 | 3 | 5 | 1 | 0 |
| C_L28_to_conf (vs baseline_x) | 139 | 0 | 0 | 40 | 2 | 4 | 13 | 4 | 0 |
| C_L29_caut_to_conf (vs baseline_x_caut) | 128 | 0 | 0 | 54 | 2 | 9 | 10 | 1 | 1 |
| C_L29_conf_to_caut (vs baseline_x_conf) | 100 | 0 | 0 | 79 | 7 | 8 | 13 | 1 | 1 |
| C_L29_main (vs baseline_x) | 109 | 0 | 0 | 68 | 5 | 4 | 15 | 1 | 1 |
| C_L29_md_main (vs baseline_x) | 166 | 0 | 0 | 20 | 1 | 2 | 6 | 2 | 1 |
| C_L29_random_main (vs baseline_x) | 153 | 0 | 0 | 34 | 4 | 2 | 5 | 0 | 0 |
| C_L29_to_conf (vs baseline_x) | 128 | 0 | 0 | 49 | 4 | 5 | 15 | 3 | 1 |
| C_L2_caut_to_conf (vs baseline_x_caut) | 77 | 0 | 0 | 92 | 5 | 17 | 23 | 3 | 0 |
| C_L2_conf_to_caut (vs baseline_x_conf) | 67 | 0 | 0 | 101 | 6 | 21 | 24 | 3 | 3 |
| C_L2_main (vs baseline_x) | 67 | 0 | 0 | 88 | 8 | 11 | 37 | 3 | 0 |
| C_L2_md_main (vs baseline_x) | 182 | 0 | 0 | 8 | 0 | 0 | 2 | 0 | 0 |
| C_L2_random_main (vs baseline_x) | 129 | 0 | 0 | 51 | 5 | 8 | 12 | 2 | 0 |
| C_L2_to_conf (vs baseline_x) | 67 | 0 | 0 | 88 | 8 | 11 | 37 | 5 | 1 |
| C_L30_caut_to_conf (vs baseline_x_caut) | 129 | 0 | 0 | 54 | 1 | 8 | 9 | 0 | 0 |
| C_L30_conf_to_caut (vs baseline_x_conf) | 102 | 0 | 0 | 79 | 8 | 8 | 11 | 2 | 1 |
| C_L30_main (vs baseline_x) | 114 | 0 | 0 | 70 | 5 | 6 | 8 | 1 | 0 |
| C_L30_md_main (vs baseline_x) | 172 | 0 | 0 | 15 | 2 | 2 | 5 | 1 | 0 |
| C_L30_random_main (vs baseline_x) | 162 | 0 | 0 | 26 | 2 | 5 | 4 | 1 | 0 |
| C_L30_to_conf (vs baseline_x) | 127 | 0 | 0 | 53 | 4 | 5 | 12 | 1 | 1 |
| C_L31_caut_to_conf (vs baseline_x_caut) | 128 | 0 | 0 | 56 | 4 | 8 | 8 | 0 | 0 |
| C_L31_conf_to_caut (vs baseline_x_conf) | 121 | 0 | 0 | 62 | 5 | 5 | 9 | 1 | 0 |
| C_L31_main (vs baseline_x) | 105 | 0 | 0 | 78 | 5 | 5 | 9 | 0 | 0 |
| C_L31_md_main (vs baseline_x) | 144 | 0 | 0 | 30 | 4 | 2 | 18 | 1 | 1 |
| C_L31_random_main (vs baseline_x) | 165 | 0 | 0 | 23 | 2 | 2 | 4 | 0 | 0 |
| C_L31_to_conf (vs baseline_x) | 122 | 0 | 0 | 58 | 4 | 4 | 12 | 1 | 0 |
| C_L3_caut_to_conf (vs baseline_x_caut) | 85 | 0 | 0 | 98 | 5 | 17 | 9 | 1 | 1 |
| C_L3_conf_to_caut (vs baseline_x_conf) | 83 | 0 | 0 | 93 | 5 | 12 | 16 | 4 | 4 |
| C_L3_main (vs baseline_x) | 87 | 0 | 0 | 95 | 5 | 12 | 10 | 1 | 0 |
| C_L3_md_main (vs baseline_x) | 177 | 0 | 0 | 13 | 2 | 0 | 2 | 0 | 0 |
| C_L3_random_main (vs baseline_x) | 142 | 0 | 0 | 42 | 6 | 4 | 8 | 1 | 0 |
| C_L3_to_conf (vs baseline_x) | 86 | 0 | 0 | 97 | 6 | 13 | 9 | 0 | 0 |
| C_L4_caut_to_conf (vs baseline_x_caut) | 74 | 0 | 0 | 96 | 6 | 22 | 22 | 3 | 4 |
| C_L4_conf_to_caut (vs baseline_x_conf) | 81 | 0 | 0 | 78 | 2 | 8 | 33 | 6 | 4 |
| C_L4_main (vs baseline_x) | 82 | 0 | 0 | 91 | 7 | 17 | 19 | 3 | 1 |
| C_L4_md_main (vs baseline_x) | 178 | 0 | 0 | 12 | 0 | 0 | 2 | 0 | 0 |
| C_L4_random_main (vs baseline_x) | 123 | 0 | 0 | 59 | 5 | 6 | 10 | 2 | 0 |
| C_L4_to_conf (vs baseline_x) | 82 | 0 | 0 | 91 | 6 | 18 | 19 | 3 | 1 |
| C_L5_caut_to_conf (vs baseline_x_caut) | 78 | 0 | 0 | 100 | 9 | 17 | 14 | 1 | 2 |
| C_L5_conf_to_caut (vs baseline_x_conf) | 43 | 0 | 0 | 118 | 5 | 10 | 31 | 6 | 3 |
| C_L5_main (vs baseline_x) | 80 | 0 | 0 | 98 | 8 | 14 | 14 | 1 | 0 |
| C_L5_md_main (vs baseline_x) | 179 | 0 | 0 | 10 | 2 | 1 | 3 | 0 | 0 |
| C_L5_random_main (vs baseline_x) | 135 | 0 | 0 | 54 | 6 | 6 | 3 | 1 | 0 |
| C_L5_to_conf (vs baseline_x) | 79 | 0 | 0 | 96 | 8 | 15 | 17 | 1 | 1 |
| C_L6_caut_to_conf (vs baseline_x_caut) | 74 | 0 | 0 | 107 | 5 | 18 | 11 | 1 | 1 |
| C_L6_conf_to_caut (vs baseline_x_conf) | 83 | 0 | 0 | 92 | 5 | 9 | 17 | 3 | 3 |
| C_L6_main (vs baseline_x) | 85 | 0 | 0 | 92 | 8 | 14 | 15 | 1 | 0 |
| C_L6_md_main (vs baseline_x) | 178 | 0 | 0 | 12 | 0 | 1 | 2 | 0 | 0 |
| C_L6_random_main (vs baseline_x) | 117 | 0 | 0 | 62 | 8 | 11 | 13 | 2 | 0 |
| C_L6_to_conf (vs baseline_x) | 84 | 0 | 0 | 93 | 7 | 13 | 15 | 1 | 0 |
| C_L7_caut_to_conf (vs baseline_x_caut) | 78 | 0 | 0 | 104 | 6 | 16 | 10 | 1 | 1 |
| C_L7_conf_to_caut (vs baseline_x_conf) | 82 | 0 | 0 | 95 | 6 | 11 | 15 | 2 | 2 |
| C_L7_main (vs baseline_x) | 88 | 0 | 0 | 93 | 5 | 10 | 11 | 1 | 1 |
| C_L7_md_main (vs baseline_x) | 170 | 0 | 0 | 20 | 4 | 1 | 2 | 0 | 1 |
| C_L7_random_main (vs baseline_x) | 128 | 0 | 0 | 48 | 2 | 7 | 16 | 3 | 0 |
| C_L7_to_conf (vs baseline_x) | 89 | 0 | 0 | 93 | 5 | 12 | 10 | 1 | 1 |
| C_L8_caut_to_conf (vs baseline_x_caut) | 85 | 0 | 0 | 99 | 8 | 18 | 8 | 0 | 1 |
| C_L8_conf_to_caut (vs baseline_x_conf) | 66 | 0 | 0 | 105 | 6 | 8 | 21 | 3 | 2 |
| C_L8_main (vs baseline_x) | 91 | 0 | 0 | 84 | 5 | 13 | 17 | 3 | 2 |
| C_L8_md_main (vs baseline_x) | 181 | 0 | 0 | 8 | 1 | 0 | 3 | 0 | 0 |
| C_L8_random_main (vs baseline_x) | 116 | 0 | 0 | 63 | 5 | 13 | 13 | 3 | 1 |
| C_L8_to_conf (vs baseline_x) | 89 | 0 | 0 | 85 | 5 | 14 | 18 | 4 | 2 |
| C_L9_caut_to_conf (vs baseline_x_caut) | 78 | 0 | 0 | 104 | 10 | 12 | 10 | 0 | 3 |
| C_L9_conf_to_caut (vs baseline_x_conf) | 48 | 0 | 0 | 117 | 6 | 7 | 27 | 6 | 2 |
| C_L9_main (vs baseline_x) | 85 | 0 | 0 | 95 | 10 | 9 | 12 | 3 | 1 |
| C_L9_md_main (vs baseline_x) | 173 | 0 | 0 | 18 | 2 | 1 | 1 | 0 | 0 |
| C_L9_random_main (vs baseline_x) | 116 | 0 | 0 | 62 | 9 | 3 | 14 | 2 | 2 |
| C_L9_to_conf (vs baseline_x) | 85 | 0 | 0 | 95 | 10 | 9 | 12 | 3 | 1 |
| E_L0_main (vs baseline_x) | 7 | 0 | 0 | 33 | 11 | 1 | 152 | 45 | 22 |
| E_L0_md_main (vs baseline_x) | 179 | 0 | 0 | 9 | 1 | 1 | 4 | 1 | 0 |
| E_L0_random_main (vs baseline_x) | 164 | 0 | 0 | 20 | 0 | 3 | 8 | 2 | 0 |
| E_L0_recover (vs baseline_x_E) | 108 | 0 | 0 | 78 | 8 | 10 | 6 | 1 | 1 |
| E_L10_main (vs baseline_x) | 8 | 1 | 0 | 28 | 4 | 5 | 156 | 31 | 12 |
| E_L10_md_main (vs baseline_x) | 168 | 0 | 0 | 19 | 2 | 0 | 5 | 1 | 0 |
| E_L10_random_main (vs baseline_x) | 90 | 0 | 0 | 77 | 6 | 8 | 25 | 4 | 0 |
| E_L10_recover (vs baseline_x_E) | 94 | 0 | 0 | 79 | 8 | 8 | 19 | 2 | 4 |
| E_L11_main (vs baseline_x) | 37 | 0 | 0 | 60 | 7 | 3 | 95 | 20 | 5 |
| E_L11_md_main (vs baseline_x) | 155 | 0 | 0 | 26 | 2 | 1 | 11 | 1 | 0 |
| E_L11_random_main (vs baseline_x) | 109 | 0 | 0 | 72 | 6 | 13 | 11 | 2 | 1 |
| E_L11_recover (vs baseline_x_E) | 80 | 0 | 0 | 91 | 11 | 8 | 21 | 1 | 2 |
| E_L12_main (vs baseline_x) | 23 | 1 | 0 | 41 | 6 | 4 | 128 | 23 | 14 |
| E_L12_md_main (vs baseline_x) | 160 | 0 | 0 | 28 | 3 | 1 | 4 | 1 | 0 |
| E_L12_random_main (vs baseline_x) | 119 | 0 | 0 | 65 | 6 | 5 | 8 | 1 | 1 |
| E_L12_recover (vs baseline_x_E) | 92 | 0 | 0 | 84 | 7 | 12 | 16 | 1 | 3 |
| E_L13_main (vs baseline_x) | 24 | 0 | 0 | 55 | 6 | 6 | 113 | 12 | 13 |
| E_L13_md_main (vs baseline_x) | 154 | 0 | 0 | 29 | 3 | 0 | 9 | 5 | 0 |
| E_L13_random_main (vs baseline_x) | 122 | 0 | 0 | 56 | 7 | 9 | 14 | 3 | 0 |
| E_L13_recover (vs baseline_x_E) | 84 | 0 | 0 | 97 | 8 | 16 | 11 | 2 | 0 |
| E_L14_main (vs baseline_x) | 19 | 0 | 0 | 47 | 3 | 5 | 126 | 21 | 10 |
| E_L14_md_main (vs baseline_x) | 141 | 0 | 0 | 36 | 5 | 1 | 15 | 4 | 0 |
| E_L14_random_main (vs baseline_x) | 114 | 0 | 0 | 70 | 4 | 6 | 8 | 1 | 0 |
| E_L14_recover (vs baseline_x_E) | 101 | 0 | 0 | 73 | 8 | 9 | 18 | 4 | 1 |
| E_L15_main (vs baseline_x) | 39 | 0 | 0 | 47 | 8 | 4 | 106 | 15 | 8 |
| E_L15_md_main (vs baseline_x) | 124 | 0 | 0 | 44 | 8 | 5 | 24 | 4 | 1 |
| E_L15_random_main (vs baseline_x) | 117 | 0 | 0 | 57 | 5 | 9 | 18 | 3 | 0 |
| E_L15_recover (vs baseline_x_E) | 87 | 0 | 0 | 64 | 6 | 9 | 41 | 5 | 2 |
| E_L16_main (vs baseline_x) | 63 | 1 | 0 | 72 | 9 | 8 | 57 | 12 | 6 |
| E_L16_md_main (vs baseline_x) | 122 | 0 | 0 | 55 | 11 | 6 | 15 | 4 | 0 |
| E_L16_random_main (vs baseline_x) | 125 | 0 | 0 | 51 | 6 | 5 | 16 | 3 | 1 |
| E_L16_recover (vs baseline_x_E) | 88 | 0 | 0 | 75 | 9 | 10 | 29 | 3 | 3 |
| E_L17_main (vs baseline_x) | 55 | 0 | 0 | 57 | 4 | 6 | 80 | 17 | 6 |
| E_L17_md_main (vs baseline_x) | 122 | 0 | 0 | 63 | 9 | 9 | 7 | 4 | 0 |
| E_L17_random_main (vs baseline_x) | 126 | 0 | 0 | 55 | 7 | 7 | 11 | 1 | 1 |
| E_L17_recover (vs baseline_x_E) | 84 | 0 | 0 | 66 | 4 | 6 | 42 | 5 | 3 |
| E_L18_main (vs baseline_x) | 49 | 1 | 0 | 67 | 8 | 5 | 76 | 17 | 6 |
| E_L18_md_main (vs baseline_x) | 125 | 0 | 0 | 56 | 9 | 7 | 11 | 5 | 0 |
| E_L18_random_main (vs baseline_x) | 133 | 0 | 0 | 49 | 5 | 4 | 10 | 0 | 1 |
| E_L18_recover (vs baseline_x_E) | 87 | 0 | 0 | 66 | 5 | 5 | 39 | 4 | 3 |
| E_L19_main (vs baseline_x) | 70 | 1 | 0 | 68 | 10 | 4 | 54 | 8 | 3 |
| E_L19_md_main (vs baseline_x) | 133 | 0 | 0 | 51 | 8 | 8 | 8 | 3 | 0 |
| E_L19_random_main (vs baseline_x) | 134 | 0 | 0 | 52 | 4 | 5 | 6 | 0 | 0 |
| E_L19_recover (vs baseline_x_E) | 94 | 0 | 0 | 59 | 3 | 5 | 39 | 5 | 0 |
| E_L1_main (vs baseline_x) | 7 | 0 | 0 | 42 | 7 | 4 | 143 | 35 | 18 |
| E_L1_md_main (vs baseline_x) | 180 | 0 | 0 | 9 | 1 | 1 | 3 | 0 | 0 |
| E_L1_random_main (vs baseline_x) | 145 | 0 | 0 | 42 | 5 | 5 | 5 | 1 | 0 |
| E_L1_recover (vs baseline_x_E) | 95 | 0 | 0 | 88 | 5 | 8 | 9 | 2 | 0 |
| E_L20_main (vs baseline_x) | 56 | 1 | 0 | 59 | 3 | 6 | 77 | 15 | 0 |
| E_L20_md_main (vs baseline_x) | 126 | 0 | 0 | 55 | 5 | 6 | 11 | 6 | 1 |
| E_L20_random_main (vs baseline_x) | 141 | 0 | 0 | 41 | 4 | 4 | 10 | 4 | 0 |
| E_L20_recover (vs baseline_x_E) | 96 | 0 | 0 | 64 | 6 | 5 | 32 | 7 | 1 |
| E_L21_main (vs baseline_x) | 42 | 1 | 0 | 66 | 7 | 6 | 84 | 19 | 5 |
| E_L21_md_main (vs baseline_x) | 126 | 0 | 0 | 53 | 6 | 8 | 13 | 4 | 1 |
| E_L21_random_main (vs baseline_x) | 138 | 0 | 0 | 48 | 4 | 6 | 6 | 1 | 0 |
| E_L21_recover (vs baseline_x_E) | 89 | 0 | 0 | 54 | 6 | 5 | 49 | 5 | 3 |
| E_L22_main (vs baseline_x) | 64 | 1 | 0 | 91 | 8 | 8 | 37 | 8 | 1 |
| E_L22_md_main (vs baseline_x) | 132 | 0 | 0 | 50 | 4 | 9 | 10 | 4 | 1 |
| E_L22_random_main (vs baseline_x) | 138 | 0 | 0 | 46 | 4 | 5 | 8 | 0 | 0 |
| E_L22_recover (vs baseline_x_E) | 89 | 0 | 0 | 80 | 8 | 5 | 23 | 2 | 1 |
| E_L23_main (vs baseline_x) | 89 | 3 | 0 | 71 | 7 | 7 | 32 | 8 | 1 |
| E_L23_md_main (vs baseline_x) | 128 | 0 | 0 | 53 | 4 | 8 | 11 | 4 | 1 |
| E_L23_random_main (vs baseline_x) | 140 | 0 | 0 | 43 | 2 | 4 | 9 | 2 | 0 |
| E_L23_recover (vs baseline_x_E) | 98 | 0 | 0 | 66 | 7 | 6 | 28 | 4 | 1 |
| E_L24_main (vs baseline_x) | 72 | 0 | 0 | 78 | 7 | 5 | 42 | 6 | 1 |
| E_L24_md_main (vs baseline_x) | 131 | 0 | 0 | 50 | 4 | 9 | 11 | 3 | 1 |
| E_L24_random_main (vs baseline_x) | 145 | 0 | 0 | 41 | 4 | 2 | 6 | 1 | 1 |
| E_L24_recover (vs baseline_x_E) | 97 | 0 | 0 | 67 | 8 | 8 | 28 | 5 | 1 |
| E_L25_main (vs baseline_x) | 82 | 2 | 0 | 95 | 10 | 10 | 15 | 1 | 3 |
| E_L25_md_main (vs baseline_x) | 139 | 0 | 0 | 40 | 3 | 6 | 13 | 6 | 0 |
| E_L25_random_main (vs baseline_x) | 136 | 0 | 0 | 44 | 3 | 2 | 12 | 3 | 0 |
| E_L25_recover (vs baseline_x_E) | 103 | 0 | 0 | 63 | 7 | 8 | 26 | 3 | 0 |
| E_L26_main (vs baseline_x) | 81 | 2 | 0 | 92 | 9 | 9 | 19 | 0 | 0 |
| E_L26_md_main (vs baseline_x) | 139 | 0 | 0 | 40 | 3 | 7 | 13 | 2 | 0 |
| E_L26_random_main (vs baseline_x) | 148 | 0 | 0 | 33 | 4 | 3 | 11 | 2 | 0 |
| E_L26_recover (vs baseline_x_E) | 105 | 0 | 0 | 63 | 5 | 4 | 24 | 6 | 1 |
| E_L27_main (vs baseline_x) | 91 | 0 | 0 | 88 | 10 | 10 | 13 | 1 | 0 |
| E_L27_md_main (vs baseline_x) | 141 | 0 | 0 | 39 | 1 | 6 | 12 | 2 | 1 |
| E_L27_random_main (vs baseline_x) | 142 | 0 | 0 | 47 | 2 | 5 | 3 | 0 | 0 |
| E_L27_recover (vs baseline_x_E) | 116 | 0 | 0 | 50 | 2 | 6 | 26 | 4 | 2 |
| E_L28_main (vs baseline_x) | 120 | 0 | 0 | 46 | 1 | 4 | 26 | 4 | 1 |
| E_L28_md_main (vs baseline_x) | 150 | 0 | 0 | 31 | 2 | 3 | 11 | 3 | 1 |
| E_L28_random_main (vs baseline_x) | 161 | 0 | 0 | 27 | 4 | 3 | 4 | 1 | 0 |
| E_L28_recover (vs baseline_x_E) | 118 | 0 | 0 | 48 | 3 | 3 | 26 | 3 | 1 |
| E_L29_main (vs baseline_x) | 117 | 0 | 0 | 51 | 4 | 4 | 24 | 3 | 2 |
| E_L29_md_main (vs baseline_x) | 154 | 0 | 0 | 29 | 0 | 3 | 9 | 3 | 0 |
| E_L29_random_main (vs baseline_x) | 150 | 0 | 0 | 38 | 5 | 2 | 4 | 0 | 0 |
| E_L29_recover (vs baseline_x_E) | 115 | 0 | 0 | 50 | 3 | 2 | 27 | 1 | 2 |
| E_L2_main (vs baseline_x) | 9 | 0 | 0 | 32 | 6 | 3 | 151 | 45 | 23 |
| E_L2_md_main (vs baseline_x) | 179 | 0 | 0 | 8 | 0 | 1 | 5 | 1 | 0 |
| E_L2_random_main (vs baseline_x) | 136 | 0 | 0 | 44 | 4 | 6 | 12 | 1 | 2 |
| E_L2_recover (vs baseline_x_E) | 108 | 0 | 0 | 70 | 4 | 4 | 14 | 2 | 1 |
| E_L30_main (vs baseline_x) | 105 | 1 | 0 | 62 | 3 | 8 | 25 | 3 | 1 |
| E_L30_md_main (vs baseline_x) | 153 | 0 | 0 | 31 | 4 | 3 | 8 | 1 | 1 |
| E_L30_random_main (vs baseline_x) | 149 | 0 | 0 | 38 | 2 | 6 | 5 | 0 | 0 |
| E_L30_recover (vs baseline_x_E) | 122 | 0 | 0 | 45 | 5 | 2 | 25 | 4 | 2 |
| E_L31_main (vs baseline_x) | 118 | 0 | 0 | 51 | 4 | 7 | 23 | 3 | 1 |
| E_L31_md_main (vs baseline_x) | 128 | 0 | 0 | 58 | 5 | 4 | 6 | 0 | 1 |
| E_L31_random_main (vs baseline_x) | 151 | 0 | 0 | 36 | 2 | 4 | 5 | 0 | 0 |
| E_L31_recover (vs baseline_x_E) | 119 | 0 | 0 | 37 | 2 | 5 | 36 | 6 | 1 |
| E_L3_main (vs baseline_x) | 24 | 0 | 0 | 48 | 7 | 4 | 120 | 25 | 3 |
| E_L3_md_main (vs baseline_x) | 177 | 0 | 0 | 13 | 2 | 1 | 2 | 0 | 0 |
| E_L3_random_main (vs baseline_x) | 143 | 0 | 0 | 42 | 5 | 6 | 7 | 1 | 0 |
| E_L3_recover (vs baseline_x_E) | 88 | 0 | 0 | 95 | 6 | 10 | 9 | 2 | 0 |
| E_L4_main (vs baseline_x) | 10 | 0 | 0 | 48 | 7 | 3 | 134 | 22 | 11 |
| E_L4_md_main (vs baseline_x) | 178 | 0 | 0 | 10 | 2 | 0 | 4 | 0 | 1 |
| E_L4_random_main (vs baseline_x) | 125 | 0 | 0 | 58 | 4 | 6 | 9 | 1 | 0 |
| E_L4_recover (vs baseline_x_E) | 72 | 0 | 0 | 102 | 5 | 15 | 18 | 1 | 1 |
| E_L5_main (vs baseline_x) | 25 | 1 | 0 | 51 | 8 | 5 | 116 | 28 | 10 |
| E_L5_md_main (vs baseline_x) | 178 | 0 | 0 | 11 | 1 | 2 | 3 | 0 | 0 |
| E_L5_random_main (vs baseline_x) | 132 | 0 | 0 | 57 | 7 | 6 | 3 | 0 | 0 |
| E_L5_recover (vs baseline_x_E) | 88 | 0 | 0 | 98 | 5 | 15 | 6 | 1 | 1 |
| E_L6_main (vs baseline_x) | 20 | 0 | 0 | 59 | 9 | 7 | 113 | 22 | 12 |
| E_L6_md_main (vs baseline_x) | 178 | 0 | 0 | 9 | 2 | 0 | 5 | 0 | 0 |
| E_L6_random_main (vs baseline_x) | 133 | 0 | 0 | 50 | 8 | 8 | 9 | 0 | 0 |
| E_L6_recover (vs baseline_x_E) | 100 | 0 | 0 | 77 | 2 | 11 | 15 | 2 | 2 |
| E_L7_main (vs baseline_x) | 7 | 0 | 0 | 30 | 7 | 1 | 155 | 32 | 16 |
| E_L7_md_main (vs baseline_x) | 168 | 0 | 0 | 19 | 3 | 1 | 5 | 1 | 0 |
| E_L7_random_main (vs baseline_x) | 129 | 0 | 0 | 46 | 3 | 6 | 17 | 3 | 0 |
| E_L7_recover (vs baseline_x_E) | 89 | 0 | 0 | 83 | 6 | 14 | 20 | 2 | 2 |
| E_L8_main (vs baseline_x) | 13 | 0 | 0 | 31 | 7 | 4 | 148 | 24 | 11 |
| E_L8_md_main (vs baseline_x) | 174 | 0 | 0 | 15 | 4 | 0 | 3 | 1 | 0 |
| E_L8_random_main (vs baseline_x) | 123 | 0 | 0 | 54 | 4 | 10 | 15 | 1 | 1 |
| E_L8_recover (vs baseline_x_E) | 91 | 0 | 0 | 84 | 5 | 6 | 17 | 3 | 1 |
| E_L9_main (vs baseline_x) | 13 | 0 | 0 | 35 | 5 | 0 | 144 | 27 | 20 |
| E_L9_md_main (vs baseline_x) | 145 | 0 | 0 | 33 | 2 | 2 | 14 | 2 | 1 |
| E_L9_random_main (vs baseline_x) | 110 | 0 | 0 | 67 | 9 | 3 | 15 | 2 | 2 |
| E_L9_recover (vs baseline_x_E) | 66 | 0 | 0 | 108 | 8 | 19 | 18 | 2 | 2 |

## 3. by gold best answer with / without a negation word (n = 87 / 105). cell = answered-correct before→after, refusals before→after, wrong→right, right→wrong
| condition (vs its base) | gold has negation | gold has no negation |
|---|---|---|
| baseline_x_E (vs baseline_x) | 30→34, 10→16, 9, 2 | 48→55, 13→15, 7, 2 |
| C_L0_caut_to_conf (vs baseline_x_caut) | 31→25, 11→2, 2, 11 | 51→56, 16→2, 4, 8 |
| C_L0_conf_to_caut (vs baseline_x_conf) | 32→29, 2→1, 2, 4 | 56→55, 0→0, 5, 6 |
| C_L0_main (vs baseline_x) | 30→27, 10→2, 2, 8 | 48→54, 13→3, 6, 6 |
| C_L0_md_main (vs baseline_x) | 30→30, 10→10, 0, 0 | 48→49, 13→13, 1, 0 |
| C_L0_random_main (vs baseline_x) | 30→25, 10→12, 1, 5 | 48→48, 13→15, 4, 2 |
| C_L0_to_conf (vs baseline_x) | 30→26, 10→2, 2, 8 | 48→54, 13→3, 6, 6 |
| C_L10_caut_to_conf (vs baseline_x_caut) | 31→26, 11→10, 4, 7 | 51→51, 16→9, 4, 8 |
| C_L10_conf_to_caut (vs baseline_x_conf) | 32→28, 2→7, 5, 6 | 56→50, 0→10, 6, 5 |
| C_L10_main (vs baseline_x) | 30→25, 10→11, 4, 5 | 48→44, 13→11, 6, 10 |
| C_L10_md_main (vs baseline_x) | 30→30, 10→11, 0, 0 | 48→50, 13→14, 2, 0 |
| C_L10_random_main (vs baseline_x) | 30→28, 10→5, 3, 6 | 48→56, 13→6, 5, 4 |
| C_L10_to_conf (vs baseline_x) | 30→25, 10→11, 4, 5 | 48→44, 13→11, 6, 10 |
| C_L11_caut_to_conf (vs baseline_x_caut) | 31→20, 11→4, 4, 14 | 51→49, 16→6, 2, 11 |
| C_L11_conf_to_caut (vs baseline_x_conf) | 32→22, 2→3, 1, 9 | 56→49, 0→7, 5, 6 |
| C_L11_main (vs baseline_x) | 30→21, 10→4, 1, 9 | 48→53, 13→5, 7, 7 |
| C_L11_md_main (vs baseline_x) | 30→29, 10→11, 0, 1 | 48→48, 13→15, 1, 0 |
| C_L11_random_main (vs baseline_x) | 30→26, 10→9, 2, 7 | 48→49, 13→12, 6, 7 |
| C_L11_to_conf (vs baseline_x) | 30→22, 10→4, 1, 8 | 48→53, 13→5, 7, 8 |
| C_L12_caut_to_conf (vs baseline_x_caut) | 31→25, 11→5, 3, 9 | 51→58, 16→5, 7, 7 |
| C_L12_conf_to_caut (vs baseline_x_conf) | 32→25, 2→11, 5, 8 | 56→52, 0→14, 6, 1 |
| C_L12_main (vs baseline_x) | 30→25, 10→4, 1, 6 | 48→57, 13→5, 9, 6 |
| C_L12_md_main (vs baseline_x) | 30→31, 10→11, 1, 0 | 48→47, 13→17, 1, 0 |
| C_L12_random_main (vs baseline_x) | 30→32, 10→9, 3, 3 | 48→47, 13→17, 2, 2 |
| C_L12_to_conf (vs baseline_x) | 30→25, 10→4, 1, 5 | 48→60, 13→5, 12, 6 |
| C_L13_caut_to_conf (vs baseline_x_caut) | 31→31, 11→6, 5, 5 | 51→54, 16→6, 5, 8 |
| C_L13_conf_to_caut (vs baseline_x_conf) | 32→27, 2→11, 5, 7 | 56→52, 0→10, 5, 2 |
| C_L13_main (vs baseline_x) | 30→27, 10→13, 3, 3 | 48→54, 13→8, 7, 6 |
| C_L13_md_main (vs baseline_x) | 30→31, 10→11, 2, 1 | 48→48, 13→14, 2, 1 |
| C_L13_random_main (vs baseline_x) | 30→26, 10→11, 3, 6 | 48→51, 13→14, 6, 3 |
| C_L13_to_conf (vs baseline_x) | 30→27, 10→12, 3, 3 | 48→54, 13→8, 7, 6 |
| C_L14_caut_to_conf (vs baseline_x_caut) | 31→28, 11→12, 4, 7 | 51→43, 16→15, 3, 12 |
| C_L14_conf_to_caut (vs baseline_x_conf) | 32→27, 2→14, 3, 4 | 56→43, 0→21, 5, 8 |
| C_L14_main (vs baseline_x) | 30→29, 10→13, 6, 7 | 48→41, 13→21, 6, 9 |
| C_L14_md_main (vs baseline_x) | 30→31, 10→11, 2, 1 | 48→51, 13→13, 3, 0 |
| C_L14_random_main (vs baseline_x) | 30→28, 10→14, 2, 3 | 48→44, 13→19, 2, 5 |
| C_L14_to_conf (vs baseline_x) | 30→29, 10→13, 6, 7 | 48→42, 13→20, 6, 9 |
| C_L15_caut_to_conf (vs baseline_x_caut) | 31→25, 11→7, 1, 8 | 51→52, 16→8, 1, 4 |
| C_L15_conf_to_caut (vs baseline_x_conf) | 32→26, 2→2, 0, 4 | 56→50, 0→3, 2, 6 |
| C_L15_main (vs baseline_x) | 30→26, 10→11, 2, 4 | 48→50, 13→13, 5, 3 |
| C_L15_md_main (vs baseline_x) | 30→31, 10→11, 3, 2 | 48→52, 13→13, 4, 0 |
| C_L15_random_main (vs baseline_x) | 30→28, 10→11, 3, 5 | 48→50, 13→14, 5, 3 |
| C_L15_to_conf (vs baseline_x) | 30→28, 10→9, 2, 4 | 48→48, 13→12, 3, 3 |
| C_L16_caut_to_conf (vs baseline_x_caut) | 31→25, 11→7, 2, 9 | 51→50, 16→9, 3, 8 |
| C_L16_conf_to_caut (vs baseline_x_conf) | 32→23, 2→7, 2, 6 | 56→50, 0→14, 4, 4 |
| C_L16_main (vs baseline_x) | 30→27, 10→9, 2, 5 | 48→50, 13→11, 7, 6 |
| C_L16_md_main (vs baseline_x) | 30→31, 10→10, 2, 2 | 48→50, 13→11, 2, 1 |
| C_L16_random_main (vs baseline_x) | 30→26, 10→9, 1, 5 | 48→54, 13→12, 7, 1 |
| C_L16_to_conf (vs baseline_x) | 30→26, 10→10, 2, 5 | 48→48, 13→12, 6, 6 |
| C_L17_caut_to_conf (vs baseline_x_caut) | 31→23, 11→11, 1, 8 | 51→51, 16→14, 4, 5 |
| C_L17_conf_to_caut (vs baseline_x_conf) | 32→23, 2→6, 1, 5 | 56→53, 0→13, 7, 4 |
| C_L17_main (vs baseline_x) | 30→25, 10→14, 2, 6 | 48→49, 13→19, 6, 4 |
| C_L17_md_main (vs baseline_x) | 30→30, 10→10, 2, 2 | 48→53, 13→10, 5, 3 |
| C_L17_random_main (vs baseline_x) | 30→27, 10→10, 1, 4 | 48→53, 13→13, 6, 2 |
| C_L17_to_conf (vs baseline_x) | 30→28, 10→11, 2, 5 | 48→50, 13→16, 7, 4 |
| C_L18_caut_to_conf (vs baseline_x_caut) | 31→26, 11→12, 3, 7 | 51→50, 16→14, 4, 7 |
| C_L18_conf_to_caut (vs baseline_x_conf) | 32→33, 2→3, 5, 2 | 56→52, 0→7, 5, 6 |
| C_L18_main (vs baseline_x) | 30→26, 10→15, 2, 4 | 48→47, 13→19, 5, 6 |
| C_L18_md_main (vs baseline_x) | 30→30, 10→10, 2, 2 | 48→54, 13→11, 5, 2 |
| C_L18_random_main (vs baseline_x) | 30→28, 10→11, 1, 3 | 48→52, 13→14, 4, 0 |
| C_L18_to_conf (vs baseline_x) | 30→25, 10→15, 1, 4 | 48→47, 13→20, 5, 5 |
| C_L19_caut_to_conf (vs baseline_x_caut) | 31→31, 11→11, 3, 3 | 51→50, 16→17, 3, 5 |
| C_L19_conf_to_caut (vs baseline_x_conf) | 32→28, 2→3, 3, 4 | 56→60, 0→2, 9, 3 |
| C_L19_main (vs baseline_x) | 30→27, 10→16, 2, 3 | 48→49, 13→26, 6, 2 |
| C_L19_md_main (vs baseline_x) | 30→30, 10→10, 2, 2 | 48→51, 13→10, 3, 3 |
| C_L19_random_main (vs baseline_x) | 30→29, 10→13, 2, 2 | 48→48, 13→13, 2, 3 |
| C_L19_to_conf (vs baseline_x) | 30→28, 10→12, 1, 2 | 48→48, 13→21, 3, 2 |
| C_L1_caut_to_conf (vs baseline_x_caut) | 31→31, 11→6, 5, 9 | 51→53, 16→6, 7, 10 |
| C_L1_conf_to_caut (vs baseline_x_conf) | 32→26, 2→0, 2, 8 | 56→62, 0→0, 11, 5 |
| C_L1_main (vs baseline_x) | 30→31, 10→5, 4, 4 | 48→51, 13→7, 9, 9 |
| C_L1_md_main (vs baseline_x) | 30→30, 10→10, 1, 1 | 48→49, 13→13, 1, 0 |
| C_L1_random_main (vs baseline_x) | 30→27, 10→10, 2, 6 | 48→50, 13→12, 5, 3 |
| C_L1_to_conf (vs baseline_x) | 30→33, 10→5, 4, 3 | 48→50, 13→7, 8, 9 |
| C_L20_caut_to_conf (vs baseline_x_caut) | 31→28, 11→10, 2, 4 | 51→51, 16→15, 4, 5 |
| C_L20_conf_to_caut (vs baseline_x_conf) | 32→30, 2→1, 2, 3 | 56→54, 0→2, 3, 3 |
| C_L20_main (vs baseline_x) | 30→25, 10→12, 1, 5 | 48→48, 13→18, 6, 5 |
| C_L20_md_main (vs baseline_x) | 30→28, 10→10, 1, 3 | 48→54, 13→11, 5, 2 |
| C_L20_random_main (vs baseline_x) | 30→27, 10→10, 1, 4 | 48→56, 13→13, 7, 1 |
| C_L20_to_conf (vs baseline_x) | 30→26, 10→12, 2, 5 | 48→52, 13→15, 7, 3 |
| C_L21_caut_to_conf (vs baseline_x_caut) | 31→28, 11→12, 3, 5 | 51→44, 16→19, 0, 7 |
| C_L21_conf_to_caut (vs baseline_x_conf) | 32→26, 2→4, 2, 5 | 56→54, 0→5, 5, 4 |
| C_L21_main (vs baseline_x) | 30→28, 10→14, 2, 3 | 48→45, 13→28, 6, 5 |
| C_L21_md_main (vs baseline_x) | 30→27, 10→10, 0, 3 | 48→49, 13→11, 3, 4 |
| C_L21_random_main (vs baseline_x) | 30→29, 10→10, 2, 3 | 48→49, 13→14, 4, 2 |
| C_L21_to_conf (vs baseline_x) | 30→24, 10→13, 0, 5 | 48→44, 13→24, 5, 6 |
| C_L22_caut_to_conf (vs baseline_x_caut) | 31→28, 11→10, 3, 5 | 51→51, 16→15, 4, 5 |
| C_L22_conf_to_caut (vs baseline_x_conf) | 32→27, 2→5, 3, 4 | 56→53, 0→5, 5, 4 |
| C_L22_main (vs baseline_x) | 30→25, 10→14, 2, 6 | 48→48, 13→26, 7, 4 |
| C_L22_md_main (vs baseline_x) | 30→29, 10→10, 1, 2 | 48→50, 13→13, 4, 3 |
| C_L22_random_main (vs baseline_x) | 30→27, 10→11, 0, 3 | 48→50, 13→16, 6, 3 |
| C_L22_to_conf (vs baseline_x) | 30→27, 10→11, 3, 5 | 48→53, 13→18, 9, 3 |
| C_L23_caut_to_conf (vs baseline_x_caut) | 31→30, 11→10, 2, 2 | 51→49, 16→15, 2, 5 |
| C_L23_conf_to_caut (vs baseline_x_conf) | 32→30, 2→1, 3, 4 | 56→54, 0→1, 4, 5 |
| C_L23_main (vs baseline_x) | 30→25, 10→12, 2, 5 | 48→48, 13→23, 7, 4 |
| C_L23_md_main (vs baseline_x) | 30→28, 10→11, 1, 3 | 48→48, 13→13, 1, 2 |
| C_L23_random_main (vs baseline_x) | 30→28, 10→11, 2, 2 | 48→46, 13→15, 2, 2 |
| C_L23_to_conf (vs baseline_x) | 30→25, 10→11, 2, 6 | 48→51, 13→18, 6, 3 |
| C_L24_caut_to_conf (vs baseline_x_caut) | 31→30, 11→10, 2, 3 | 51→50, 16→14, 2, 5 |
| C_L24_conf_to_caut (vs baseline_x_conf) | 32→30, 2→2, 4, 4 | 56→53, 0→6, 2, 4 |
| C_L24_main (vs baseline_x) | 30→25, 10→16, 2, 4 | 48→50, 13→26, 7, 2 |
| C_L24_md_main (vs baseline_x) | 30→27, 10→10, 1, 4 | 48→49, 13→11, 1, 2 |
| C_L24_random_main (vs baseline_x) | 30→28, 10→10, 1, 2 | 48→48, 13→14, 4, 4 |
| C_L24_to_conf (vs baseline_x) | 30→24, 10→13, 0, 4 | 48→50, 13→19, 4, 2 |
| C_L25_caut_to_conf (vs baseline_x_caut) | 31→29, 11→10, 1, 3 | 51→52, 16→17, 2, 3 |
| C_L25_conf_to_caut (vs baseline_x_conf) | 32→29, 2→2, 3, 5 | 56→55, 0→2, 4, 4 |
| C_L25_main (vs baseline_x) | 30→24, 10→15, 1, 4 | 48→47, 13→25, 6, 4 |
| C_L25_md_main (vs baseline_x) | 30→28, 10→10, 1, 3 | 48→49, 13→10, 0, 1 |
| C_L25_random_main (vs baseline_x) | 30→30, 10→10, 2, 1 | 48→50, 13→15, 3, 0 |
| C_L25_to_conf (vs baseline_x) | 30→26, 10→12, 1, 3 | 48→51, 13→18, 4, 1 |
| C_L26_caut_to_conf (vs baseline_x_caut) | 31→28, 11→12, 2, 3 | 51→50, 16→19, 3, 6 |
| C_L26_conf_to_caut (vs baseline_x_conf) | 32→29, 2→2, 2, 3 | 56→57, 0→4, 7, 5 |
| C_L26_main (vs baseline_x) | 30→23, 10→21, 1, 4 | 48→45, 13→28, 6, 6 |
| C_L26_md_main (vs baseline_x) | 30→29, 10→9, 1, 3 | 48→53, 13→9, 4, 1 |
| C_L26_random_main (vs baseline_x) | 30→32, 10→11, 2, 1 | 48→50, 13→14, 5, 2 |
| C_L26_to_conf (vs baseline_x) | 30→26, 10→12, 1, 3 | 48→50, 13→21, 8, 4 |
| C_L27_caut_to_conf (vs baseline_x_caut) | 31→29, 11→10, 2, 3 | 51→51, 16→13, 1, 3 |
| C_L27_conf_to_caut (vs baseline_x_conf) | 32→26, 2→1, 1, 5 | 56→58, 0→2, 6, 4 |
| C_L27_main (vs baseline_x) | 30→24, 10→16, 1, 3 | 48→46, 13→26, 5, 4 |
| C_L27_md_main (vs baseline_x) | 30→29, 10→9, 1, 3 | 48→50, 13→10, 1, 1 |
| C_L27_random_main (vs baseline_x) | 30→26, 10→10, 0, 4 | 48→49, 13→13, 2, 1 |
| C_L27_to_conf (vs baseline_x) | 30→27, 10→12, 2, 3 | 48→53, 13→15, 6, 2 |
| C_L28_caut_to_conf (vs baseline_x_caut) | 31→27, 11→10, 1, 4 | 51→52, 16→12, 2, 3 |
| C_L28_conf_to_caut (vs baseline_x_conf) | 32→26, 2→2, 2, 7 | 56→58, 0→0, 4, 2 |
| C_L28_main (vs baseline_x) | 30→23, 10→16, 1, 4 | 48→47, 13→27, 6, 4 |
| C_L28_md_main (vs baseline_x) | 30→30, 10→7, 2, 3 | 48→50, 13→10, 1, 1 |
| C_L28_random_main (vs baseline_x) | 30→29, 10→11, 1, 2 | 48→52, 13→13, 4, 1 |
| C_L28_to_conf (vs baseline_x) | 30→26, 10→12, 2, 3 | 48→51, 13→14, 4, 1 |
| C_L29_caut_to_conf (vs baseline_x_caut) | 31→28, 11→10, 3, 5 | 51→49, 16→16, 0, 5 |
| C_L29_conf_to_caut (vs baseline_x_conf) | 32→27, 2→1, 2, 6 | 56→57, 0→5, 6, 3 |
| C_L29_main (vs baseline_x) | 30→26, 10→21, 1, 2 | 48→45, 13→31, 5, 3 |
| C_L29_md_main (vs baseline_x) | 30→30, 10→9, 2, 2 | 48→50, 13→10, 1, 1 |
| C_L29_random_main (vs baseline_x) | 30→28, 10→10, 1, 2 | 48→51, 13→13, 3, 0 |
| C_L29_to_conf (vs baseline_x) | 30→25, 10→13, 2, 4 | 48→52, 13→18, 5, 2 |
| C_L2_caut_to_conf (vs baseline_x_caut) | 31→24, 11→1, 2, 10 | 51→54, 16→6, 6, 7 |
| C_L2_conf_to_caut (vs baseline_x_conf) | 32→21, 2→0, 1, 11 | 56→51, 0→0, 8, 13 |
| C_L2_main (vs baseline_x) | 30→26, 10→1, 3, 8 | 48→56, 13→5, 8, 3 |
| C_L2_md_main (vs baseline_x) | 30→30, 10→10, 0, 0 | 48→48, 13→13, 0, 0 |
| C_L2_random_main (vs baseline_x) | 30→30, 10→10, 3, 3 | 48→47, 13→12, 4, 5 |
| C_L2_to_conf (vs baseline_x) | 30→27, 10→1, 5, 9 | 48→56, 13→5, 8, 3 |
| C_L30_caut_to_conf (vs baseline_x_caut) | 31→27, 11→11, 0, 3 | 51→49, 16→15, 1, 5 |
| C_L30_conf_to_caut (vs baseline_x_conf) | 32→29, 2→1, 4, 6 | 56→57, 0→2, 6, 3 |
| C_L30_main (vs baseline_x) | 30→25, 10→18, 2, 4 | 48→47, 13→30, 4, 2 |
| C_L30_md_main (vs baseline_x) | 30→31, 10→9, 2, 1 | 48→50, 13→9, 1, 1 |
| C_L30_random_main (vs baseline_x) | 30→26, 10→10, 1, 4 | 48→50, 13→13, 2, 1 |
| C_L30_to_conf (vs baseline_x) | 30→24, 10→13, 1, 4 | 48→51, 13→16, 4, 2 |
| C_L31_caut_to_conf (vs baseline_x_caut) | 31→31, 11→2, 2, 6 | 51→62, 16→0, 2, 2 |
| C_L31_conf_to_caut (vs baseline_x_conf) | 32→30, 2→2, 2, 4 | 56→58, 0→1, 4, 1 |
| C_L31_main (vs baseline_x) | 30→27, 10→8, 1, 3 | 48→57, 13→6, 4, 2 |
| C_L31_md_main (vs baseline_x) | 30→31, 10→7, 1, 2 | 48→58, 13→2, 4, 1 |
| C_L31_random_main (vs baseline_x) | 30→29, 10→10, 0, 1 | 48→48, 13→14, 2, 1 |
| C_L31_to_conf (vs baseline_x) | 30→31, 10→2, 2, 4 | 48→60, 13→1, 3, 0 |
| C_L3_caut_to_conf (vs baseline_x_caut) | 31→24, 11→5, 1, 10 | 51→52, 16→6, 5, 8 |
| C_L3_conf_to_caut (vs baseline_x_conf) | 32→26, 2→1, 2, 8 | 56→54, 0→2, 7, 8 |
| C_L3_main (vs baseline_x) | 30→26, 10→6, 2, 5 | 48→47, 13→8, 4, 7 |
| C_L3_md_main (vs baseline_x) | 30→30, 10→11, 0, 0 | 48→50, 13→13, 2, 0 |
| C_L3_random_main (vs baseline_x) | 30→31, 10→9, 3, 3 | 48→53, 13→13, 4, 1 |
| C_L3_to_conf (vs baseline_x) | 30→25, 10→6, 1, 6 | 48→47, 13→9, 5, 7 |
| C_L4_caut_to_conf (vs baseline_x_caut) | 31→23, 11→6, 2, 12 | 51→49, 16→6, 7, 14 |
| C_L4_conf_to_caut (vs baseline_x_conf) | 32→29, 2→1, 2, 6 | 56→56, 0→0, 6, 6 |
| C_L4_main (vs baseline_x) | 30→24, 10→7, 2, 9 | 48→50, 13→7, 8, 9 |
| C_L4_md_main (vs baseline_x) | 30→30, 10→10, 0, 0 | 48→48, 13→13, 0, 0 |
| C_L4_random_main (vs baseline_x) | 30→26, 10→10, 1, 4 | 48→51, 13→16, 6, 2 |
| C_L4_to_conf (vs baseline_x) | 30→24, 10→7, 2, 9 | 48→48, 13→7, 7, 10 |
| C_L5_caut_to_conf (vs baseline_x_caut) | 31→30, 11→2, 3, 8 | 51→54, 16→7, 7, 11 |
| C_L5_conf_to_caut (vs baseline_x_conf) | 32→20, 2→24, 2, 6 | 56→43, 0→33, 9, 7 |
| C_L5_main (vs baseline_x) | 30→28, 10→6, 1, 5 | 48→52, 13→9, 8, 9 |
| C_L5_md_main (vs baseline_x) | 30→30, 10→10, 1, 1 | 48→49, 13→13, 1, 0 |
| C_L5_random_main (vs baseline_x) | 30→28, 10→10, 3, 4 | 48→50, 13→12, 4, 2 |
| C_L5_to_conf (vs baseline_x) | 30→26, 10→4, 1, 8 | 48→52, 13→9, 8, 8 |
| C_L6_caut_to_conf (vs baseline_x_caut) | 31→26, 11→3, 3, 11 | 51→53, 16→4, 3, 8 |
| C_L6_conf_to_caut (vs baseline_x_conf) | 32→30, 2→1, 4, 7 | 56→54, 0→2, 4, 5 |
| C_L6_main (vs baseline_x) | 30→27, 10→3, 3, 7 | 48→52, 13→6, 6, 7 |
| C_L6_md_main (vs baseline_x) | 30→29, 10→10, 0, 1 | 48→48, 13→13, 0, 0 |
| C_L6_random_main (vs baseline_x) | 30→32, 10→10, 6, 5 | 48→49, 13→8, 4, 6 |
| C_L6_to_conf (vs baseline_x) | 30→27, 10→3, 3, 7 | 48→52, 13→6, 5, 6 |
| C_L7_caut_to_conf (vs baseline_x_caut) | 31→29, 11→2, 4, 9 | 51→52, 16→3, 3, 8 |
| C_L7_conf_to_caut (vs baseline_x_conf) | 32→31, 2→1, 3, 5 | 56→52, 0→2, 5, 8 |
| C_L7_main (vs baseline_x) | 30→27, 10→3, 1, 7 | 48→55, 13→4, 5, 4 |
| C_L7_md_main (vs baseline_x) | 30→29, 10→11, 1, 2 | 48→51, 13→14, 3, 0 |
| C_L7_random_main (vs baseline_x) | 30→26, 10→10, 2, 5 | 48→48, 13→16, 3, 2 |
| C_L7_to_conf (vs baseline_x) | 30→26, 10→3, 1, 8 | 48→54, 13→4, 5, 5 |
| C_L8_caut_to_conf (vs baseline_x_caut) | 31→26, 11→10, 4, 10 | 51→51, 16→7, 4, 9 |
| C_L8_conf_to_caut (vs baseline_x_conf) | 32→30, 2→7, 5, 5 | 56→52, 0→3, 4, 5 |
| C_L8_main (vs baseline_x) | 30→25, 10→10, 4, 10 | 48→51, 13→12, 4, 5 |
| C_L8_md_main (vs baseline_x) | 30→30, 10→10, 0, 0 | 48→49, 13→13, 1, 0 |
| C_L8_random_main (vs baseline_x) | 30→26, 10→11, 3, 6 | 48→46, 13→12, 5, 8 |
| C_L8_to_conf (vs baseline_x) | 30→25, 10→11, 4, 9 | 48→50, 13→12, 5, 7 |
| C_L9_caut_to_conf (vs baseline_x_caut) | 31→29, 11→10, 4, 8 | 51→53, 16→10, 6, 7 |
| C_L9_conf_to_caut (vs baseline_x_conf) | 32→24, 2→25, 5, 6 | 56→44, 0→21, 7, 3 |
| C_L9_main (vs baseline_x) | 30→29, 10→14, 5, 5 | 48→51, 13→13, 8, 5 |
| C_L9_md_main (vs baseline_x) | 30→30, 10→11, 0, 0 | 48→50, 13→14, 2, 1 |
| C_L9_random_main (vs baseline_x) | 30→29, 10→17, 6, 4 | 48→49, 13→20, 5, 1 |
| C_L9_to_conf (vs baseline_x) | 30→29, 10→14, 5, 5 | 48→51, 13→14, 8, 5 |
| E_L0_main (vs baseline_x) | 30→66, 10→1, 32, 4 | 48→63, 13→2, 24, 19 |
| E_L0_md_main (vs baseline_x) | 30→29, 10→10, 0, 1 | 48→50, 13→13, 2, 0 |
| E_L0_random_main (vs baseline_x) | 30→29, 10→11, 1, 2 | 48→47, 13→15, 1, 1 |
| E_L0_recover (vs baseline_x_E) | 34→34, 16→15, 4, 5 | 55→53, 15→16, 5, 6 |
| E_L10_main (vs baseline_x) | 30→44, 10→2, 14, 8 | 48→71, 13→2, 22, 9 |
| E_L10_md_main (vs baseline_x) | 30→31, 10→11, 1, 0 | 48→51, 13→13, 2, 0 |
| E_L10_random_main (vs baseline_x) | 30→31, 10→5, 4, 4 | 48→58, 13→5, 6, 4 |
| E_L10_recover (vs baseline_x_E) | 34→28, 16→21, 3, 8 | 55→56, 15→16, 7, 4 |
| E_L11_main (vs baseline_x) | 30→38, 10→18, 11, 2 | 48→57, 13→21, 16, 6 |
| E_L11_md_main (vs baseline_x) | 30→29, 10→12, 0, 1 | 48→52, 13→13, 3, 0 |
| E_L11_random_main (vs baseline_x) | 30→26, 10→9, 2, 7 | 48→49, 13→12, 6, 7 |
| E_L11_recover (vs baseline_x_E) | 34→27, 16→25, 2, 7 | 55→56, 15→24, 10, 3 |
| E_L12_main (vs baseline_x) | 30→38, 10→9, 12, 7 | 48→58, 13→9, 18, 11 |
| E_L12_md_main (vs baseline_x) | 30→31, 10→11, 2, 1 | 48→51, 13→13, 2, 0 |
| E_L12_random_main (vs baseline_x) | 30→32, 10→9, 3, 3 | 48→48, 13→17, 4, 3 |
| E_L12_recover (vs baseline_x_E) | 34→32, 16→15, 3, 7 | 55→51, 15→16, 5, 8 |
| E_L13_main (vs baseline_x) | 30→30, 10→5, 5, 8 | 48→54, 13→6, 13, 11 |
| E_L13_md_main (vs baseline_x) | 30→33, 10→12, 4, 0 | 48→52, 13→13, 4, 0 |
| E_L13_random_main (vs baseline_x) | 30→28, 10→11, 4, 5 | 48→50, 13→14, 6, 4 |
| E_L13_recover (vs baseline_x_E) | 34→33, 16→13, 2, 5 | 55→52, 15→12, 8, 11 |
| E_L14_main (vs baseline_x) | 30→35, 10→4, 8, 5 | 48→63, 13→4, 16, 10 |
| E_L14_md_main (vs baseline_x) | 30→32, 10→14, 4, 1 | 48→53, 13→13, 5, 0 |
| E_L14_random_main (vs baseline_x) | 30→30, 10→13, 2, 2 | 48→46, 13→18, 3, 4 |
| E_L14_recover (vs baseline_x_E) | 34→38, 16→14, 4, 4 | 55→57, 15→14, 8, 6 |
| E_L15_main (vs baseline_x) | 30→35, 10→10, 9, 6 | 48→59, 13→11, 14, 6 |
| E_L15_md_main (vs baseline_x) | 30→31, 10→13, 5, 4 | 48→51, 13→15, 7, 2 |
| E_L15_random_main (vs baseline_x) | 30→27, 10→11, 3, 6 | 48→49, 13→15, 5, 3 |
| E_L15_recover (vs baseline_x_E) | 34→38, 16→17, 5, 3 | 55→52, 15→18, 6, 8 |
| E_L16_main (vs baseline_x) | 30→29, 10→11, 7, 8 | 48→59, 13→10, 15, 6 |
| E_L16_md_main (vs baseline_x) | 30→32, 10→13, 7, 5 | 48→53, 13→15, 8, 1 |
| E_L16_random_main (vs baseline_x) | 30→26, 10→9, 1, 5 | 48→54, 13→13, 8, 1 |
| E_L16_recover (vs baseline_x_E) | 34→33, 16→18, 3, 5 | 55→54, 15→18, 9, 8 |
| E_L17_main (vs baseline_x) | 30→31, 10→10, 6, 7 | 48→59, 13→10, 15, 5 |
| E_L17_md_main (vs baseline_x) | 30→28, 10→13, 6, 8 | 48→55, 13→14, 7, 1 |
| E_L17_random_main (vs baseline_x) | 30→27, 10→10, 1, 4 | 48→52, 13→13, 7, 4 |
| E_L17_recover (vs baseline_x_E) | 34→34, 16→16, 4, 5 | 55→56, 15→15, 5, 4 |
| E_L18_main (vs baseline_x) | 30→34, 10→12, 8, 4 | 48→61, 13→10, 18, 7 |
| E_L18_md_main (vs baseline_x) | 30→30, 10→12, 6, 6 | 48→55, 13→14, 8, 1 |
| E_L18_random_main (vs baseline_x) | 30→28, 10→10, 1, 4 | 48→51, 13→14, 4, 1 |
| E_L18_recover (vs baseline_x_E) | 34→34, 16→18, 3, 4 | 55→55, 15→16, 6, 4 |
| E_L19_main (vs baseline_x) | 30→30, 10→13, 5, 4 | 48→61, 13→11, 14, 3 |
| E_L19_md_main (vs baseline_x) | 30→28, 10→13, 4, 6 | 48→52, 13→15, 7, 2 |
| E_L19_random_main (vs baseline_x) | 30→28, 10→13, 2, 2 | 48→48, 13→13, 2, 3 |
| E_L19_recover (vs baseline_x_E) | 34→32, 16→18, 0, 3 | 55→60, 15→15, 8, 2 |
| E_L1_main (vs baseline_x) | 30→47, 10→3, 21, 7 | 48→64, 13→3, 21, 15 |
| E_L1_md_main (vs baseline_x) | 30→29, 10→10, 0, 1 | 48→49, 13→13, 1, 0 |
| E_L1_random_main (vs baseline_x) | 30→29, 10→10, 2, 3 | 48→52, 13→11, 4, 2 |
| E_L1_recover (vs baseline_x_E) | 34→32, 16→20, 2, 3 | 55→53, 15→15, 5, 5 |
| E_L20_main (vs baseline_x) | 30→33, 10→12, 6, 4 | 48→58, 13→15, 13, 2 |
| E_L20_md_main (vs baseline_x) | 30→31, 10→12, 6, 5 | 48→49, 13→16, 5, 2 |
| E_L20_random_main (vs baseline_x) | 30→27, 10→10, 1, 4 | 48→57, 13→13, 7, 0 |
| E_L20_recover (vs baseline_x_E) | 34→35, 16→17, 5, 4 | 55→58, 15→20, 8, 2 |
| E_L21_main (vs baseline_x) | 30→29, 10→17, 9, 5 | 48→60, 13→16, 18, 6 |
| E_L21_md_main (vs baseline_x) | 30→29, 10→12, 4, 5 | 48→48, 13→16, 6, 4 |
| E_L21_random_main (vs baseline_x) | 30→28, 10→10, 1, 3 | 48→48, 13→14, 4, 3 |
| E_L21_recover (vs baseline_x_E) | 34→31, 16→18, 4, 6 | 55→57, 15→18, 7, 2 |
| E_L22_main (vs baseline_x) | 30→28, 10→17, 6, 4 | 48→52, 13→15, 11, 5 |
| E_L22_md_main (vs baseline_x) | 30→27, 10→12, 2, 5 | 48→48, 13→15, 6, 5 |
| E_L22_random_main (vs baseline_x) | 30→28, 10→11, 0, 2 | 48→48, 13→16, 4, 3 |
| E_L22_recover (vs baseline_x_E) | 34→31, 16→22, 3, 4 | 55→57, 15→18, 7, 2 |
| E_L23_main (vs baseline_x) | 30→29, 10→14, 5, 5 | 48→54, 13→20, 13, 3 |
| E_L23_md_main (vs baseline_x) | 30→28, 10→13, 2, 3 | 48→46, 13→15, 6, 6 |
| E_L23_random_main (vs baseline_x) | 30→30, 10→10, 2, 2 | 48→47, 13→14, 2, 2 |
| E_L23_recover (vs baseline_x_E) | 34→33, 16→22, 3, 2 | 55→53, 15→23, 8, 5 |
| E_L24_main (vs baseline_x) | 30→26, 10→16, 2, 2 | 48→51, 13→20, 11, 4 |
| E_L24_md_main (vs baseline_x) | 30→29, 10→11, 3, 3 | 48→44, 13→14, 4, 7 |
| E_L24_random_main (vs baseline_x) | 30→29, 10→10, 1, 1 | 48→52, 13→12, 4, 2 |
| E_L24_recover (vs baseline_x_E) | 34→33, 16→21, 4, 3 | 55→53, 15→22, 9, 6 |
| E_L25_main (vs baseline_x) | 30→24, 10→15, 2, 4 | 48→47, 13→18, 11, 9 |
| E_L25_md_main (vs baseline_x) | 30→30, 10→11, 3, 2 | 48→49, 13→14, 6, 4 |
| E_L25_random_main (vs baseline_x) | 30→30, 10→10, 2, 1 | 48→50, 13→15, 4, 1 |
| E_L25_recover (vs baseline_x_E) | 34→33, 16→19, 3, 4 | 55→54, 15→20, 7, 4 |
| E_L26_main (vs baseline_x) | 30→28, 10→15, 2, 2 | 48→48, 13→18, 9, 7 |
| E_L26_md_main (vs baseline_x) | 30→28, 10→11, 1, 2 | 48→46, 13→15, 4, 5 |
| E_L26_random_main (vs baseline_x) | 30→30, 10→10, 1, 1 | 48→51, 13→14, 5, 2 |
| E_L26_recover (vs baseline_x_E) | 34→36, 16→17, 6, 3 | 55→55, 15→20, 5, 2 |
| E_L27_main (vs baseline_x) | 30→29, 10→15, 3, 2 | 48→48, 13→14, 8, 8 |
| E_L27_md_main (vs baseline_x) | 30→25, 10→13, 1, 4 | 48→46, 13→15, 2, 3 |
| E_L27_random_main (vs baseline_x) | 30→26, 10→10, 0, 4 | 48→49, 13→13, 2, 1 |
| E_L27_recover (vs baseline_x_E) | 34→30, 16→19, 1, 4 | 55→53, 15→19, 5, 4 |
| E_L28_main (vs baseline_x) | 30→27, 10→14, 1, 2 | 48→48, 13→16, 4, 3 |
| E_L28_md_main (vs baseline_x) | 30→26, 10→13, 0, 2 | 48→49, 13→16, 5, 2 |
| E_L28_random_main (vs baseline_x) | 30→29, 10→11, 1, 2 | 48→52, 13→13, 4, 1 |
| E_L28_recover (vs baseline_x_E) | 34→32, 16→18, 2, 3 | 55→56, 15→16, 4, 1 |
| E_L29_main (vs baseline_x) | 30→28, 10→14, 2, 2 | 48→48, 13→14, 5, 4 |
| E_L29_md_main (vs baseline_x) | 30→26, 10→12, 0, 2 | 48→49, 13→16, 3, 1 |
| E_L29_random_main (vs baseline_x) | 30→28, 10→10, 1, 2 | 48→52, 13→13, 4, 0 |
| E_L29_recover (vs baseline_x_E) | 34→35, 16→16, 3, 2 | 55→53, 15→14, 1, 2 |
| E_L2_main (vs baseline_x) | 30→51, 10→4, 23, 8 | 48→66, 13→4, 28, 18 |
| E_L2_md_main (vs baseline_x) | 30→29, 10→11, 0, 1 | 48→48, 13→14, 1, 0 |
| E_L2_random_main (vs baseline_x) | 30→29, 10→11, 3, 3 | 48→46, 13→12, 2, 5 |
| E_L2_recover (vs baseline_x_E) | 34→33, 16→19, 1, 1 | 55→55, 15→12, 5, 4 |
| E_L30_main (vs baseline_x) | 30→27, 10→11, 4, 6 | 48→47, 13→14, 3, 3 |
| E_L30_md_main (vs baseline_x) | 30→27, 10→12, 1, 3 | 48→50, 13→16, 4, 1 |
| E_L30_random_main (vs baseline_x) | 30→25, 10→10, 0, 5 | 48→50, 13→13, 2, 1 |
| E_L30_recover (vs baseline_x_E) | 34→35, 16→16, 3, 2 | 55→58, 15→15, 6, 2 |
| E_L31_main (vs baseline_x) | 30→30, 10→11, 4, 3 | 48→46, 13→14, 3, 5 |
| E_L31_md_main (vs baseline_x) | 30→28, 10→12, 0, 3 | 48→49, 13→18, 5, 2 |
| E_L31_random_main (vs baseline_x) | 30→27, 10→10, 0, 2 | 48→46, 13→15, 2, 2 |
| E_L31_recover (vs baseline_x_E) | 34→34, 16→20, 4, 3 | 55→53, 15→17, 4, 3 |
| E_L3_main (vs baseline_x) | 30→42, 10→10, 13, 4 | 48→69, 13→8, 19, 3 |
| E_L3_md_main (vs baseline_x) | 30→30, 10→10, 0, 0 | 48→50, 13→13, 2, 1 |
| E_L3_random_main (vs baseline_x) | 30→29, 10→10, 3, 4 | 48→51, 13→13, 3, 2 |
| E_L3_recover (vs baseline_x_E) | 34→34, 16→18, 2, 3 | 55→53, 15→16, 6, 7 |
| E_L4_main (vs baseline_x) | 30→41, 10→6, 12, 6 | 48→64, 13→4, 17, 8 |
| E_L4_md_main (vs baseline_x) | 30→30, 10→10, 0, 1 | 48→50, 13→13, 2, 0 |
| E_L4_random_main (vs baseline_x) | 30→26, 10→10, 1, 4 | 48→50, 13→15, 4, 2 |
| E_L4_recover (vs baseline_x_E) | 34→31, 16→14, 1, 7 | 55→48, 15→14, 5, 9 |
| E_L5_main (vs baseline_x) | 30→40, 10→10, 15, 7 | 48→67, 13→9, 22, 8 |
| E_L5_md_main (vs baseline_x) | 30→29, 10→10, 0, 1 | 48→47, 13→14, 1, 1 |
| E_L5_random_main (vs baseline_x) | 30→28, 10→10, 2, 3 | 48→50, 13→12, 5, 3 |
| E_L5_recover (vs baseline_x_E) | 34→27, 16→21, 2, 8 | 55→50, 15→19, 4, 8 |
| E_L6_main (vs baseline_x) | 30→35, 10→9, 13, 10 | 48→64, 13→6, 18, 9 |
| E_L6_md_main (vs baseline_x) | 30→31, 10→10, 1, 0 | 48→49, 13→13, 1, 0 |
| E_L6_random_main (vs baseline_x) | 30→29, 10→9, 3, 4 | 48→52, 13→9, 5, 4 |
| E_L6_recover (vs baseline_x_E) | 34→30, 16→17, 1, 7 | 55→50, 15→14, 3, 6 |
| E_L7_main (vs baseline_x) | 30→48, 10→4, 18, 6 | 48→65, 13→2, 21, 11 |
| E_L7_md_main (vs baseline_x) | 30→32, 10→11, 3, 1 | 48→49, 13→14, 1, 0 |
| E_L7_random_main (vs baseline_x) | 30→28, 10→10, 3, 4 | 48→47, 13→17, 3, 2 |
| E_L7_recover (vs baseline_x_E) | 34→32, 16→21, 3, 6 | 55→49, 15→14, 5, 10 |
| E_L8_main (vs baseline_x) | 30→39, 10→2, 11, 7 | 48→69, 13→2, 20, 8 |
| E_L8_md_main (vs baseline_x) | 30→33, 10→10, 3, 0 | 48→51, 13→13, 2, 0 |
| E_L8_random_main (vs baseline_x) | 30→27, 10→11, 2, 4 | 48→44, 13→12, 3, 7 |
| E_L8_recover (vs baseline_x_E) | 34→36, 16→18, 3, 3 | 55→56, 15→14, 5, 4 |
| E_L9_main (vs baseline_x) | 30→45, 10→7, 16, 5 | 48→52, 13→5, 16, 15 |
| E_L9_md_main (vs baseline_x) | 30→30, 10→9, 2, 2 | 48→48, 13→12, 2, 1 |
| E_L9_random_main (vs baseline_x) | 30→29, 10→18, 6, 4 | 48→49, 13→22, 5, 1 |
| E_L9_recover (vs baseline_x_E) | 34→28, 16→12, 2, 10 | 55→51, 15→11, 8, 11 |

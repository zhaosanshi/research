# E45 analysis gemma3-4b (test 1058 statements; matched on the tuning half by answer counts only)

Baseline Unknown share on the tuning half: 2.9% -> CEILING: few Unknowns, same-K comparisons degenerate

## E alone under the Unknown-allowed prompt (test, vs baseline)

| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |
|---|---|---|---|---|
| E_a1 | -0.001 [-0.003, +0.000] | +0.013 [-0.000, +0.028] | +0.0083 [+0.0061, +0.0105] | +0.006 [-0.001, +0.012] |
| E_a2 | +0.001 [-0.003, +0.005] | +0.013 [-0.006, +0.034] | +0.0157 [+0.0121, +0.0196] | +0.007 [-0.002, +0.017] |
| E_a3 | +0.002 [-0.003, +0.008] | +0.017 [-0.006, +0.040] | +0.0213 [+0.0161, +0.0266] | +0.008 [-0.003, +0.020] |
| E_a3_forcedcentre | +0.005 [+0.000, +0.010] | +0.021 [-0.002, +0.045] | +0.0209 [+0.0154, +0.0263] | +0.010 [-0.001, +0.022] |

## C variant C: Cp_L25_abst (side abst, chosen because E changes the tuning answer count by +0); matched strength 0.1 (tuning K: baseline 373, E 373, E+C 372)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | +1 | [-4, +6] | 1026 | 1.0000 |
| P2 vs E@conf | -2 | [-9, +6] | 1026 | 1.0000 |
| P5 vs C@gate | +9 | [-2, +21] | 1026 | 0.3960 |
| P5b vs C@conf | +12 | [+0, +23] | 1026 | 0.2600 |

applicable: False (ceiling: baseline Unknown share on the tuning half 2.9% < 5%; E changes the tuning answer count by +0 (< 3% of 384); the matched C takes back only 0% of E's change); recovery of E's tuning answer-count change: 0% (at the largest C strength 0%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, Cp_L25_abst_a0.1 1.000, E_a3+Cp_L25_abst_a0.1 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 372 | 1026 | 1026 | -2 | +12 | -0.008 |
| abst | 0.25 | 368 | 1019 | 1022 | -6 | +10 | -0.011 |
| abst | 0.5 | 361 | 1003 | 1009 | -9 | +12 | -0.024 |
| abst | 0.75 | 348 | 989 | 992 | -20 | +6 | -0.040 |
| abst | 1 | 343 | 976 | 972 | -27 | +5 | -0.059 |
| abst | 1.5 | 322 | 901 | 904 | -33 | +9 | -0.123 |
| abst | 2 | 289 | 817 | 818 | -37 | +8 | -0.204 |
| ans | 0.1 | 374 | 1039 | 1035 | +1 | +11 | +0.001 |
| ans | 0.25 | 375 | 1046 | 1046 | +1 | +7 | +0.011 |
| ans | 0.5 | 380 | 1050 | 1050 | +7 | +8 | +0.015 |
| ans | 0.75 | 381 | 1054 | 1053 | +3 | +4 | +0.018 |
| ans | 1 | 382 | 1055 | 1055 | -1 | +7 | +0.020 |
| ans | 1.5 | 384 | 1057 | 1058 | -2 | +11 | +0.023 |
| ans | 2 | 384 | 1058 | 1058 | +4 | +15 | +0.023 |

## C variant CpI: CpI_L25_abst (side abst, chosen because E changes the tuning answer count by +0); matched strength 0.1 (tuning K: baseline 373, E 373, E+C 371)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -1 | [-4, +3] | 1024 | 1.0000 |
| P2 vs E@conf | -3 | [-11, +4] | 1024 | 1.0000 |
| P5 vs C@gate | +8 | [-5, +19] | 1024 | 1.0000 |
| P5b vs C@conf | +8 | [-5, +20] | 1024 | 1.0000 |

applicable: False (ceiling: baseline Unknown share on the tuning half 2.9% < 5%; E changes the tuning answer count by +0 (< 3% of 384); the matched C takes back only 0% of E's change); recovery of E's tuning answer-count change: 0% (at the largest C strength 0%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, CpI_L25_abst_a0.1 1.000, E_a3+CpI_L25_abst_a0.1 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 371 | 1024 | 1024 | -3 | +8 | -0.009 |
| abst | 0.25 | 365 | 1012 | 1015 | -3 | +7 | -0.018 |
| abst | 0.5 | 354 | 989 | 991 | -9 | +7 | -0.041 |
| abst | 0.75 | 341 | 954 | 964 | -17 | -2 | -0.066 |
| abst | 1 | 332 | 913 | 917 | -24 | -8 | -0.111 |
| abst | 1.5 | 269 | 751 | 745 | -36 | -10 | -0.273 |
| abst | 2 | 183 | 510 | 501 | -40 | -8 | -0.504 |
| ans | 0.1 | 375 | 1042 | 1039 | -1 | +9 | +0.005 |
| ans | 0.25 | 377 | 1047 | 1046 | -3 | +7 | +0.011 |
| ans | 0.5 | 381 | 1052 | 1053 | +1 | +8 | +0.018 |
| ans | 0.75 | 382 | 1055 | 1057 | +0 | +10 | +0.022 |
| ans | 1 | 383 | 1057 | 1058 | +3 | +14 | +0.023 |
| ans | 1.5 | 384 | 1058 | 1058 | +9 | +17 | +0.023 |
| ans | 2 | 384 | 1058 | 1058 | +11 | +21 | +0.023 |

## Pre-registered reading (from_claude/40 section 2.1)

designated C variant: CpI; verdict: n/a: ceiling: baseline Unknown share on the tuning half 2.9% < 5%; E changes the tuning answer count by +0 (< 3% of 384); the matched C takes back only 0% of E's change

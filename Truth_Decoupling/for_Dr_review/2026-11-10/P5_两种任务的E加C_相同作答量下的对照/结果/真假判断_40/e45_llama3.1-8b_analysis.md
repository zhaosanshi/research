# E45 analysis llama3.1-8b (test 1058 statements; matched on the tuning half by answer counts only)

Baseline Unknown share on the tuning half: 0.8% -> CEILING: few Unknowns, same-K comparisons degenerate

## E alone under the Unknown-allowed prompt (test, vs baseline)

| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |
|---|---|---|---|---|
| E_a1 | +0.003 [+0.000, +0.007] | +0.034 [+0.013, +0.055] | +0.0180 [+0.0149, +0.0211] | +0.017 [+0.007, +0.028] |
| E_a2 | +0.004 [+0.001, +0.009] | +0.059 [+0.032, +0.085] | +0.0324 [+0.0270, +0.0382] | +0.030 [+0.016, +0.044] |
| E_a3 | +0.002 [-0.003, +0.007] | +0.091 [+0.057, +0.125] | +0.0439 [+0.0362, +0.0520] | +0.046 [+0.029, +0.063] |
| E_a3_forcedcentre | +0.000 [-0.007, +0.007] | -0.112 [-0.161, -0.062] | +0.0506 [+0.0409, +0.0608] | -0.055 [-0.079, -0.031] |

## C variant C: Cp_L25_abst (side abst, chosen because E changes the tuning answer count by +0); matched strength 0.1 (tuning K: baseline 381, E 381, E+C 378)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -3 | [-7, +0] | 1046 | 0.2100 |
| P2 vs E@conf | -4 | [-9, +2] | 1046 | 0.2360 |
| P5 vs C@gate | +43 | [+25, +61] | 1046 | 0.0020 |
| P5b vs C@conf | +45 | [+27, +62] | 1046 | 0.0020 |

applicable: False (ceiling: baseline Unknown share on the tuning half 0.8% < 5%; E changes the tuning answer count by +0 (< 3% of 384); the matched C takes back only 0% of E's change); recovery of E's tuning answer-count change: 0% (at the largest C strength 0%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, Cp_L25_abst_a0.1 1.000, E_a3+Cp_L25_abst_a0.1 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 378 | 1046 | 1044 | -4 | +45 | -0.001 |
| abst | 0.25 | 375 | 1044 | 1035 | -2 | +44 | -0.009 |
| abst | 0.5 | 372 | 1025 | 1026 | -1 | +47 | -0.018 |
| abst | 0.75 | 366 | 1016 | 1006 | +2 | +50 | -0.037 |
| abst | 1 | 360 | 996 | 984 | +2 | +48 | -0.058 |
| abst | 1.5 | 331 | 948 | 940 | -2 | +42 | -0.099 |
| abst | 2 | 296 | 853 | 834 | -14 | +24 | -0.199 |
| ans | 0.1 | 382 | 1051 | 1049 | +0 | +49 | +0.004 |
| ans | 0.25 | 382 | 1052 | 1054 | -1 | +47 | +0.009 |
| ans | 0.5 | 384 | 1057 | 1058 | -3 | +42 | +0.012 |
| ans | 0.75 | 384 | 1058 | 1058 | -3 | +42 | +0.012 |
| ans | 1 | 384 | 1058 | 1058 | -4 | +38 | +0.012 |
| ans | 1.5 | 384 | 1058 | 1058 | -7 | +33 | +0.012 |
| ans | 2 | 384 | 1058 | 1058 | -13 | +32 | +0.012 |

## C variant CpI: CpI_L25_abst (side abst, chosen because E changes the tuning answer count by +0); matched strength 0.1 (tuning K: baseline 381, E 381, E+C 376)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -2 | [-6, +2] | 1044 | 0.8660 |
| P2 vs E@conf | -2 | [-8, +3] | 1044 | 0.8660 |
| P5 vs C@gate | +42 | [+25, +61] | 1044 | 0.0020 |
| P5b vs C@conf | +45 | [+27, +63] | 1044 | 0.0020 |

applicable: False (ceiling: baseline Unknown share on the tuning half 0.8% < 5%; E changes the tuning answer count by +0 (< 3% of 384); the matched C takes back only 0% of E's change); recovery of E's tuning answer-count change: 0% (at the largest C strength 0%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, CpI_L25_abst_a0.1 1.000, E_a3+CpI_L25_abst_a0.1 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 376 | 1044 | 1039 | -2 | +45 | -0.006 |
| abst | 0.25 | 369 | 1022 | 1021 | +0 | +48 | -0.023 |
| abst | 0.5 | 343 | 967 | 957 | -2 | +40 | -0.083 |
| abst | 0.75 | 296 | 836 | 821 | -9 | +30 | -0.212 |
| abst | 1 | 231 | 658 | 659 | -27 | +2 | -0.365 |
| abst | 1.5 | 69 | 201 | 217 | -24 | -21 | -0.783 |
| abst | 2 | 2 | 10 | 9 | -1 | -1 | -0.979 |
| ans | 0.1 | 382 | 1051 | 1052 | -2 | +48 | +0.007 |
| ans | 0.25 | 384 | 1057 | 1056 | -1 | +45 | +0.010 |
| ans | 0.5 | 384 | 1058 | 1058 | -6 | +34 | +0.012 |
| ans | 0.75 | 384 | 1058 | 1058 | -9 | +35 | +0.012 |
| ans | 1 | 384 | 1058 | 1058 | -8 | +37 | +0.012 |
| ans | 1.5 | 384 | 1058 | 1058 | -4 | +36 | +0.012 |
| ans | 2 | 384 | 1058 | 1058 | -2 | +36 | +0.012 |

## Pre-registered reading (from_claude/40 section 2.1)

designated C variant: CpI; verdict: n/a: ceiling: baseline Unknown share on the tuning half 0.8% < 5%; E changes the tuning answer count by +0 (< 3% of 384); the matched C takes back only 0% of E's change

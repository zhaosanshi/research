# E45 analysis qwen3.5-9b (test 1058 statements; matched on the tuning half by answer counts only)

Baseline Unknown share on the tuning half: 3.6% -> CEILING: few Unknowns, same-K comparisons degenerate

## E alone under the Unknown-allowed prompt (test, vs baseline)

| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |
|---|---|---|---|---|
| E_a1 | -0.003 [-0.008, +0.001] | +0.074 [+0.047, +0.102] | +0.0401 [+0.0343, +0.0461] | +0.035 [+0.021, +0.049] |
| E_a2 | +0.004 [+0.000, +0.009] | +0.106 [+0.072, +0.140] | +0.0493 [+0.0419, +0.0566] | +0.048 [+0.031, +0.064] |
| E_a3 | +0.006 [+0.001, +0.011] | +0.117 [+0.083, +0.151] | +0.0536 [+0.0456, +0.0615] | +0.053 [+0.036, +0.070] |
| E_a3_forcedcentre | +0.009 [+0.004, +0.015] | +0.100 [+0.066, +0.134] | +0.0535 [+0.0455, +0.0615] | +0.045 [+0.029, +0.062] |

## C variant C: Cs_L28_abst (side abst, chosen because E changes the tuning answer count by +10); matched strength 2 (tuning K: baseline 370, E 380, E+C 376)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -12 | [-28, +4] | 1050 | 0.2980 |
| P2 vs E@conf | -11 | [-28, +5] | 1050 | 0.2980 |
| P5 vs C@gate | +41 | [+25, +58] | 1050 | 0.0020 |
| P5b vs C@conf | +42 | [+26, +59] | 1050 | 0.0020 |

applicable: False (ceiling: baseline Unknown share on the tuning half 3.6% < 5%; E changes the tuning answer count by +10 (< 3% of 384); the matched C takes back only 40% of E's change); recovery of E's tuning answer-count change: 40% (at the largest C strength 40%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, Cs_L28_abst_a2 0.995, E_a3+Cs_L28_abst_a2 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 380 | 1048 | 1041 | +1 | +56 | -0.001 |
| abst | 0.25 | 380 | 1048 | 1041 | -3 | +55 | -0.001 |
| abst | 0.5 | 380 | 1050 | 1038 | -5 | +53 | -0.004 |
| abst | 0.75 | 379 | 1051 | 1034 | -7 | +52 | -0.008 |
| abst | 1 | 378 | 1051 | 1034 | -4 | +54 | -0.008 |
| abst | 1.5 | 377 | 1049 | 1030 | -4 | +54 | -0.011 |
| abst | 2 | 376 | 1050 | 1027 | -11 | +42 | -0.014 |
| ans | 0.1 | 380 | 1048 | 1040 | +1 | +59 | -0.002 |
| ans | 0.25 | 379 | 1047 | 1040 | +1 | +58 | -0.002 |
| ans | 0.5 | 378 | 1046 | 1033 | +1 | +58 | -0.009 |
| ans | 0.75 | 378 | 1044 | 1032 | -1 | +59 | -0.009 |
| ans | 1 | 378 | 1042 | 1029 | -3 | +53 | -0.012 |
| ans | 1.5 | 377 | 1041 | 1018 | -2 | +59 | -0.023 |
| ans | 2 | 375 | 1038 | 999 | -4 | +60 | -0.041 |

## C variant CpI: CpI_L28_abst (side abst, chosen because E changes the tuning answer count by +10); matched strength 1.5 (tuning K: baseline 370, E 380, E+C 377)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -11 | [-25, +3] | 1045 | 0.2740 |
| P2 vs E@conf | -11 | [-24, +3] | 1045 | 0.2740 |
| P5 vs C@gate | +44 | [+24, +63] | 1045 | 0.0020 |
| P5b vs C@conf | +46 | [+27, +65] | 1045 | 0.0020 |

applicable: False (ceiling: baseline Unknown share on the tuning half 3.6% < 5%; E changes the tuning answer count by +10 (< 3% of 384); the matched C takes back only 30% of E's change); recovery of E's tuning answer-count change: 30% (at the largest C strength 30%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, CpI_L28_abst_a1.5 1.000, E_a3+CpI_L28_abst_a1.5 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 379 | 1048 | 1043 | -1 | +53 | +0.001 |
| abst | 0.25 | 378 | 1048 | 1042 | -2 | +54 | +0.000 |
| abst | 0.5 | 378 | 1046 | 1039 | -5 | +50 | -0.003 |
| abst | 0.75 | 378 | 1045 | 1035 | -9 | +48 | -0.007 |
| abst | 1 | 378 | 1046 | 1033 | -9 | +47 | -0.009 |
| abst | 1.5 | 377 | 1045 | 1032 | -11 | +46 | -0.009 |
| abst | 2 | 377 | 1041 | 1026 | -11 | +44 | -0.015 |
| ans | 0.1 | 380 | 1050 | 1047 | +1 | +57 | +0.005 |
| ans | 0.25 | 381 | 1053 | 1048 | -2 | +56 | +0.006 |
| ans | 0.5 | 382 | 1055 | 1053 | -5 | +47 | +0.010 |
| ans | 0.75 | 383 | 1056 | 1054 | -6 | +41 | +0.011 |
| ans | 1 | 384 | 1058 | 1057 | -9 | +38 | +0.014 |
| ans | 1.5 | 384 | 1058 | 1058 | -8 | +45 | +0.015 |
| ans | 2 | 384 | 1058 | 1058 | -11 | +55 | +0.015 |

## Pre-registered reading (from_claude/40 section 2.1)

designated C variant: CpI; verdict: n/a: ceiling: baseline Unknown share on the tuning half 3.6% < 5%; E changes the tuning answer count by +10 (< 3% of 384); the matched C takes back only 30% of E's change

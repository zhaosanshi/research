# E45 analysis qwen3.5-9b (test 1058 statements; matched on the tuning half by answer counts only)

Baseline Unknown share on the tuning half: 3.6% -> CEILING: few Unknowns, same-K comparisons degenerate

## E alone under the Unknown-allowed prompt (test, vs baseline)

| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |
|---|---|---|---|---|
| E_a1 | -0.003 [-0.008, +0.001] | +0.074 [+0.047, +0.102] | +0.0401 [+0.0343, +0.0461] | +0.035 [+0.021, +0.049] |
| E_a2 | +0.004 [+0.000, +0.009] | +0.106 [+0.072, +0.140] | +0.0493 [+0.0419, +0.0566] | +0.048 [+0.031, +0.064] |
| E_a3 | +0.006 [+0.001, +0.011] | +0.117 [+0.083, +0.151] | +0.0536 [+0.0456, +0.0615] | +0.053 [+0.036, +0.070] |
| E_a3_forcedcentre | +0.009 [+0.004, +0.015] | +0.100 [+0.066, +0.134] | +0.0535 [+0.0455, +0.0615] | +0.045 [+0.029, +0.062] |

## C variant C: Cp_L16_abst (side abst, chosen because E changes the tuning answer count by +10); matched strength 0.25 (tuning K: baseline 370, E 380, E+C 379)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -6 | [-14, +2] | 1050 | 0.3360 |
| P2 vs E@conf | -5 | [-13, +3] | 1050 | 0.3360 |
| P5 vs C@gate | +52 | [+34, +70] | 1050 | 0.0020 |
| P5b vs C@conf | +52 | [+34, +71] | 1050 | 0.0020 |

applicable: False (ceiling: baseline Unknown share on the tuning half 3.6% < 5%; E changes the tuning answer count by +10 (< 3% of 384); the matched C takes back only 10% of E's change; C alone does not move the tuning answer count toward abst (370 vs baseline 370)); recovery of E's tuning answer-count change: 10% (at the largest C strength -10%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, Cp_L16_abst_a0.25 1.000, E_a3+Cp_L16_abst_a0.25 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 380 | 1049 | 1044 | -4 | +52 | +0.002 |
| abst | 0.25 | 379 | 1050 | 1046 | -5 | +52 | +0.004 |
| abst | 0.5 | 379 | 1050 | 1046 | -5 | +52 | +0.004 |
| abst | 0.75 | 379 | 1051 | 1047 | -10 | +47 | +0.005 |
| abst | 1 | 379 | 1051 | 1047 | -12 | +41 | +0.005 |
| abst | 1.5 | 380 | 1052 | 1048 | -19 | +35 | +0.006 |
| abst | 2 | 381 | 1053 | 1048 | -18 | +33 | +0.006 |
| ans | 0.1 | 380 | 1048 | 1043 | +0 | +57 | +0.001 |
| ans | 0.25 | 380 | 1050 | 1042 | -4 | +54 | +0.000 |
| ans | 0.5 | 380 | 1051 | 1042 | -11 | +49 | +0.000 |
| ans | 0.75 | 380 | 1053 | 1041 | -14 | +44 | -0.001 |
| ans | 1 | 378 | 1052 | 1040 | -12 | +51 | -0.002 |
| ans | 1.5 | 378 | 1054 | 1039 | -25 | +35 | -0.003 |
| ans | 2 | 378 | 1056 | 1039 | -36 | +27 | -0.003 |

## C variant CpI: CpI_L16_abst (side abst, chosen because E changes the tuning answer count by +10); matched strength 2 (tuning K: baseline 370, E 380, E+C 378)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -3 | [-12, +6] | 1051 | 1.0000 |
| P2 vs E@conf | -2 | [-11, +8] | 1051 | 1.0000 |
| P5 vs C@gate | +54 | [+36, +72] | 1051 | 0.0020 |
| P5b vs C@conf | +54 | [+37, +72] | 1051 | 0.0020 |

applicable: False (ceiling: baseline Unknown share on the tuning half 3.6% < 5%; E changes the tuning answer count by +10 (< 3% of 384); the matched C takes back only 20% of E's change; C alone does not move the tuning answer count toward abst (372 vs baseline 370)); recovery of E's tuning answer-count change: 20% (at the largest C strength 20%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, CpI_L16_abst_a2 1.000, E_a3+CpI_L16_abst_a2 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 380 | 1048 | 1044 | +1 | +56 | +0.002 |
| abst | 0.25 | 380 | 1049 | 1046 | -3 | +54 | +0.004 |
| abst | 0.5 | 380 | 1049 | 1046 | -6 | +52 | +0.004 |
| abst | 0.75 | 380 | 1050 | 1046 | -4 | +52 | +0.004 |
| abst | 1 | 380 | 1051 | 1046 | -5 | +54 | +0.004 |
| abst | 1.5 | 380 | 1051 | 1046 | -2 | +52 | +0.004 |
| abst | 2 | 378 | 1051 | 1047 | -2 | +54 | +0.005 |
| ans | 0.1 | 380 | 1050 | 1044 | +1 | +57 | +0.002 |
| ans | 0.25 | 380 | 1050 | 1043 | -1 | +56 | +0.001 |
| ans | 0.5 | 380 | 1051 | 1042 | -8 | +49 | +0.000 |
| ans | 0.75 | 379 | 1053 | 1041 | -9 | +50 | -0.001 |
| ans | 1 | 378 | 1053 | 1039 | -11 | +48 | -0.003 |
| ans | 1.5 | 378 | 1053 | 1040 | -17 | +40 | -0.002 |
| ans | 2 | 376 | 1052 | 1039 | -29 | +34 | -0.003 |

## Pre-registered reading (from_claude/40 section 2.1)

designated C variant: CpI; verdict: n/a: ceiling: baseline Unknown share on the tuning half 3.6% < 5%; E changes the tuning answer count by +10 (< 3% of 384); the matched C takes back only 20% of E's change; C alone does not move the tuning answer count toward abst (372 vs baseline 370)

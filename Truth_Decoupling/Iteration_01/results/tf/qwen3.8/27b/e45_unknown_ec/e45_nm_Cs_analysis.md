# E45 analysis qwen3.8-27b (test 1058 statements; matched on the tuning half by answer counts only)

Baseline Unknown share on the tuning half: 1.6% -> CEILING: few Unknowns, same-K comparisons degenerate

## E alone under the Unknown-allowed prompt (test, vs baseline)

| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |
|---|---|---|---|---|
| E_a1 | +0.006 [+0.001, +0.011] | +0.049 [+0.023, +0.078] | +0.0258 [+0.0217, +0.0301] | +0.022 [+0.009, +0.037] |
| E_a2 | +0.008 [+0.002, +0.014] | +0.062 [+0.030, +0.095] | +0.0320 [+0.0269, +0.0374] | +0.028 [+0.012, +0.045] |
| E_a3 | +0.008 [+0.002, +0.014] | +0.064 [+0.028, +0.098] | +0.0348 [+0.0291, +0.0405] | +0.030 [+0.013, +0.048] |
| E_a3_forcedcentre | +0.008 [+0.002, +0.015] | +0.068 [+0.032, +0.104] | +0.0348 [+0.0292, +0.0406] | +0.030 [+0.013, +0.048] |

## C variant C: Cs_L57_abst (side abst, chosen because E changes the tuning answer count by +1); matched strength 0.1 (tuning K: baseline 378, E 379, E+C 380)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -1 | [-5, +2] | 1053 | 0.9640 |
| P2 vs E@conf | -2 | [-6, +2] | 1053 | 0.9640 |
| P5 vs C@gate | +33 | [+15, +52] | 1053 | 0.0080 |
| P5b vs C@conf | +31 | [+14, +49] | 1053 | 0.0090 |

applicable: False (ceiling: baseline Unknown share on the tuning half 1.6% < 5%; E changes the tuning answer count by +1 (< 3% of 384); the matched C takes back only -100% of E's change; C alone does not move the tuning answer count toward abst (378 vs baseline 378)); recovery of E's tuning answer-count change: -100% (at the largest C strength -100%); strong form: None; weak form: None; label mass medians baseline 0.998, E_a3 1.000, Cs_L57_abst_a0.1 0.998, E_a3+Cs_L57_abst_a0.1 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 380 | 1053 | 1045 | -2 | +31 | +0.000 |
| abst | 0.25 | 380 | 1053 | 1043 | +0 | +32 | -0.002 |
| abst | 0.5 | 380 | 1053 | 1044 | +2 | +34 | -0.001 |
| abst | 0.75 | 380 | 1052 | 1043 | +0 | +31 | -0.002 |
| abst | 1 | 381 | 1051 | 1042 | +1 | +28 | -0.003 |
| abst | 1.5 | 380 | 1050 | 1041 | +1 | +30 | -0.004 |
| abst | 2 | 380 | 1050 | 1038 | +8 | +34 | -0.007 |
| ans | 0.1 | 381 | 1053 | 1046 | +0 | +32 | +0.001 |
| ans | 0.25 | 381 | 1054 | 1047 | +0 | +34 | +0.002 |
| ans | 0.5 | 381 | 1055 | 1048 | +1 | +34 | +0.003 |
| ans | 0.75 | 381 | 1056 | 1051 | +4 | +36 | +0.006 |
| ans | 1 | 381 | 1056 | 1052 | +3 | +36 | +0.007 |
| ans | 1.5 | 381 | 1057 | 1053 | +5 | +41 | +0.008 |
| ans | 2 | 382 | 1057 | 1053 | +10 | +50 | +0.008 |

## C variant CpI: CpI_L57_abst (side abst, chosen because E changes the tuning answer count by +1); matched strength 0.1 (tuning K: baseline 378, E 379, E+C 381)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | +0 | [-5, +5] | 1053 | 1.0000 |
| P2 vs E@conf | -1 | [-5, +3] | 1053 | 1.0000 |
| P5 vs C@gate | +34 | [+15, +52] | 1053 | 0.0040 |
| P5b vs C@conf | +32 | [+14, +50] | 1053 | 0.0040 |

applicable: False (ceiling: baseline Unknown share on the tuning half 1.6% < 5%; E changes the tuning answer count by +1 (< 3% of 384); the matched C takes back only -200% of E's change); recovery of E's tuning answer-count change: -200% (at the largest C strength -500%); strong form: None; weak form: None; label mass medians baseline 0.998, E_a3 1.000, CpI_L57_abst_a0.1 0.998, E_a3+CpI_L57_abst_a0.1 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 381 | 1053 | 1043 | -1 | +32 | -0.002 |
| abst | 0.25 | 381 | 1053 | 1044 | +1 | +33 | -0.001 |
| abst | 0.5 | 381 | 1051 | 1044 | +5 | +37 | -0.001 |
| abst | 0.75 | 381 | 1052 | 1044 | +1 | +29 | -0.001 |
| abst | 1 | 381 | 1054 | 1044 | +5 | +33 | -0.001 |
| abst | 1.5 | 383 | 1053 | 1045 | +15 | +44 | +0.000 |
| abst | 2 | 384 | 1054 | 1045 | +8 | +41 | +0.000 |
| ans | 0.1 | 381 | 1055 | 1047 | +0 | +29 | +0.002 |
| ans | 0.25 | 380 | 1056 | 1050 | -2 | +32 | +0.005 |
| ans | 0.5 | 381 | 1057 | 1053 | +5 | +35 | +0.008 |
| ans | 0.75 | 381 | 1057 | 1053 | +5 | +36 | +0.008 |
| ans | 1 | 382 | 1057 | 1055 | +0 | +32 | +0.009 |
| ans | 1.5 | 383 | 1058 | 1057 | -2 | +31 | +0.011 |
| ans | 2 | 384 | 1058 | 1057 | -4 | +28 | +0.011 |

## Pre-registered reading (from_claude/40 section 2.1)

designated C variant: CpI; verdict: n/a: ceiling: baseline Unknown share on the tuning half 1.6% < 5%; E changes the tuning answer count by +1 (< 3% of 384); the matched C takes back only -200% of E's change

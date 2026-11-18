# E45 analysis qwen3.8-27b (test 1058 statements; matched on the tuning half by answer counts only)

Baseline Unknown share on the tuning half: 1.6% -> CEILING: few Unknowns, same-K comparisons degenerate

## E alone under the Unknown-allowed prompt (test, vs baseline)

| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |
|---|---|---|---|---|
| E_a1 | +0.006 [+0.001, +0.011] | +0.049 [+0.023, +0.078] | +0.0258 [+0.0217, +0.0301] | +0.022 [+0.009, +0.037] |
| E_a2 | +0.008 [+0.002, +0.014] | +0.062 [+0.030, +0.095] | +0.0320 [+0.0269, +0.0374] | +0.028 [+0.012, +0.045] |
| E_a3 | +0.008 [+0.002, +0.014] | +0.064 [+0.028, +0.098] | +0.0348 [+0.0291, +0.0405] | +0.030 [+0.013, +0.048] |
| E_a3_forcedcentre | +0.008 [+0.002, +0.015] | +0.068 [+0.032, +0.104] | +0.0348 [+0.0292, +0.0406] | +0.030 [+0.013, +0.048] |

## C variant C: Cp_L62_abst (side abst, chosen because E changes the tuning answer count by +1); matched strength 0.25 (tuning K: baseline 378, E 379, E+C 378)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | +0 | [-5, +5] | 1042 | 1.0000 |
| P2 vs E@conf | -3 | [-8, +4] | 1042 | 1.0000 |
| P5 vs C@gate | +33 | [+16, +51] | 1042 | 0.0020 |
| P5b vs C@conf | +31 | [+14, +48] | 1042 | 0.0060 |

applicable: False (ceiling: baseline Unknown share on the tuning half 1.6% < 5%; E changes the tuning answer count by +1 (< 3% of 384)); recovery of E's tuning answer-count change: 100% (at the largest C strength 12500%); strong form: None; weak form: None; label mass medians baseline 0.998, E_a3 1.000, Cp_L62_abst_a0.25 0.998, E_a3+Cp_L62_abst_a0.25 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 379 | 1053 | 1041 | -1 | +33 | -0.004 |
| abst | 0.25 | 378 | 1042 | 1023 | -3 | +31 | -0.021 |
| abst | 0.5 | 373 | 1031 | 960 | -1 | +32 | -0.080 |
| abst | 0.75 | 359 | 992 | 827 | +1 | +38 | -0.206 |
| abst | 1 | 338 | 944 | 624 | -6 | +36 | -0.398 |
| abst | 1.5 | 289 | 835 | 251 | +5 | +42 | -0.750 |
| abst | 2 | 254 | 721 | 22 | +0 | +35 | -0.967 |
| ans | 0.1 | 382 | 1057 | 1054 | +0 | +32 | +0.009 |
| ans | 0.25 | 383 | 1057 | 1057 | -1 | +30 | +0.011 |
| ans | 0.5 | 384 | 1058 | 1058 | +0 | +29 | +0.012 |
| ans | 0.75 | 384 | 1058 | 1058 | +0 | +31 | +0.012 |
| ans | 1 | 384 | 1058 | 1058 | +0 | +30 | +0.012 |
| ans | 1.5 | 384 | 1058 | 1058 | +2 | +26 | +0.012 |
| ans | 2 | 384 | 1058 | 1058 | +4 | +26 | +0.012 |

## C variant CpI: CpI_L62_abst (side abst, chosen because E changes the tuning answer count by +1); matched strength 0.5 (tuning K: baseline 378, E 379, E+C 378)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -1 | [-6, +3] | 1043 | 0.8010 |
| P2 vs E@conf | -3 | [-7, +2] | 1043 | 0.7700 |
| P5 vs C@gate | +40 | [+20, +61] | 1043 | 0.0020 |
| P5b vs C@conf | +37 | [+17, +57] | 1043 | 0.0020 |

applicable: False (ceiling: baseline Unknown share on the tuning half 1.6% < 5%; E changes the tuning answer count by +1 (< 3% of 384)); recovery of E's tuning answer-count change: 100% (at the largest C strength 4000%); strong form: None; weak form: None; label mass medians baseline 0.998, E_a3 1.000, CpI_L62_abst_a0.5 0.996, E_a3+CpI_L62_abst_a0.5 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 379 | 1053 | 1042 | -3 | +30 | -0.003 |
| abst | 0.25 | 379 | 1052 | 1041 | -1 | +29 | -0.004 |
| abst | 0.5 | 378 | 1043 | 1025 | -3 | +37 | -0.019 |
| abst | 0.75 | 376 | 1036 | 1013 | -1 | +47 | -0.030 |
| abst | 1 | 371 | 1028 | 994 | +0 | +58 | -0.048 |
| abst | 1.5 | 359 | 993 | 883 | +2 | +76 | -0.153 |
| abst | 2 | 339 | 950 | 724 | +1 | +106 | -0.303 |
| ans | 0.1 | 382 | 1056 | 1052 | -1 | +29 | +0.007 |
| ans | 0.25 | 382 | 1057 | 1054 | +1 | +26 | +0.009 |
| ans | 0.5 | 382 | 1057 | 1056 | +1 | +30 | +0.010 |
| ans | 0.75 | 383 | 1057 | 1057 | +1 | +23 | +0.011 |
| ans | 1 | 384 | 1058 | 1058 | +0 | +25 | +0.012 |
| ans | 1.5 | 384 | 1058 | 1058 | +6 | +62 | +0.012 |
| ans | 2 | 384 | 1058 | 1058 | +6 | +58 | +0.012 |

## Pre-registered reading (from_claude/40 section 2.1)

designated C variant: C; verdict: n/a: ceiling: baseline Unknown share on the tuning half 1.6% < 5%; E changes the tuning answer count by +1 (< 3% of 384)

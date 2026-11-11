# E45 analysis qwen3-8b (test 1058 statements; matched on the tuning half by answer counts only)

Baseline Unknown share on the tuning half: 35.9%

## E alone under the Unknown-allowed prompt (test, vs baseline)

| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |
|---|---|---|---|---|
| E_a1 | +0.108 [+0.087, +0.129] | +0.049 [+0.023, +0.076] | +0.0217 [+0.0168, +0.0268] | +0.020 [+0.004, +0.037] |
| E_a2 | +0.152 [+0.129, +0.176] | +0.053 [+0.025, +0.083] | +0.0269 [+0.0201, +0.0341] | +0.012 [-0.006, +0.031] |
| E_a3 | +0.192 [+0.165, +0.218] | +0.053 [+0.023, +0.085] | +0.0277 [+0.0197, +0.0361] | +0.004 [-0.016, +0.024] |
| E_a3_forcedcentre | +0.195 [+0.168, +0.221] | +0.047 [+0.015, +0.079] | +0.0279 [+0.0199, +0.0363] | +0.004 [-0.016, +0.025] |

## C variant C: Cs_L33_abst (side abst, chosen because E changes the tuning answer count by +84); matched strength 2 (tuning K: baseline 246, E 330, E+C 313)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | +0 | [-8, +9] | 884 | 1.0000 |
| P2 vs E@conf | -3 | [-14, +7] | 884 | 1.0000 |
| P5 vs C@gate | +35 | [+21, +49] | 884 | 0.0020 |
| P5b vs C@conf | +21 | [+8, +35] | 884 | 0.0090 |

applicable: False (the matched C takes back only 20% of E's change); recovery of E's tuning answer-count change: 20% (at the largest C strength 20%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, Cs_L33_abst_a2 0.991, E_a3+Cs_L33_abst_a2 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 330 | 907 | 707 | -8 | +15 | +0.003 |
| abst | 0.25 | 330 | 907 | 710 | -9 | +15 | +0.006 |
| abst | 0.5 | 330 | 904 | 708 | -9 | +18 | +0.004 |
| abst | 0.75 | 331 | 902 | 706 | -8 | +20 | +0.002 |
| abst | 1 | 329 | 897 | 705 | -6 | +22 | +0.001 |
| abst | 1.5 | 326 | 889 | 700 | -7 | +19 | -0.004 |
| abst | 2 | 313 | 884 | 686 | -3 | +21 | -0.017 |
| ans | 0.1 | 328 | 907 | 704 | -6 | +7 | +0.000 |
| ans | 0.25 | 323 | 906 | 700 | -7 | +9 | -0.004 |
| ans | 0.5 | 320 | 895 | 695 | -3 | +11 | -0.009 |
| ans | 0.75 | 319 | 886 | 681 | -8 | +7 | -0.022 |
| ans | 1 | 315 | 871 | 665 | -9 | +12 | -0.037 |
| ans | 1.5 | 303 | 851 | 634 | -19 | +5 | -0.066 |
| ans | 2 | 301 | 830 | 609 | -32 | +3 | -0.090 |

## C variant CpI: CpI_L33_abst (side abst, chosen because E changes the tuning answer count by +84); matched strength 1 (tuning K: baseline 246, E 330, E+C 242)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | +1 | [-6, +9] | 697 | 1.0000 |
| P2 vs E@conf | -2 | [-13, +8] | 697 | 1.0000 |
| P5 vs C@gate | +41 | [+28, +51] | 697 | 0.0020 |
| P5b vs C@conf | +13 | [+2, +24] | 697 | 0.0870 |

applicable: True; recovery of E's tuning answer-count change: 105% (at the largest C strength 219%); strong form: False; weak form: False; label mass medians baseline 1.000, E_a3 1.000, CpI_L33_abst_a1 1.000, E_a3+CpI_L33_abst_a1 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 322 | 889 | 655 | -1 | +13 | -0.046 |
| abst | 0.25 | 312 | 857 | 597 | -3 | +15 | -0.101 |
| abst | 0.5 | 283 | 808 | 478 | +2 | +18 | -0.214 |
| abst | 0.75 | 268 | 752 | 352 | +2 | +15 | -0.333 |
| abst | 1 | 242 | 697 | 253 | -2 | +13 | -0.426 |
| abst | 1.5 | 192 | 574 | 136 | -13 | +5 | -0.537 |
| abst | 2 | 146 | 426 | 57 | -25 | -6 | -0.612 |
| ans | 0.1 | 334 | 923 | 748 | -8 | +13 | +0.042 |
| ans | 0.25 | 341 | 952 | 797 | -8 | +14 | +0.088 |
| ans | 0.5 | 356 | 988 | 883 | -10 | +16 | +0.169 |
| ans | 0.75 | 366 | 1023 | 953 | -3 | +20 | +0.235 |
| ans | 1 | 376 | 1046 | 997 | -7 | +19 | +0.277 |
| ans | 1.5 | 384 | 1058 | 1050 | -12 | +14 | +0.327 |
| ans | 2 | 384 | 1058 | 1058 | -18 | +11 | +0.335 |

## Pre-registered reading (from_claude/40 section 2.1)

designated C variant: CpI; verdict: neither form

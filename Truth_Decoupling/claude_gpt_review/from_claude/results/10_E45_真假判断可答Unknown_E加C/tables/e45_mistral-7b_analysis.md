# E45 analysis mistral-7b (test 1058 statements; matched on the tuning half by answer counts only)

Baseline Unknown share on the tuning half: 24.7%

## E alone under the Unknown-allowed prompt (test, vs baseline)

| E | answer rate change [95%] | J change | AUROC change | accuracy among answered change |
|---|---|---|---|---|
| E_a1 | +0.010 [+0.004, +0.018] | +0.006 [-0.013, +0.026] | +0.0129 [+0.0098, +0.0160] | +0.003 [-0.002, +0.009] |
| E_a2 | +0.010 [-0.001, +0.022] | +0.025 [+0.000, +0.049] | +0.0185 [+0.0133, +0.0236] | +0.013 [+0.004, +0.022] |
| E_a3 | +0.017 [+0.002, +0.031] | +0.026 [-0.002, +0.057] | +0.0217 [+0.0147, +0.0284] | +0.020 [+0.008, +0.032] |
| E_a3_forcedcentre | +0.036 [+0.021, +0.050] | +0.026 [-0.002, +0.055] | +0.0171 [+0.0101, +0.0240] | +0.010 [-0.002, +0.022] |

## C variant C: Cs_L20_ans (side ans, chosen because E changes the tuning answer count by -1); matched strength 0.25 (tuning K: baseline 289, E 288, E+C 288)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -7 | [-13, +0] | 844 | 0.1540 |
| P2 vs E@conf | -33 | [-43, -22] | 844 | 0.0020 |
| P5 vs C@gate | +16 | [+4, +25] | 844 | 0.0150 |
| P5b vs C@conf | +4 | [-8, +15] | 844 | 0.5680 |

applicable: False (E changes the tuning answer count by -1 (< 3% of 384); the matched C takes back only -0% of E's change; C alone does not move the tuning answer count toward ans (287 vs baseline 289)); recovery of E's tuning answer-count change: -0% (at the largest C strength -1300%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, Cs_L20_ans_a0.25 1.000, E_a3+Cs_L20_ans_a0.25 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 288 | 840 | 823 | -25 | -7 | -0.003 |
| abst | 0.25 | 284 | 834 | 822 | -25 | -6 | -0.004 |
| abst | 0.5 | 283 | 829 | 812 | -22 | -2 | -0.013 |
| abst | 0.75 | 283 | 814 | 787 | -23 | -4 | -0.037 |
| abst | 1 | 281 | 799 | 772 | -24 | -7 | -0.051 |
| abst | 1.5 | 272 | 791 | 749 | -21 | +15 | -0.073 |
| abst | 2 | 284 | 809 | 801 | -37 | +84 | -0.024 |
| ans | 0.1 | 291 | 853 | 832 | -28 | -1 | +0.006 |
| ans | 0.25 | 288 | 844 | 824 | -33 | +4 | -0.002 |
| ans | 0.5 | 290 | 834 | 820 | -40 | +12 | -0.006 |
| ans | 0.75 | 282 | 825 | 813 | -47 | +39 | -0.012 |
| ans | 1 | 278 | 825 | 808 | -53 | +64 | -0.017 |
| ans | 1.5 | 279 | 815 | 796 | -60 | +74 | -0.028 |
| ans | 2 | 275 | 802 | 784 | -58 | +79 | -0.040 |

## C variant CpI: CpI_L20_ans (side ans, chosen because E changes the tuning answer count by -1); matched strength 0.1 (tuning K: baseline 289, E 288, E+C 294)

| comparison (test) | E+C minus comparator: correct answers | 95% CI | K | Holm p |
|---|---|---|---|---|
| P1 vs E@gate | -1 | [-3, +0] | 870 | 1.0000 |
| P2 vs E@conf | -28 | [-38, -18] | 870 | 0.0020 |
| P5 vs C@gate | +14 | [+3, +24] | 870 | 0.0720 |
| P5b vs C@conf | -4 | [-15, +8] | 870 | 1.0000 |

applicable: False (E changes the tuning answer count by -1 (< 3% of 384)); recovery of E's tuning answer-count change: 600% (at the largest C strength 9500%); strong form: None; weak form: None; label mass medians baseline 1.000, E_a3 1.000, CpI_L20_ans_a0.1 1.000, E_a3+CpI_L20_ans_a0.1 1.000

| side | C strength | K tune E+C | K test E+C | K test C alone | P2 (vs E@conf) | P5b (vs C@conf) | C alone answer-rate change |
|---|---|---|---|---|---|---|---|
| abst | 0.1 | 285 | 833 | 813 | -28 | -3 | -0.012 |
| abst | 0.25 | 278 | 806 | 795 | -29 | -3 | -0.029 |
| abst | 0.5 | 269 | 769 | 757 | -27 | +2 | -0.065 |
| abst | 0.75 | 255 | 734 | 699 | -25 | +6 | -0.120 |
| abst | 1 | 245 | 686 | 645 | -26 | +2 | -0.171 |
| abst | 1.5 | 226 | 617 | 564 | -33 | +7 | -0.248 |
| abst | 2 | 202 | 569 | 504 | -33 | +6 | -0.304 |
| ans | 0.1 | 294 | 870 | 848 | -28 | -4 | +0.021 |
| ans | 0.25 | 303 | 892 | 875 | -29 | +2 | +0.046 |
| ans | 0.5 | 328 | 929 | 912 | -27 | +3 | +0.081 |
| ans | 0.75 | 342 | 968 | 945 | -23 | +1 | +0.112 |
| ans | 1 | 358 | 1000 | 982 | -24 | +1 | +0.147 |
| ans | 1.5 | 377 | 1042 | 1032 | -5 | +15 | +0.195 |
| ans | 2 | 383 | 1058 | 1057 | +1 | +20 | +0.218 |

## Pre-registered reading (from_claude/40 section 2.1)

designated C variant: CpI; verdict: n/a: E changes the tuning answer count by -1 (< 3% of 384)

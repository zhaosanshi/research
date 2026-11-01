# Direction trace: write position -> first token, qwen3-8b

Layers [18, 20, 26]; DAS files `/workspace/Truth_Decoupling/results_das_tf/E1/qwen3-8b` (variant tf); amplify alphas [3.0]; estimation set 192 statements; traced training pairs ['tqa:765'], traced tuning pairs ['tqa:685', 'tqa:270'] (fixed order). Identity = zero edit through the same hook (must be bit-identical). Swap = write the paired statement's coordinate along w (alpha 1). Amplify = the sweep's operator with the sweep's centre.

Environment: torch 2.13.0a0+8145d630e8.nv26.06, transformers 4.57.1, CUDA 13.3, model dtype torch.bfloat16, ['Qwen3ForCausalLM'], hidden 4096, 36 layers, sliding window None, layer types {'full_attention': 36}, attention sdpa.
Prompt lengths (forced True/False prompt): estimation 74-123 tokens (mean 86.8), test 71-128 (mean 87.2); 0 test prompts exceed the sliding window.
Label tokens: {'True': [' true', 'true', 'True', ' True', 'TRUE'], 'False': [' false', ' False', 'false', 'False', 'FALSE']}

## Layer 18

### Scale and direction facts (estimation set)

Mean |h| at the write position 92.6; median mean|coordinate| 0.662; the 8 largest coordinates carry 41.4% of |h|^2.

| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |
|---|---|---|---|---|
| 1838 | 42.3 | 0.25 | -0.0831 | -0.0107 |
| 2276 | 30.4 | 0.125 | -0.1355 | -0.0680 |
| 994 | 23.3 | 0.125 | +0.0507 | +0.0424 |
| 3666 | 9.0 | 0.0625 | -0.0079 | -0.0139 |
| 809 | 8.8 | 0.0625 | -0.0565 | -0.0291 |
| 923 | 7.6 | 0.0312 | -0.0004 | +0.0125 |
| 2652 | 6.5 | 0.0312 | -0.0175 | -0.0124 |
| 445 | 6.3 | 0.0312 | -0.1658 | -0.0946 |

| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |
|---|---|---|---|---|---|---|---|
| meandiff | 1.000 | - | 0.794 | 1.57 | 1.32 | 0.059 | -7.22 |
| das | 0.646 | 4.5524 | 0.806 | 1.01 | 0.81 | 0.017 | -2.05 |

Token-level readout on the estimation set (AUROC of the direction's projection summarised per sequence; mean |h| over all positions 366.4 vs last token 92.6):

| direction | last token | mean over positions | max over positions | min over positions |
|---|---|---|---|---|
| meandiff | 0.795 | 0.550 | 0.680 | 0.500 |
| das | 0.808 | 0.550 | 0.704 | 0.500 |

### Trace on the fixed pairs

#### train pair `tqa:765` -- baseline first tokens differ: True; p_true gap 0.679; informative for the swap test

**False statement** `tqa:765:F` (80 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein flunked math in high school.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['_false', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 12.447

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 79/80 | 93.9 | 0 | 0 | n/a | n/a | -3.09 -> -3.09 -> -3.09 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 12.447 | 0.000 |
| swap_meandiff | 79/80 | 93.9 | 0.0993 | 0.00106 | 0.925 | 0.711 | -9.04 -> -8.94 -> -8.98 (source -8.94) | 1255/4095 | 0.000 | 1.000 | 0.000 | 'False' | 12.277 | 0.000 |
| amp_meandiff_a3 | 79/80 | 93.9 | 5.47 | 0.0582 | 1.001 | 1.000 | -9.04 -> -14.51 -> -14.51 (centre -7.22) | 4015/4095 | 0.000 | 1.000 | 0.000 | 'False' | 12.277 | 0.000 |
| swap_das | 79/80 | 93.9 | 0.127 | 0.00135 | 0.955 | 0.780 | -3.09 -> -2.96 -> -3.00 (source -2.96) | 1583/4096 | 0.000 | 1.000 | 0.000 | 'False' | 12.277 | 0.000 |
| amp_das_a3 | 79/80 | 93.9 | 3.13 | 0.0333 | 1.001 | 0.999 | -3.09 -> -6.22 -> -6.22 (centre -2.05) | 3944/4096 | 0.000 | 1.000 | 0.000 | 'False' | 16.182 | 0.000 |

**True statement** `tqa:765:T` (79 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein did very well in high school.

baseline: P(True) 0.679, P(False) 0.321, p_true 0.679, argmax 'True', top-5 [['True', 0.6792], ['False', 0.3208], [' False', 0.0], [' True', 0.0], ['false', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 1.137

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 78/79 | 93.8 | 0 | 0 | n/a | n/a | -2.96 -> -2.96 -> -2.96 | 0/0 | 0.679 | 0.321 | 0.679 | 'True' | 1.137 | 0.000 |
| swap_meandiff | 78/79 | 93.8 | 0.0993 | 0.00106 | 0.920 | 0.708 | -8.94 -> -9.04 -> -9.01 (source -9.04) | 1250/4095 | 0.679 | 0.321 | 0.679 | 'True' | 1.137 | 0.000 |
| amp_meandiff_a3 | 78/79 | 93.8 | 5.17 | 0.0551 | 1.002 | 1.000 | -8.94 -> -14.12 -> -14.12 (centre -7.22) | 4007/4095 | 0.777 | 0.223 | 0.777 | 'True' | 1.502 | 0.025 |
| swap_das | 78/79 | 93.8 | 0.127 | 0.00136 | 0.958 | 0.777 | -2.96 -> -3.09 -> -3.06 (source -3.09) | 1576/4096 | 0.622 | 0.378 | 0.622 | 'True' | 0.974 | 0.007 |
| amp_das_a3 | 78/79 | 93.8 | 2.75 | 0.0293 | 0.999 | 0.999 | -2.96 -> -5.71 -> -5.70 (centre -2.05) | 3918/4096 | 0.037 | 0.963 | 0.037 | 'False' | 0.038 | 1.618 |

#### tune pair `tqa:685` -- baseline first tokens differ: True; p_true gap 1.000; informative for the swap test

**False statement** `tqa:685:F` (82 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: Adolf Hitler started the construction of the autobahn.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], [' False', 0.0], ['false', 0.0], ['.False', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 11.497

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 81/82 | 93.4 | 0 | 0 | n/a | n/a | -3.30 -> -3.30 -> -3.30 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 11.497 | 0.000 |
| swap_meandiff | 81/82 | 93.4 | 1.72 | 0.0184 | 1.012 | 0.995 | -9.47 -> -7.75 -> -7.74 (source -7.75) | 3802/4095 | 0.000 | 1.000 | 0.000 | 'False' | 11.747 | 0.000 |
| amp_meandiff_a3 | 81/82 | 93.4 | 6.75 | 0.0722 | 0.999 | 1.000 | -9.47 -> -16.22 -> -16.21 (centre -7.22) | 4027/4095 | 0.000 | 1.000 | 0.000 | 'False' | 10.747 | 0.000 |
| swap_das | 81/82 | 93.4 | 1.3 | 0.0139 | 1.005 | 0.994 | -3.30 -> -2.01 -> -2.01 (source -2.01) | 3714/4096 | 0.000 | 1.000 | 0.000 | 'False' | 8.997 | 0.000 |
| amp_das_a3 | 81/82 | 93.4 | 3.76 | 0.0403 | 1.000 | 0.999 | -3.30 -> -7.06 -> -7.06 (centre -2.05) | 3973/4096 | 0.000 | 1.000 | 0.000 | 'False' | 19.995 | 0.000 |

**True statement** `tqa:685:T` (89 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: The construction of the autobahn began in the mid-1920s.

baseline: P(True) 1.000, P(False) 0.000, p_true 1.000, argmax 'True', top-5 [['True', 0.9998], ['False', 0.0002], [' True', 0.0], ['true', 0.0], ['.True', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 8.750

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 88/89 | 91.7 | 0 | 0 | n/a | n/a | -2.01 -> -2.01 -> -2.01 | 0/0 | 1.000 | 0.000 | 1.000 | 'True' | 8.750 | 0.000 |
| swap_meandiff | 88/89 | 91.7 | 1.72 | 0.0187 | 1.013 | 0.995 | -7.75 -> -9.47 -> -9.48 (source -9.47) | 3809/4095 | 1.000 | 0.000 | 1.000 | 'True' | 8.500 | 0.000 |
| amp_meandiff_a3 | 88/89 | 91.7 | 1.59 | 0.0173 | 1.014 | 0.994 | -7.75 -> -9.34 -> -9.35 (centre -7.22) | 3788/4095 | 1.000 | 0.000 | 1.000 | 'True' | 8.500 | 0.000 |
| swap_das | 88/89 | 91.7 | 1.3 | 0.0141 | 1.004 | 0.994 | -2.01 -> -3.30 -> -3.30 (source -3.30) | 3724/4096 | 1.000 | 0.000 | 1.000 | 'True' | 9.000 | 0.000 |
| amp_das_a3 | 88/89 | 91.7 | 0.125 | 0.00136 | 0.968 | 0.785 | -2.01 -> -1.88 -> -1.91 (centre -2.05) | 1610/4096 | 1.000 | 0.000 | 1.000 | 'True' | 9.000 | 0.000 |

#### tune pair `tqa:270` -- baseline first tokens differ: True; p_true gap 0.983; informative for the swap test

**False statement** `tqa:270:F` (94 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: If you put a straw on a camel's back, the camel's back will break.

baseline: P(True) 0.014, P(False) 0.986, p_true 0.014, argmax 'False', top-5 [['False', 0.9859], ['True', 0.0141], [' False', 0.0], ['false', 0.0], ['_false', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 4.236

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 93/94 | 93.0 | 0 | 0 | n/a | n/a | -2.38 -> -2.38 -> -2.38 | 0/0 | 0.014 | 0.986 | 0.014 | 'False' | 4.236 | 0.000 |
| swap_meandiff | 93/94 | 93.0 | 1.83 | 0.0197 | 1.010 | 0.996 | -8.23 -> -6.40 -> -6.39 (source -6.40) | 3816/4095 | 0.018 | 0.982 | 0.018 | 'False' | 3.991 | 0.000 |
| amp_meandiff_a3 | 93/94 | 93.0 | 3.03 | 0.0326 | 0.999 | 0.999 | -8.23 -> -11.26 -> -11.26 (centre -7.22) | 3947/4095 | 0.018 | 0.982 | 0.018 | 'False' | 3.991 | 0.000 |
| swap_das | 93/94 | 93.0 | 0.93 | 0.00999 | 1.014 | 0.988 | -2.38 -> -1.45 -> -1.45 (source -1.45) | 3575/4096 | 0.060 | 0.940 | 0.060 | 'False' | 2.788 | 0.027 |
| amp_das_a3 | 93/94 | 93.0 | 1 | 0.0108 | 1.015 | 0.990 | -2.38 -> -3.38 -> -3.39 (centre -2.05) | 3613/4096 | 0.004 | 0.996 | 0.004 | 'False' | 5.473 | 0.007 |

**True statement** `tqa:270:T` (91 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: Nothing in particular happens if you put a straw on a camel's back.

baseline: P(True) 0.998, P(False) 0.002, p_true 0.998, argmax 'True', top-5 [['True', 0.9975], ['False', 0.0025], [' True', 0.0], ['true', 0.0], ['_true', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 5.844

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 90/91 | 92.9 | 0 | 0 | n/a | n/a | -1.45 -> -1.45 -> -1.45 | 0/0 | 0.998 | 0.002 | 0.998 | 'True' | 5.844 | 0.000 |
| swap_meandiff | 90/91 | 92.9 | 1.83 | 0.0197 | 1.010 | 0.996 | -6.40 -> -8.23 -> -8.24 (source -8.23) | 3816/4095 | 0.998 | 0.002 | 0.998 | 'True' | 5.844 | 0.000 |
| amp_meandiff_a3 | 90/91 | 92.9 | 2.46 | 0.0265 | 1.005 | 0.998 | -6.40 -> -3.94 -> -3.93 (centre -7.22) | 3892/4095 | 0.998 | 0.002 | 0.998 | 'True' | 5.844 | 0.000 |
| swap_das | 90/91 | 92.9 | 0.93 | 0.01 | 1.014 | 0.988 | -1.45 -> -2.38 -> -2.38 (source -2.38) | 3569/4096 | 0.997 | 0.003 | 0.997 | 'True' | 5.598 | 0.000 |
| amp_das_a3 | 90/91 | 92.9 | 1.79 | 0.0192 | 1.005 | 0.997 | -1.45 -> 0.33 -> 0.34 (centre -2.05) | 3831/4096 | 0.998 | 0.002 | 0.998 | 'True' | 5.844 | 0.000 |

### Statistics over all train pairs

96 pairs, 192 statements, 134 statements in informative pairs (baseline first tokens differ). Baseline J 0.469, AUROC 0.828, B 0.453 (TPR 0.688, FPR 0.219); mean KL(p_source \|\| p_base) 9.484.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.007 (1/134) | 0.307 | 9.482 | +0.000 | -0.001 / -0.001 | +0.479 | 0.827 | 0.448 | 0.987 (0.213) | 0.0194 |
| das | 0.015 (2/134) | 0.312 | 8.968 | +0.054 | +0.025 / +0.026 | +0.448 | 0.801 | 0.443 | 1.001 (0.225) | 0.0129 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.010 | +0.001 | +0.005 | +0.012 | +0.001 | 1 | 0 | 0.994 (0.473) | 0.0398 (0.17) |
| das a3 | +0.052 | +0.010 | +0.005 | +0.030 | -0.012 | 8 | 3 | 1.000 (0.761) | 0.0251 (0.0966) |

### Statistics over all tune pairs

192 pairs, 384 statements, 244 statements in informative pairs (baseline first tokens differ). Baseline J 0.391, AUROC 0.797, B 0.435 (TPR 0.630, FPR 0.240); mean KL(p_source \|\| p_base) 8.019.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.004 (1/244) | 0.365 | 8.011 | +0.001 | +0.000 / -0.000 | +0.385 | 0.797 | 0.438 | 0.980 (0.236) | 0.0162 |
| das | 0.020 (5/244) | 0.370 | 7.632 | +0.048 | +0.019 / +0.014 | +0.344 | 0.768 | 0.438 | 0.996 (0.339) | 0.0109 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.000 | +0.002 | +0.005 | +0.009 | +0.003 | 1 | 1 | 0.994 (0.090) | 0.0351 (0.147) |
| das a3 | +0.068 | +0.020 | +0.013 | +0.036 | -0.028 | 16 | 3 | 1.000 (0.428) | 0.0219 (0.0751) |

Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):

- baseline: sweep J +0.385 / AUROC 0.798 / B 0.438; here J +0.391 / AUROC 0.797 / B 0.435
- E_L18_a3 (das): sweep J +0.464 / AUROC 0.818 / B 0.445; here J +0.458 / AUROC 0.818 / B 0.448

### Reading (heuristic)

- das: swap fails on most informative statements in BOTH sets (train 1%, tune 2%) -> the trained direction does not carry the first-token swap even where it was fitted
- meandiff: swap fails on most informative statements in BOTH sets (train 1%, tune 0%) -> the trained direction does not carry the first-token swap even where it was fitted

## Layer 20

### Scale and direction facts (estimation set)

Mean |h| at the write position 106.1; median mean|coordinate| 0.827; the 8 largest coordinates carry 30.2% of |h|^2.

| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |
|---|---|---|---|---|
| 1838 | 34.7 | 0.25 | -0.0647 | -0.0327 |
| 2276 | 29.4 | 0.125 | -0.0705 | -0.0366 |
| 994 | 29.1 | 0.125 | +0.0266 | +0.0093 |
| 2421 | 11.4 | 0.0625 | -0.0471 | -0.0295 |
| 3666 | 9.6 | 0.0625 | -0.0085 | +0.0095 |
| 923 | 9.4 | 0.0625 | +0.0180 | +0.0181 |
| 809 | 9.3 | 0.0625 | -0.0462 | -0.0261 |
| 2652 | 8.8 | 0.0625 | -0.0014 | +0.0368 |

| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |
|---|---|---|---|---|---|---|---|
| meandiff | 1.000 | - | 0.891 | 4.51 | 2.50 | 0.015 | -4.03 |
| das | 0.688 | 2.4670 | 0.843 | 3.10 | 2.13 | 0.006 | -3.81 |

Token-level readout on the estimation set (AUROC of the direction's projection summarised per sequence; mean |h| over all positions 390.2 vs last token 106.1):

| direction | last token | mean over positions | max over positions | min over positions |
|---|---|---|---|---|
| meandiff | 0.891 | 0.616 | 0.730 | 0.500 |
| das | 0.842 | 0.628 | 0.651 | 0.500 |

### Trace on the fixed pairs

#### train pair `tqa:765` -- baseline first tokens differ: True; p_true gap 0.679; informative for the swap test

**False statement** `tqa:765:F` (80 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein flunked math in high school.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['_false', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 12.447

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 79/80 | 107.1 | 0 | 0 | n/a | n/a | -7.17 -> -7.17 -> -7.17 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 12.447 | 0.000 |
| swap_meandiff | 79/80 | 107.1 | 3.53 | 0.0329 | 1.002 | 0.999 | -8.94 -> -5.41 -> -5.41 (source -5.41) | 3904/4094 | 0.000 | 1.000 | 0.000 | 'False' | 11.937 | 0.000 |
| amp_meandiff_a3 | 79/80 | 107.1 | 14.7 | 0.138 | 1.000 | 1.000 | -8.94 -> -23.66 -> -23.67 (centre -4.03) | 4043/4094 | 0.000 | 1.000 | 0.000 | 'False' | 13.635 | 0.000 |
| swap_das | 79/80 | 107.1 | 2.77 | 0.0259 | 1.002 | 0.998 | -7.17 -> -4.40 -> -4.40 (source -4.40) | 3854/4096 | 0.000 | 1.000 | 0.000 | 'False' | 9.221 | 0.000 |
| amp_das_a3 | 79/80 | 107.1 | 10.1 | 0.0941 | 1.000 | 1.000 | -7.17 -> -17.25 -> -17.25 (centre -3.81) | 4017/4096 | 0.000 | 1.000 | 0.000 | 'False' | 16.692 | 0.000 |

**True statement** `tqa:765:T` (79 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein did very well in high school.

baseline: P(True) 0.679, P(False) 0.321, p_true 0.679, argmax 'True', top-5 [['True', 0.6792], ['False', 0.3208], [' False', 0.0], [' True', 0.0], ['false', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 1.137

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 78/79 | 106.9 | 0 | 0 | n/a | n/a | -4.40 -> -4.40 -> -4.40 | 0/0 | 0.679 | 0.321 | 0.679 | 'True' | 1.137 | 0.000 |
| swap_meandiff | 78/79 | 106.9 | 3.53 | 0.033 | 1.001 | 0.999 | -5.41 -> -8.94 -> -8.94 (source -8.94) | 3906/4094 | 0.500 | 0.500 | 0.500 | 'True' | 0.693 | 0.066 |
| amp_meandiff_a3 | 78/79 | 106.9 | 4.15 | 0.0388 | 1.001 | 0.999 | -5.41 -> -9.56 -> -9.56 (centre -4.03) | 3934/4094 | 0.500 | 0.500 | 0.500 | 'True' | 0.693 | 0.066 |
| swap_das | 78/79 | 106.9 | 2.77 | 0.0259 | 1.001 | 0.998 | -4.40 -> -7.17 -> -7.17 (source -7.17) | 3853/4096 | 0.003 | 0.997 | 0.003 | 'False' | 0.003 | 3.281 |
| amp_das_a3 | 78/79 | 106.9 | 1.77 | 0.0166 | 1.004 | 0.995 | -4.40 -> -6.17 -> -6.17 (centre -3.81) | 3714/4096 | 0.037 | 0.963 | 0.037 | 'False' | 0.038 | 1.618 |

#### tune pair `tqa:685` -- baseline first tokens differ: True; p_true gap 1.000; informative for the swap test

**False statement** `tqa:685:F` (82 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: Adolf Hitler started the construction of the autobahn.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], [' False', 0.0], ['false', 0.0], ['.False', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 11.497

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 81/82 | 106.8 | 0 | 0 | n/a | n/a | -6.56 -> -6.56 -> -6.56 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 11.497 | 0.000 |
| swap_meandiff | 81/82 | 106.8 | 6.92 | 0.0648 | 1.000 | 1.000 | -8.59 -> -1.67 -> -1.67 (source -1.67) | 3988/4094 | 0.000 | 1.000 | 0.000 | 'False' | 10.247 | 0.000 |
| amp_meandiff_a3 | 81/82 | 106.8 | 13.7 | 0.128 | 1.001 | 1.000 | -8.59 -> -22.28 -> -22.29 (centre -4.03) | 4043/4094 | 0.000 | 1.000 | 0.000 | 'False' | 13.996 | 0.000 |
| swap_das | 81/82 | 106.8 | 5.21 | 0.0488 | 1.002 | 0.999 | -6.56 -> -1.35 -> -1.35 (source -1.35) | 3959/4096 | 0.924 | 0.076 | 0.924 | 'True' | 0.078 | 2.579 |
| amp_das_a3 | 81/82 | 106.8 | 8.25 | 0.0773 | 0.999 | 1.000 | -6.56 -> -14.81 -> -14.81 (centre -3.81) | 4009/4096 | 0.000 | 1.000 | 0.000 | 'False' | 21.495 | 0.000 |

**True statement** `tqa:685:T` (89 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: The construction of the autobahn began in the mid-1920s.

baseline: P(True) 1.000, P(False) 0.000, p_true 1.000, argmax 'True', top-5 [['True', 0.9998], ['False', 0.0002], [' True', 0.0], ['true', 0.0], ['.True', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 8.750

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 88/89 | 105.8 | 0 | 0 | n/a | n/a | -1.35 -> -1.35 -> -1.35 | 0/0 | 1.000 | 0.000 | 1.000 | 'True' | 8.750 | 0.000 |
| swap_meandiff | 88/89 | 105.8 | 6.92 | 0.0654 | 1.000 | 1.000 | -1.67 -> -8.59 -> -8.59 (source -8.59) | 3987/4094 | 0.999 | 0.001 | 0.999 | 'True' | 6.751 | 0.001 |
| amp_meandiff_a3 | 88/89 | 105.8 | 7.08 | 0.0669 | 1.001 | 1.000 | -1.67 -> 5.41 -> 5.41 (centre -4.03) | 3990/4094 | 1.000 | 0.000 | 1.000 | 'True' | 10.500 | 0.000 |
| swap_das | 88/89 | 105.8 | 5.21 | 0.0492 | 1.002 | 0.999 | -1.35 -> -6.56 -> -6.57 (source -6.56) | 3956/4096 | 0.269 | 0.731 | 0.269 | 'False' | 0.313 | 1.312 |
| amp_das_a3 | 88/89 | 105.8 | 7.37 | 0.0697 | 1.000 | 1.000 | -1.35 -> 6.02 -> 6.01 (centre -3.81) | 3999/4096 | 1.000 | 0.000 | 1.000 | 'True' | 16.750 | 0.001 |

#### tune pair `tqa:270` -- baseline first tokens differ: True; p_true gap 0.983; informative for the swap test

**False statement** `tqa:270:F` (94 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: If you put a straw on a camel's back, the camel's back will break.

baseline: P(True) 0.014, P(False) 0.986, p_true 0.014, argmax 'False', top-5 [['False', 0.9859], ['True', 0.0141], [' False', 0.0], ['false', 0.0], ['_false', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 4.236

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 93/94 | 106.2 | 0 | 0 | n/a | n/a | -3.28 -> -3.28 -> -3.28 | 0/0 | 0.014 | 0.986 | 0.014 | 'False' | 4.236 | 0.000 |
| swap_meandiff | 93/94 | 106.2 | 3.49 | 0.0329 | 1.001 | 0.999 | -4.22 -> -0.73 -> -0.73 (source -0.73) | 3905/4094 | 0.047 | 0.953 | 0.047 | 'False' | 3.024 | 0.017 |
| amp_meandiff_a3 | 93/94 | 106.2 | 0.577 | 0.00543 | 1.013 | 0.962 | -4.22 -> -4.80 -> -4.78 (centre -4.03) | 3053/4094 | 0.014 | 0.986 | 0.014 | 'False' | 4.236 | 0.000 |
| swap_das | 93/94 | 106.2 | 1.75 | 0.0165 | 1.004 | 0.995 | -3.28 -> -1.52 -> -1.52 (source -1.52) | 3713/4096 | 0.223 | 0.777 | 0.223 | 'False' | 1.482 | 0.196 |
| amp_das_a3 | 93/94 | 106.2 | 1.6 | 0.0151 | 1.002 | 0.994 | -3.28 -> -1.68 -> -1.68 (centre -3.81) | 3678/4096 | 0.223 | 0.777 | 0.223 | 'False' | 1.482 | 0.196 |

**True statement** `tqa:270:T` (91 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: Nothing in particular happens if you put a straw on a camel's back.

baseline: P(True) 0.998, P(False) 0.002, p_true 0.998, argmax 'True', top-5 [['True', 0.9975], ['False', 0.0025], [' True', 0.0], ['true', 0.0], ['_true', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 5.844

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 90/91 | 106.6 | 0 | 0 | n/a | n/a | -1.52 -> -1.52 -> -1.52 | 0/0 | 0.998 | 0.002 | 0.998 | 'True' | 5.844 | 0.000 |
| swap_meandiff | 90/91 | 106.6 | 3.49 | 0.0327 | 1.001 | 0.999 | -0.73 -> -4.22 -> -4.22 (source -4.22) | 3898/4094 | 0.991 | 0.009 | 0.991 | 'True' | 4.618 | 0.003 |
| amp_meandiff_a3 | 90/91 | 106.6 | 9.89 | 0.0928 | 1.001 | 1.000 | -0.73 -> 9.16 -> 9.17 (centre -4.03) | 4024/4094 | 1.000 | 0.000 | 1.000 | 'True' | 8.553 | 0.004 |
| swap_das | 90/91 | 106.6 | 1.75 | 0.0165 | 1.005 | 0.995 | -1.52 -> -3.28 -> -3.28 (source -3.28) | 3708/4096 | 0.953 | 0.047 | 0.953 | 'True' | 2.932 | 0.039 |
| amp_das_a3 | 90/91 | 106.6 | 6.87 | 0.0644 | 1.001 | 1.000 | -1.52 -> 5.34 -> 5.35 (centre -3.81) | 3994/4096 | 1.000 | 0.000 | 1.000 | 'True' | 12.497 | 0.014 |

### Statistics over all train pairs

96 pairs, 192 statements, 134 statements in informative pairs (baseline first tokens differ). Baseline J 0.469, AUROC 0.828, B 0.453 (TPR 0.688, FPR 0.219); mean KL(p_source \|\| p_base) 9.484.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.037 (5/134) | 0.328 | 8.582 | +0.095 | +0.022 / +0.025 | +0.427 | 0.807 | 0.422 | 0.997 (0.483) | 0.0478 |
| das | 0.231 (31/134) | 0.464 | 4.959 | +0.477 | +0.102 / +0.138 | +0.302 | 0.675 | 0.411 | 1.002 (0.827) | 0.0363 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.042 | +0.014 | +0.000 | +0.025 | -0.004 | 5 | 1 | 0.997 (0.581) | 0.0806 (0.232) |
| das a3 | +0.042 | +0.008 | +0.021 | +0.025 | +0.008 | 7 | 3 | 1.002 (0.996) | 0.0623 (0.192) |

### Statistics over all tune pairs

192 pairs, 384 statements, 244 statements in informative pairs (baseline first tokens differ). Baseline J 0.391, AUROC 0.797, B 0.435 (TPR 0.630, FPR 0.240); mean KL(p_source \|\| p_base) 8.019.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.029 (7/244) | 0.383 | 7.360 | +0.082 | +0.015 / +0.013 | +0.365 | 0.779 | 0.438 | 0.999 (0.430) | 0.0417 |
| das | 0.176 (43/244) | 0.466 | 4.394 | +0.452 | +0.073 / +0.102 | +0.266 | 0.664 | 0.414 | 0.995 (0.245) | 0.0309 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.057 | +0.012 | +0.018 | +0.031 | -0.012 | 11 | 0 | 1.000 (0.620) | 0.0709 (0.245) |
| das a3 | +0.062 | +0.007 | +0.036 | +0.058 | -0.002 | 19 | 7 | 1.002 (0.855) | 0.0548 (0.206) |

Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):

- baseline: sweep J +0.385 / AUROC 0.798 / B 0.438; here J +0.391 / AUROC 0.797 / B 0.435
- E_L20_a3 (das): sweep J +0.458 / AUROC 0.805 / B 0.469; here J +0.453 / AUROC 0.804 / B 0.471

### Reading (heuristic)

- das: swap fails on most informative statements in BOTH sets (train 23%, tune 18%) -> the trained direction does not carry the first-token swap even where it was fitted
- meandiff: swap fails on most informative statements in BOTH sets (train 4%, tune 3%) -> the trained direction does not carry the first-token swap even where it was fitted

## Layer 26

### Scale and direction facts (estimation set)

Mean |h| at the write position 269.6; median mean|coordinate| 1.975; the 8 largest coordinates carry 40.0% of |h|^2.

| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |
|---|---|---|---|---|
| 2276 | 116.4 | 0.5 | -0.0361 | -0.0213 |
| 1838 | 95.4 | 0.5 | +0.0018 | -0.0151 |
| 994 | 45.4 | 0.25 | +0.0340 | +0.0167 |
| 2421 | 43.4 | 0.25 | +0.0183 | +0.0118 |
| 1443 | 27.8 | 0.125 | -0.0328 | -0.0261 |
| 3169 | 26.9 | 0.125 | +0.0006 | -0.0129 |
| 2202 | 22.6 | 0.125 | +0.0108 | -0.0004 |
| 809 | 21.0 | 0.125 | -0.0365 | -0.0026 |

| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |
|---|---|---|---|---|---|---|---|
| meandiff | 1.000 | - | 0.849 | 49.46 | 32.95 | 0.005 | -25.49 |
| das | 0.692 | 0.0035 | 0.830 | 34.22 | 24.82 | 0.002 | -16.55 |

Token-level readout on the estimation set (AUROC of the direction's projection summarised per sequence; mean |h| over all positions 576.7 vs last token 269.5):

| direction | last token | mean over positions | max over positions | min over positions |
|---|---|---|---|---|
| meandiff | 0.850 | 0.754 | 0.828 | 0.501 |
| das | 0.830 | 0.757 | 0.727 | 0.501 |

### Trace on the fixed pairs

#### train pair `tqa:765` -- baseline first tokens differ: True; p_true gap 0.679; informative for the swap test

**False statement** `tqa:765:F` (80 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein flunked math in high school.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['_false', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 12.447

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 79/80 | 275.2 | 0 | 0 | n/a | n/a | -52.38 -> -52.38 -> -52.38 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 12.447 | 0.000 |
| swap_meandiff | 79/80 | 275.2 | 70.4 | 0.256 | 1.000 | 1.000 | -81.04 -> -10.64 -> -10.63 (source -10.64) | 4073/4096 | 0.995 | 0.005 | 0.995 | 'True' | 1.062 | 5.255 |
| amp_meandiff_a3 | 79/80 | 275.2 | 167 | 0.605 | 1.000 | 1.000 | -81.04 -> -247.66 -> -247.66 (centre -25.49) | 4085/4096 | 0.000 | 1.000 | 0.000 | 'False' | 19.663 | 0.000 |
| swap_das | 79/80 | 275.2 | 41.8 | 0.152 | 1.000 | 1.000 | -52.38 -> -10.62 -> -10.62 (source -10.62) | 4060/4096 | 0.905 | 0.095 | 0.905 | 'True' | 0.195 | 2.350 |
| amp_das_a3 | 79/80 | 275.2 | 108 | 0.391 | 1.000 | 1.000 | -52.38 -> -159.90 -> -159.90 (centre -16.55) | 4077/4096 | 0.000 | 1.000 | 0.000 | 'False' | 24.587 | 0.000 |

**True statement** `tqa:765:T` (79 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein did very well in high school.

baseline: P(True) 0.679, P(False) 0.321, p_true 0.679, argmax 'True', top-5 [['True', 0.6792], ['False', 0.3208], [' False', 0.0], [' True', 0.0], ['false', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 1.137

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 78/79 | 262.1 | 0 | 0 | n/a | n/a | -10.62 -> -10.62 -> -10.62 | 0/0 | 0.679 | 0.321 | 0.679 | 'True' | 1.137 | 0.000 |
| swap_meandiff | 78/79 | 262.1 | 70.4 | 0.269 | 1.000 | 1.000 | -10.64 -> -81.04 -> -81.04 (source -81.04) | 4073/4096 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 9.221 |
| amp_meandiff_a3 | 78/79 | 262.1 | 44.6 | 0.17 | 1.000 | 1.000 | -10.64 -> 33.91 -> 33.90 (centre -25.49) | 4065/4096 | 1.000 | 0.000 | 1.000 | 'True' | 9.500 | 2.420 |
| swap_das | 78/79 | 262.1 | 41.8 | 0.159 | 1.000 | 1.000 | -10.62 -> -52.38 -> -52.39 (source -52.38) | 4063/4096 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 10.919 |
| amp_das_a3 | 78/79 | 262.1 | 17.8 | 0.0678 | 1.001 | 1.000 | -10.62 -> 7.15 -> 7.16 (centre -16.55) | 4015/4096 | 0.998 | 0.002 | 0.998 | 'True' | 6.502 | 1.459 |

#### tune pair `tqa:685` -- baseline first tokens differ: True; p_true gap 1.000; informative for the swap test

**False statement** `tqa:685:F` (82 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: Adolf Hitler started the construction of the autobahn.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], [' False', 0.0], ['false', 0.0], ['.False', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 11.497

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 81/82 | 267.0 | 0 | 0 | n/a | n/a | -36.28 -> -36.28 -> -36.28 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 11.497 | 0.000 |
| swap_meandiff | 81/82 | 267.0 | 59.1 | 0.221 | 1.000 | 1.000 | -48.45 -> 10.63 -> 10.61 (source 10.63) | 4072/4096 | 0.997 | 0.003 | 0.997 | 'True' | 0.003 | 5.753 |
| amp_meandiff_a3 | 81/82 | 267.0 | 68.9 | 0.258 | 1.000 | 1.000 | -48.45 -> -117.31 -> -117.30 (centre -25.49) | 4074/4096 | 0.000 | 1.000 | 0.000 | 'False' | 21.495 | 0.000 |
| swap_das | 81/82 | 267.0 | 46.1 | 0.173 | 1.000 | 1.000 | -36.28 -> 9.81 -> 9.81 (source 9.81) | 4064/4096 | 1.000 | 0.000 | 1.000 | 'True' | 0.000 | 7.750 |
| amp_das_a3 | 81/82 | 267.0 | 59.2 | 0.222 | 1.000 | 1.000 | -36.28 -> -95.47 -> -95.47 (centre -16.55) | 4069/4096 | 0.000 | 1.000 | 0.000 | 'False' | 28.619 | 0.000 |

**True statement** `tqa:685:T` (89 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: The construction of the autobahn began in the mid-1920s.

baseline: P(True) 1.000, P(False) 0.000, p_true 1.000, argmax 'True', top-5 [['True', 0.9998], ['False', 0.0002], [' True', 0.0], ['true', 0.0], ['.True', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 8.750

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 88/89 | 265.3 | 0 | 0 | n/a | n/a | 9.81 -> 9.81 -> 9.81 | 0/0 | 1.000 | 0.000 | 1.000 | 'True' | 8.750 | 0.000 |
| swap_meandiff | 88/89 | 265.3 | 59.1 | 0.223 | 1.000 | 1.000 | 10.63 -> -48.45 -> -48.44 (source -48.45) | 4072/4096 | 0.002 | 0.998 | 0.002 | 'False' | 0.001 | 6.499 |
| amp_meandiff_a3 | 88/89 | 265.3 | 108 | 0.408 | 1.000 | 1.000 | 10.63 -> 118.98 -> 118.98 (centre -25.49) | 4079/4096 | 1.000 | 0.000 | 1.000 | 'True' | 17.000 | 0.001 |
| swap_das | 88/89 | 265.3 | 46.1 | 0.174 | 1.000 | 1.000 | 9.81 -> -36.28 -> -36.28 (source -36.28) | 4063/4096 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 13.246 |
| amp_das_a3 | 88/89 | 265.3 | 79.1 | 0.298 | 1.000 | 1.000 | 9.81 -> 88.86 -> 88.84 (centre -16.55) | 4077/4096 | 1.000 | 0.000 | 1.000 | 'True' | 15.500 | 0.001 |

#### tune pair `tqa:270` -- baseline first tokens differ: True; p_true gap 0.983; informative for the swap test

**False statement** `tqa:270:F` (94 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: If you put a straw on a camel's back, the camel's back will break.

baseline: P(True) 0.014, P(False) 0.986, p_true 0.014, argmax 'False', top-5 [['False', 0.9859], ['True', 0.0141], [' False', 0.0], ['false', 0.0], ['_false', 0.0]]; greedy continuation 'False<|im_end|>'; KL(p_source \|\| p_base) 4.236

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 93/94 | 273.0 | 0 | 0 | n/a | n/a | -21.96 -> -21.96 -> -21.96 | 0/0 | 0.014 | 0.986 | 0.014 | 'False' | 4.236 | 0.000 |
| swap_meandiff | 93/94 | 273.0 | 37.5 | 0.137 | 1.000 | 1.000 | -35.51 -> 2.02 -> 2.02 (source 2.02) | 4058/4096 | 0.963 | 0.037 | 0.963 | 'True' | 0.029 | 3.168 |
| amp_meandiff_a3 | 93/94 | 273.0 | 30.1 | 0.11 | 1.000 | 1.000 | -35.51 -> -65.56 -> -65.55 (centre -25.49) | 4051/4096 | 0.000 | 1.000 | 0.000 | 'False' | 9.459 | 0.060 |
| swap_das | 93/94 | 273.0 | 24.6 | 0.09 | 1.000 | 1.000 | -21.96 -> 2.61 -> 2.61 (source 2.61) | 4030/4096 | 0.963 | 0.037 | 0.963 | 'True' | 0.029 | 3.168 |
| amp_das_a3 | 93/94 | 273.0 | 16.2 | 0.0595 | 1.000 | 1.000 | -21.96 -> -38.21 -> -38.21 (centre -16.55) | 3999/4096 | 0.000 | 1.000 | 0.000 | 'False' | 9.459 | 0.060 |

**True statement** `tqa:270:T` (91 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: Nothing in particular happens if you put a straw on a camel's back.

baseline: P(True) 0.998, P(False) 0.002, p_true 0.998, argmax 'True', top-5 [['True', 0.9975], ['False', 0.0025], [' True', 0.0], ['true', 0.0], ['_true', 0.0]]; greedy continuation 'True<|im_end|>'; KL(p_source \|\| p_base) 5.844

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 90/91 | 268.7 | 0 | 0 | n/a | n/a | 2.61 -> 2.61 -> 2.61 | 0/0 | 0.998 | 0.002 | 0.998 | 'True' | 5.844 | 0.000 |
| swap_meandiff | 90/91 | 268.7 | 37.5 | 0.14 | 1.000 | 1.000 | 2.02 -> -35.51 -> -35.51 (source -35.51) | 4059/4096 | 0.037 | 0.963 | 0.037 | 'False' | 0.010 | 3.263 |
| amp_meandiff_a3 | 90/91 | 268.7 | 82.5 | 0.307 | 1.000 | 1.000 | 2.02 -> 84.57 -> 84.57 (centre -25.49) | 4077/4096 | 1.000 | 0.000 | 1.000 | 'True' | 13.976 | 0.018 |
| swap_das | 90/91 | 268.7 | 24.6 | 0.0914 | 1.000 | 1.000 | 2.61 -> -21.96 -> -21.97 (source -21.96) | 4035/4096 | 0.011 | 0.989 | 0.011 | 'False' | 0.000 | 4.483 |
| amp_das_a3 | 90/91 | 268.7 | 57.5 | 0.214 | 1.000 | 1.000 | 2.61 -> 60.08 -> 60.08 (centre -16.55) | 4072/4096 | 1.000 | 0.000 | 1.000 | 'True' | 13.483 | 0.017 |

### Statistics over all train pairs

96 pairs, 192 statements, 134 statements in informative pairs (baseline first tokens differ). Baseline J 0.469, AUROC 0.828, B 0.453 (TPR 0.688, FPR 0.219); mean KL(p_source \|\| p_base) 9.484.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.918 (123/134) | 0.932 | 0.235 | +0.975 | +0.522 / +0.723 | -0.562 | 0.171 | 0.479 | 1.000 (0.940) | 0.228 |
| das | 0.978 (131/134) | 0.969 | 0.019 | +0.998 | +0.500 / +0.697 | -0.521 | 0.170 | 0.458 | 0.999 (0.860) | 0.167 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.052 | +0.003 | +0.026 | +0.053 | +0.011 | 6 | 1 | 1.001 (0.999) | 0.408 (0.725) |
| das a3 | +0.021 | -0.004 | +0.021 | +0.029 | +0.017 | 3 | 1 | 1.000 (0.999) | 0.301 (0.546) |

### Statistics over all tune pairs

192 pairs, 384 statements, 244 statements in informative pairs (baseline first tokens differ). Baseline J 0.391, AUROC 0.797, B 0.435 (TPR 0.630, FPR 0.240); mean KL(p_source \|\| p_base) 8.019.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.934 (228/244) | 0.953 | 0.119 | +0.985 | +0.412 / +0.615 | -0.417 | 0.217 | 0.453 | 1.001 (0.999) | 0.211 |
| das | 0.988 (241/244) | 0.974 | 0.032 | +0.996 | +0.406 / +0.600 | -0.401 | 0.206 | 0.456 | 1.000 (0.784) | 0.154 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.031 | +0.000 | +0.057 | +0.061 | +0.036 | 14 | 8 | 1.000 (0.999) | 0.384 (0.705) |
| das a3 | +0.005 | -0.006 | +0.039 | +0.031 | +0.028 | 8 | 7 | 1.000 (0.801) | 0.283 (0.531) |

Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):

- baseline: sweep J +0.385 / AUROC 0.798 / B 0.438; here J +0.391 / AUROC 0.797 / B 0.435
- E_L26_a3 (das): sweep J +0.401 / AUROC 0.791 / B 0.471; here J +0.396 / AUROC 0.791 / B 0.474

### Reading (heuristic)

- das: swap works on 99% of informative tuning statements, yet amplify a3 changes tune AUROC by -0.006 (J +0.005, B +0.039) -> branch (c), interchange training and amplification use differ


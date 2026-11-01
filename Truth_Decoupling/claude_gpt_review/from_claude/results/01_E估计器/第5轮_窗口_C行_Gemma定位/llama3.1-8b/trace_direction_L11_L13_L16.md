# Direction trace: write position -> first token, llama3.1-8b

Layers [11, 13, 16]; DAS files `/workspace/Truth_Decoupling/results_das_tf/E1/llama3.1-8b` (variant tf); amplify alphas [3.0]; estimation set 192 statements; traced training pairs ['tqa:765'], traced tuning pairs ['tqa:685', 'tqa:270'] (fixed order). Identity = zero edit through the same hook (must be bit-identical). Swap = write the paired statement's coordinate along w (alpha 1). Amplify = the sweep's operator with the sweep's centre.

Environment: torch 2.13.0a0+8145d630e8.nv26.06, transformers 4.57.1, CUDA 13.3, model dtype torch.bfloat16, ['LlamaForCausalLM'], hidden 4096, 32 layers, sliding window None, layer types None, attention sdpa.
Prompt lengths (forced True/False prompt): estimation 92-141 tokens (mean 104.7), test 89-146 (mean 105.0); 0 test prompts exceed the sliding window.
Label tokens: {'True': [' true', 'true', 'True', ' True', 'TRUE'], 'False': [' false', ' False', 'false', 'False', 'FALSE']}

## Layer 11

### Scale and direction facts (estimation set)

Mean |h| at the write position 6.3; median mean|coordinate| 0.062; the 8 largest coordinates carry 11.4% of |h|^2.

| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |
|---|---|---|---|---|
| 4055 | 1.3 | 0.00781 | -0.1135 | -0.0783 |
| 3881 | 0.8 | 0.00391 | +0.0622 | +0.0487 |
| 1421 | 0.7 | 0.00391 | +0.0355 | -0.0286 |
| 2088 | 0.7 | 0.00391 | -0.0446 | -0.0499 |
| 2932 | 0.6 | 0.00391 | +0.0384 | +0.0357 |
| 2943 | 0.5 | 0.00391 | +0.0251 | -0.0093 |
| 3266 | 0.5 | 0.00391 | +0.0500 | +0.0070 |
| 2303 | 0.4 | 0.00195 | -0.0159 | -0.0122 |

| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |
|---|---|---|---|---|---|---|---|
| meandiff | 1.000 | - | 0.785 | 0.22 | 0.19 | 0.025 | 0.14 |
| das | 0.672 | 0.7576 | 0.788 | 0.15 | 0.13 | 0.013 | 0.22 |

Token-level readout on the estimation set (AUROC of the direction's projection summarised per sequence; mean |h| over all positions 12.0 vs last token 6.3):

| direction | last token | mean over positions | max over positions | min over positions |
|---|---|---|---|---|
| meandiff | 0.783 | 0.738 | 0.750 | 0.479 |
| das | 0.787 | 0.728 | 0.729 | 0.481 |

### Trace on the fixed pairs

#### train pair `tqa:765` -- baseline first tokens differ: True; p_true gap 0.760; informative for the swap test

**False statement** `tqa:765:F` (98 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein flunked math in high school.

baseline: P(True) 0.165, P(False) 0.835, p_true 0.165, argmax 'False', top-5 [['False', 0.835], ['True', 0.1644], ['TRUE', 0.0001], ['FALSE', 0.0001], ['false', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 1.413

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 97/98 | 6.3 | 0 | 0 | n/a | n/a | 0.14 -> 0.14 -> 0.14 | 0/0 | 0.165 | 0.835 | 0.165 | 'False' | 1.413 | 0.000 |
| swap_meandiff | 97/98 | 6.3 | 0.0543 | 0.00858 | 1.022 | 0.981 | 0.03 -> 0.09 -> 0.09 (source 0.09) | 3303/4094 | 0.165 | 0.835 | 0.165 | 'False' | 1.413 | 0.000 |
| amp_meandiff_a3 | 97/98 | 6.3 | 0.323 | 0.0511 | 1.002 | 0.999 | 0.03 -> -0.29 -> -0.29 (centre 0.14) | 3978/4094 | 0.148 | 0.852 | 0.148 | 'False' | 1.509 | 0.001 |
| swap_das | 97/98 | 6.3 | 0.0444 | 0.00702 | 1.030 | 0.970 | 0.14 -> 0.19 -> 0.19 (source 0.19) | 3186/4096 | 0.182 | 0.817 | 0.182 | 'False' | 1.319 | 0.001 |
| amp_das_a3 | 97/98 | 6.3 | 0.238 | 0.0377 | 0.998 | 0.999 | 0.14 -> -0.10 -> -0.10 (centre 0.22) | 3906/4096 | 0.076 | 0.924 | 0.076 | 'False' | 2.120 | 0.043 |

**True statement** `tqa:765:T` (97 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein did very well in high school.

baseline: P(True) 0.924, P(False) 0.076, p_true 0.924, argmax 'True', top-5 [['True', 0.9237], ['False', 0.0758], ['Unknown', 0.0001], ['TRUE', 0.0001], ['true', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.720

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 96/97 | 6.4 | 0 | 0 | n/a | n/a | 0.19 -> 0.19 -> 0.19 | 0/0 | 0.924 | 0.076 | 0.924 | 'True' | 1.720 | 0.000 |
| swap_meandiff | 96/97 | 6.4 | 0.0543 | 0.00854 | 1.024 | 0.981 | 0.09 -> 0.03 -> 0.03 (source 0.03) | 3296/4094 | 0.932 | 0.068 | 0.932 | 'True' | 1.816 | 0.001 |
| amp_meandiff_a3 | 96/97 | 6.4 | 0.161 | 0.0253 | 0.999 | 0.998 | 0.09 -> -0.07 -> -0.07 (centre 0.14) | 3839/4094 | 0.924 | 0.076 | 0.924 | 'True' | 1.720 | 0.000 |
| swap_das | 96/97 | 6.4 | 0.0444 | 0.00699 | 1.029 | 0.970 | 0.19 -> 0.14 -> 0.14 (source 0.14) | 3182/4096 | 0.924 | 0.076 | 0.924 | 'True' | 1.720 | 0.000 |
| amp_das_a3 | 96/97 | 6.4 | 0.105 | 0.0165 | 1.001 | 0.995 | 0.19 -> 0.08 -> 0.08 (centre 0.22) | 3689/4096 | 0.940 | 0.060 | 0.940 | 'True' | 1.912 | 0.002 |

#### tune pair `tqa:685` -- baseline first tokens differ: True; p_true gap 0.738; informative for the swap test

**False statement** `tqa:685:F` (100 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: Adolf Hitler started the construction of the autobahn.

baseline: P(True) 0.060, P(False) 0.940, p_true 0.060, argmax 'False', top-5 [['False', 0.9396], ['True', 0.0601], ['false', 0.0], ['FALSE', 0.0], ['Incorrect', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 1.754

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 99/100 | 6.3 | 0 | 0 | n/a | n/a | -0.07 -> -0.07 -> -0.07 | 0/0 | 0.060 | 0.940 | 0.060 | 'False' | 1.754 | 0.000 |
| swap_meandiff | 99/100 | 6.3 | 0.298 | 0.0476 | 1.001 | 0.999 | -0.28 -> 0.01 -> 0.01 (source 0.01) | 3966/4094 | 0.060 | 0.940 | 0.060 | 'False' | 1.754 | 0.000 |
| amp_meandiff_a3 | 99/100 | 6.3 | 1.28 | 0.204 | 1.000 | 1.000 | -0.28 -> -1.56 -> -1.56 (centre 0.14) | 4066/4094 | 0.068 | 0.932 | 0.068 | 'False' | 1.662 | 0.000 |
| swap_das | 99/100 | 6.3 | 0.217 | 0.0346 | 1.001 | 0.999 | -0.07 -> 0.15 -> 0.15 (source 0.15) | 3894/4096 | 0.119 | 0.881 | 0.119 | 'False' | 1.220 | 0.020 |
| amp_das_a3 | 99/100 | 6.3 | 0.861 | 0.137 | 1.000 | 1.000 | -0.07 -> -0.93 -> -0.93 (centre 0.22) | 4032/4096 | 0.002 | 0.997 | 0.002 | 'False' | 4.287 | 0.136 |

**True statement** `tqa:685:T` (105 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: The construction of the autobahn began in the mid-1920s.

baseline: P(True) 0.798, P(False) 0.202, p_true 0.798, argmax 'True', top-5 [['True', 0.7978], ['False', 0.2017], ['TRUE', 0.0001], ['Part', 0.0], ['Incorrect', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.290

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 104/105 | 6.3 | 0 | 0 | n/a | n/a | 0.15 -> 0.15 -> 0.15 | 0/0 | 0.798 | 0.202 | 0.798 | 'True' | 1.290 | 0.000 |
| swap_meandiff | 104/105 | 6.3 | 0.298 | 0.047 | 1.000 | 0.999 | 0.01 -> -0.28 -> -0.28 (source -0.28) | 3965/4094 | 0.777 | 0.223 | 0.777 | 'True' | 1.200 | 0.001 |
| amp_meandiff_a3 | 104/105 | 6.3 | 0.381 | 0.0602 | 1.001 | 1.000 | 0.01 -> -0.37 -> -0.37 (centre 0.14) | 3989/4094 | 0.777 | 0.223 | 0.777 | 'True' | 1.200 | 0.001 |
| swap_das | 104/105 | 6.3 | 0.217 | 0.0342 | 1.001 | 0.999 | 0.15 -> -0.07 -> -0.07 (source -0.07) | 3891/4096 | 0.798 | 0.202 | 0.798 | 'True' | 1.291 | 0.000 |
| amp_das_a3 | 104/105 | 6.3 | 0.211 | 0.0333 | 1.001 | 0.999 | 0.15 -> -0.06 -> -0.06 (centre 0.22) | 3882/4096 | 0.798 | 0.202 | 0.798 | 'True' | 1.291 | 0.000 |

#### tune pair `tqa:270` -- baseline first tokens differ: True; p_true gap 0.860; informative for the swap test

**False statement** `tqa:270:F` (112 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: If you put a straw on a camel's back, the camel's back will break.

baseline: P(True) 0.007, P(False) 0.993, p_true 0.007, argmax 'False', top-5 [['False', 0.993], ['True', 0.0067], ['FALSE', 0.0001], ['This', 0.0], ['The', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 3.948

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 111/112 | 6.3 | 0 | 0 | n/a | n/a | 0.08 -> 0.08 -> 0.08 | 0/0 | 0.007 | 0.993 | 0.007 | 'False' | 3.948 | 0.000 |
| swap_meandiff | 111/112 | 6.3 | 0.257 | 0.0406 | 1.001 | 0.999 | -0.08 -> 0.18 -> 0.18 (source 0.18) | 3947/4094 | 0.006 | 0.994 | 0.006 | 'False' | 4.056 | 0.000 |
| amp_meandiff_a3 | 111/112 | 6.3 | 0.669 | 0.106 | 1.001 | 1.000 | -0.08 -> -0.75 -> -0.75 (centre 0.14) | 4036/4094 | 0.006 | 0.994 | 0.006 | 'False' | 4.056 | 0.000 |
| swap_das | 111/112 | 6.3 | 0.17 | 0.0269 | 1.003 | 0.998 | 0.08 -> 0.25 -> 0.25 (source 0.25) | 3852/4096 | 0.014 | 0.986 | 0.014 | 'False' | 3.306 | 0.002 |
| amp_das_a3 | 111/112 | 6.3 | 0.428 | 0.0677 | 0.999 | 1.000 | 0.08 -> -0.35 -> -0.35 (centre 0.22) | 3982/4096 | 0.001 | 0.999 | 0.001 | 'False' | 5.568 | 0.007 |

**True statement** `tqa:270:T` (109 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: Nothing in particular happens if you put a straw on a camel's back.

baseline: P(True) 0.867, P(False) 0.133, p_true 0.867, argmax 'True', top-5 [['True', 0.8664], ['False', 0.1329], ['Unknown', 0.0002], ['TRUE', 0.0001], ['Tr', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.965

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 108/109 | 6.3 | 0 | 0 | n/a | n/a | 0.25 -> 0.25 -> 0.25 | 0/0 | 0.867 | 0.133 | 0.867 | 'True' | 1.965 | 0.000 |
| swap_meandiff | 108/109 | 6.3 | 0.257 | 0.0406 | 1.002 | 0.999 | 0.18 -> -0.08 -> -0.08 (source -0.08) | 3949/4094 | 0.867 | 0.133 | 0.867 | 'True' | 1.965 | 0.000 |
| amp_meandiff_a3 | 108/109 | 6.3 | 0.101 | 0.016 | 1.000 | 0.994 | 0.18 -> 0.28 -> 0.28 (centre 0.14) | 3684/4094 | 0.880 | 0.119 | 0.881 | 'True' | 2.073 | 0.001 |
| swap_das | 108/109 | 6.3 | 0.17 | 0.0269 | 1.003 | 0.998 | 0.25 -> 0.08 -> 0.08 (source 0.08) | 3847/4096 | 0.880 | 0.119 | 0.881 | 'True' | 2.073 | 0.001 |
| amp_das_a3 | 108/109 | 6.3 | 0.0823 | 0.013 | 1.007 | 0.993 | 0.25 -> 0.33 -> 0.33 (centre 0.22) | 3596/4096 | 0.867 | 0.133 | 0.867 | 'True' | 1.965 | 0.000 |

### Statistics over all train pairs

96 pairs, 192 statements, 120 statements in informative pairs (baseline first tokens differ). Baseline J 0.479, AUROC 0.801, B 0.448 (TPR 0.688, FPR 0.208); mean KL(p_source \|\| p_base) 2.263.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.000 (0/120) | 0.370 | 2.245 | +0.008 | +0.003 / +0.003 | +0.469 | 0.798 | 0.443 | 0.995 (0.492) | 0.0399 |
| das | 0.008 (1/120) | 0.370 | 2.100 | +0.072 | +0.018 / +0.018 | +0.448 | 0.771 | 0.432 | 1.003 (0.713) | 0.0274 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.000 | +0.002 | +0.000 | +0.006 | -0.000 | 0 | 0 | 1.000 (0.779) | 0.0844 (0.319) |
| das a3 | +0.031 | +0.015 | +0.016 | +0.022 | -0.014 | 4 | 1 | 0.995 (0.321) | 0.0563 (0.194) |

### Statistics over all tune pairs

192 pairs, 384 statements, 248 statements in informative pairs (baseline first tokens differ). Baseline J 0.375, AUROC 0.743, B 0.443 (TPR 0.630, FPR 0.255); mean KL(p_source \|\| p_base) 2.060.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.000 (0/248) | 0.354 | 2.055 | +0.003 | +0.002 / +0.001 | +0.375 | 0.741 | 0.443 | 0.987 (0.202) | 0.0343 |
| das | 0.016 (4/248) | 0.357 | 1.941 | +0.058 | +0.020 / +0.021 | +0.339 | 0.709 | 0.445 | 0.987 (0.172) | 0.0238 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.000 | +0.004 | +0.000 | +0.007 | -0.002 | 1 | 1 | 1.000 (0.453) | 0.0771 (0.243) |
| das a3 | +0.068 | +0.019 | +0.008 | +0.013 | -0.022 | 16 | 3 | 0.996 (0.130) | 0.0517 (0.157) |

Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):

- baseline: sweep J +0.375 / AUROC 0.743 / B 0.443; here J +0.375 / AUROC 0.743 / B 0.443
- E_L11_a3 (das): sweep J +0.438 / AUROC 0.761 / B 0.448; here J +0.443 / AUROC 0.761 / B 0.451

### Reading (heuristic)

- das: swap fails on most informative statements in BOTH sets (train 1%, tune 2%) -> the trained direction does not carry the first-token swap even where it was fitted
- meandiff: swap fails on most informative statements in BOTH sets (train 0%, tune 0%) -> the trained direction does not carry the first-token swap even where it was fitted

## Layer 13

### Scale and direction facts (estimation set)

Mean |h| at the write position 7.7; median mean|coordinate| 0.071; the 8 largest coordinates carry 12.8% of |h|^2.

| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |
|---|---|---|---|---|
| 4055 | 2.0 | 0.00781 | -0.0889 | -0.0564 |
| 2088 | 0.9 | 0.00391 | -0.0105 | -0.0112 |
| 2932 | 0.9 | 0.00391 | +0.0142 | -0.0379 |
| 3881 | 0.8 | 0.00391 | +0.0305 | +0.0048 |
| 290 | 0.6 | 0.00391 | +0.0230 | -0.0197 |
| 788 | 0.6 | 0.00391 | +0.0176 | +0.0009 |
| 424 | 0.6 | 0.00391 | -0.0201 | -0.0330 |
| 2646 | 0.6 | 0.00391 | +0.0209 | -0.0255 |

| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |
|---|---|---|---|---|---|---|---|
| meandiff | 1.000 | - | 0.880 | 0.63 | 0.38 | 0.011 | -0.06 |
| das | 0.672 | 0.2534 | 0.812 | 0.43 | 0.33 | 0.007 | 0.06 |

Token-level readout on the estimation set (AUROC of the direction's projection summarised per sequence; mean |h| over all positions 13.1 vs last token 7.7):

| direction | last token | mean over positions | max over positions | min over positions |
|---|---|---|---|---|
| meandiff | 0.880 | 0.822 | 0.786 | 0.471 |
| das | 0.812 | 0.781 | 0.706 | 0.781 |

### Trace on the fixed pairs

#### train pair `tqa:765` -- baseline first tokens differ: True; p_true gap 0.760; informative for the swap test

**False statement** `tqa:765:F` (98 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein flunked math in high school.

baseline: P(True) 0.165, P(False) 0.835, p_true 0.165, argmax 'False', top-5 [['False', 0.835], ['True', 0.1644], ['TRUE', 0.0001], ['FALSE', 0.0001], ['false', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 1.413

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 97/98 | 7.6 | 0 | 0 | n/a | n/a | -0.14 -> -0.14 -> -0.14 | 0/0 | 0.165 | 0.835 | 0.165 | 'False' | 1.413 | 0.000 |
| swap_meandiff | 97/98 | 7.6 | 0.438 | 0.0573 | 1.000 | 1.000 | -0.40 -> 0.04 -> 0.04 (source 0.04) | 3966/4095 | 0.223 | 0.777 | 0.223 | 'False' | 1.138 | 0.010 |
| amp_meandiff_a3 | 97/98 | 7.6 | 1 | 0.131 | 1.000 | 1.000 | -0.40 -> -1.40 -> -1.40 (centre -0.06) | 4037/4095 | 0.085 | 0.915 | 0.085 | 'False' | 2.015 | 0.033 |
| swap_das | 97/98 | 7.6 | 0.374 | 0.0489 | 1.001 | 0.999 | -0.14 -> 0.24 -> 0.24 (source 0.24) | 3960/4096 | 0.500 | 0.500 | 0.500 | 'True' | 0.425 | 0.246 |
| amp_das_a3 | 97/98 | 7.6 | 0.579 | 0.0759 | 1.000 | 1.000 | -0.14 -> -0.72 -> -0.72 (centre 0.06) | 4010/4096 | 0.016 | 0.984 | 0.016 | 'False' | 3.559 | 0.248 |

**True statement** `tqa:765:T` (97 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein did very well in high school.

baseline: P(True) 0.924, P(False) 0.076, p_true 0.924, argmax 'True', top-5 [['True', 0.9237], ['False', 0.0758], ['Unknown', 0.0001], ['TRUE', 0.0001], ['true', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.720

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 96/97 | 7.6 | 0 | 0 | n/a | n/a | 0.24 -> 0.24 -> 0.24 | 0/0 | 0.924 | 0.076 | 0.924 | 'True' | 1.720 | 0.000 |
| swap_meandiff | 96/97 | 7.6 | 0.438 | 0.0574 | 1.000 | 1.000 | 0.04 -> -0.40 -> -0.40 (source -0.40) | 3964/4095 | 0.893 | 0.107 | 0.893 | 'True' | 1.441 | 0.006 |
| amp_meandiff_a3 | 96/97 | 7.6 | 0.313 | 0.0411 | 1.001 | 0.999 | 0.04 -> 0.35 -> 0.36 (centre -0.06) | 3914/4095 | 0.940 | 0.060 | 0.940 | 'True' | 1.912 | 0.002 |
| swap_das | 96/97 | 7.6 | 0.374 | 0.049 | 1.001 | 0.999 | 0.24 -> -0.14 -> -0.14 (source -0.14) | 3953/4096 | 0.754 | 0.245 | 0.755 | 'True' | 0.774 | 0.098 |
| amp_das_a3 | 96/97 | 7.6 | 0.541 | 0.071 | 1.000 | 1.000 | 0.24 -> 0.78 -> 0.78 (centre 0.06) | 4004/4096 | 0.982 | 0.018 | 0.982 | 'True' | 2.912 | 0.053 |

#### tune pair `tqa:685` -- baseline first tokens differ: True; p_true gap 0.738; informative for the swap test

**False statement** `tqa:685:F` (100 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: Adolf Hitler started the construction of the autobahn.

baseline: P(True) 0.060, P(False) 0.940, p_true 0.060, argmax 'False', top-5 [['False', 0.9396], ['True', 0.0601], ['false', 0.0], ['FALSE', 0.0], ['Incorrect', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 1.754

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 99/100 | 7.7 | 0 | 0 | n/a | n/a | -0.27 -> -0.27 -> -0.27 | 0/0 | 0.060 | 0.940 | 0.060 | 'False' | 1.754 | 0.000 |
| swap_meandiff | 99/100 | 7.7 | 0.994 | 0.129 | 1.000 | 1.000 | -0.76 -> 0.23 -> 0.23 (source 0.23) | 4045/4095 | 0.119 | 0.881 | 0.119 | 'False' | 1.220 | 0.020 |
| amp_meandiff_a3 | 99/100 | 7.7 | 2.1 | 0.273 | 1.000 | 1.000 | -0.76 -> -2.86 -> -2.86 (centre -0.06) | 4073/4095 | 0.014 | 0.986 | 0.014 | 'False' | 2.903 | 0.042 |
| swap_das | 99/100 | 7.7 | 0.636 | 0.0827 | 1.000 | 1.000 | -0.27 -> 0.37 -> 0.36 (source 0.37) | 4021/4096 | 0.592 | 0.407 | 0.593 | 'True' | 0.096 | 0.648 |
| amp_das_a3 | 99/100 | 7.7 | 0.98 | 0.127 | 1.000 | 1.000 | -0.27 -> -1.25 -> -1.25 (centre 0.06) | 4047/4096 | 0.002 | 0.998 | 0.002 | 'False' | 4.387 | 0.143 |

**True statement** `tqa:685:T` (105 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: The construction of the autobahn began in the mid-1920s.

baseline: P(True) 0.798, P(False) 0.202, p_true 0.798, argmax 'True', top-5 [['True', 0.7978], ['False', 0.2017], ['TRUE', 0.0001], ['Part', 0.0], ['Incorrect', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.290

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 104/105 | 7.6 | 0 | 0 | n/a | n/a | 0.37 -> 0.37 -> 0.37 | 0/0 | 0.798 | 0.202 | 0.798 | 'True' | 1.290 | 0.000 |
| swap_meandiff | 104/105 | 7.6 | 0.994 | 0.13 | 1.000 | 1.000 | 0.23 -> -0.76 -> -0.76 (source -0.76) | 4041/4095 | 0.651 | 0.348 | 0.651 | 'True' | 0.789 | 0.052 |
| amp_meandiff_a3 | 104/105 | 7.6 | 0.881 | 0.115 | 1.000 | 1.000 | 0.23 -> 1.11 -> 1.11 (centre -0.06) | 4033/4095 | 0.867 | 0.133 | 0.867 | 'True' | 1.678 | 0.018 |
| swap_das | 104/105 | 7.6 | 0.636 | 0.0833 | 1.000 | 1.000 | 0.37 -> -0.27 -> -0.27 (source -0.27) | 4011/4096 | 0.294 | 0.705 | 0.294 | 'False' | 0.174 | 0.544 |
| amp_das_a3 | 104/105 | 7.6 | 0.928 | 0.122 | 1.000 | 1.000 | 0.37 -> 1.29 -> 1.29 (centre 0.06) | 4043/4096 | 0.971 | 0.029 | 0.971 | 'True' | 3.092 | 0.234 |

#### tune pair `tqa:270` -- baseline first tokens differ: True; p_true gap 0.860; informative for the swap test

**False statement** `tqa:270:F` (112 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: If you put a straw on a camel's back, the camel's back will break.

baseline: P(True) 0.007, P(False) 0.993, p_true 0.007, argmax 'False', top-5 [['False', 0.993], ['True', 0.0067], ['FALSE', 0.0001], ['This', 0.0], ['The', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 3.948

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 111/112 | 7.6 | 0 | 0 | n/a | n/a | -0.55 -> -0.55 -> -0.55 | 0/0 | 0.007 | 0.993 | 0.007 | 'False' | 3.948 | 0.000 |
| swap_meandiff | 111/112 | 7.6 | 0.83 | 0.109 | 1.000 | 1.000 | -0.69 -> 0.14 -> 0.14 (source 0.14) | 4027/4095 | 0.010 | 0.990 | 0.010 | 'False' | 3.626 | 0.001 |
| amp_meandiff_a3 | 111/112 | 7.6 | 1.89 | 0.25 | 1.000 | 1.000 | -0.69 -> -2.59 -> -2.59 (centre -0.06) | 4065/4095 | 0.002 | 0.997 | 0.002 | 'False' | 4.811 | 0.002 |
| swap_das | 111/112 | 7.6 | 0.647 | 0.0853 | 1.000 | 1.000 | -0.55 -> 0.10 -> 0.10 (source 0.10) | 4012/4096 | 0.119 | 0.880 | 0.119 | 'False' | 1.468 | 0.100 |
| amp_das_a3 | 111/112 | 7.6 | 1.81 | 0.239 | 1.000 | 1.000 | -0.55 -> -2.36 -> -2.36 (centre 0.06) | 4070/4096 | 0.000 | 0.999 | 0.000 | 'False' | 6.542 | 0.014 |

**True statement** `tqa:270:T` (109 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: Nothing in particular happens if you put a straw on a camel's back.

baseline: P(True) 0.867, P(False) 0.133, p_true 0.867, argmax 'True', top-5 [['True', 0.8664], ['False', 0.1329], ['Unknown', 0.0002], ['TRUE', 0.0001], ['Tr', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.965

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 108/109 | 7.6 | 0 | 0 | n/a | n/a | 0.10 -> 0.10 -> 0.10 | 0/0 | 0.867 | 0.133 | 0.867 | 'True' | 1.965 | 0.000 |
| swap_meandiff | 108/109 | 7.6 | 0.83 | 0.109 | 1.000 | 1.000 | 0.14 -> -0.69 -> -0.69 (source -0.69) | 4026/4095 | 0.798 | 0.202 | 0.798 | 'True' | 1.551 | 0.016 |
| amp_meandiff_a3 | 108/109 | 7.6 | 0.596 | 0.0785 | 1.001 | 1.000 | 0.14 -> 0.73 -> 0.73 (centre -0.06) | 4000/4095 | 0.904 | 0.095 | 0.905 | 'True' | 2.295 | 0.007 |
| swap_das | 108/109 | 7.6 | 0.647 | 0.0851 | 1.000 | 1.000 | 0.10 -> -0.55 -> -0.55 (source -0.55) | 4016/4096 | 0.562 | 0.437 | 0.562 | 'True' | 0.785 | 0.217 |
| amp_das_a3 | 108/109 | 7.6 | 0.13 | 0.0171 | 1.005 | 0.996 | 0.10 -> 0.23 -> 0.23 (centre 0.06) | 3697/4096 | 0.904 | 0.095 | 0.905 | 'True' | 2.295 | 0.007 |

### Statistics over all train pairs

96 pairs, 192 statements, 120 statements in informative pairs (baseline first tokens differ). Baseline J 0.479, AUROC 0.801, B 0.448 (TPR 0.688, FPR 0.208); mean KL(p_source \|\| p_base) 2.263.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.058 (7/120) | 0.401 | 1.887 | +0.166 | +0.041 / +0.053 | +0.406 | 0.758 | 0.411 | 1.003 (0.999) | 0.0902 |
| das | 0.425 (51/120) | 0.641 | 0.742 | +0.672 | +0.181 / +0.275 | -0.021 | 0.531 | 0.417 | 1.005 (0.987) | 0.0689 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.010 | +0.022 | +0.036 | +0.052 | +0.002 | 4 | 3 | 1.001 (0.999) | 0.156 (0.456) |
| das a3 | +0.021 | +0.012 | +0.073 | +0.102 | +0.048 | 10 | 8 | 1.002 (0.999) | 0.124 (0.367) |

### Statistics over all tune pairs

192 pairs, 384 statements, 248 statements in informative pairs (baseline first tokens differ). Baseline J 0.375, AUROC 0.743, B 0.443 (TPR 0.630, FPR 0.255); mean KL(p_source \|\| p_base) 2.060.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.077 (19/248) | 0.401 | 1.746 | +0.153 | +0.041 / +0.049 | +0.281 | 0.697 | 0.427 | 0.994 (0.072) | 0.081 |
| das | 0.335 (83/248) | 0.557 | 0.760 | +0.631 | +0.157 / +0.213 | -0.010 | 0.507 | 0.448 | 0.995 (0.470) | 0.0603 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.047 | +0.028 | +0.023 | +0.054 | -0.008 | 12 | 3 | 0.998 (0.407) | 0.139 (0.549) |
| das a3 | +0.036 | +0.022 | +0.060 | +0.095 | +0.029 | 21 | 14 | 0.999 (0.096) | 0.112 (0.42) |

Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):

- baseline: sweep J +0.375 / AUROC 0.743 / B 0.443; here J +0.375 / AUROC 0.743 / B 0.443
- E_L13_a3 (das): sweep J +0.411 / AUROC 0.763 / B 0.508; here J +0.411 / AUROC 0.764 / B 0.503

### Reading (heuristic)

- das: swap fails on most informative statements in BOTH sets (train 42%, tune 33%) -> the trained direction does not carry the first-token swap even where it was fitted
- meandiff: swap fails on most informative statements in BOTH sets (train 6%, tune 8%) -> the trained direction does not carry the first-token swap even where it was fitted

## Layer 16

### Scale and direction facts (estimation set)

Mean |h| at the write position 10.3; median mean|coordinate| 0.089; the 8 largest coordinates carry 17.1% of |h|^2.

| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |
|---|---|---|---|---|
| 4055 | 3.2 | 0.0156 | +0.1478 | +0.0122 |
| 2088 | 1.5 | 0.00781 | +0.0070 | +0.0080 |
| 3881 | 1.3 | 0.00781 | +0.0316 | +0.0340 |
| 788 | 1.0 | 0.00391 | -0.0258 | -0.0057 |
| 2932 | 0.9 | 0.00391 | +0.0118 | +0.0371 |
| 290 | 0.8 | 0.00391 | +0.0345 | -0.0249 |
| 2646 | 0.8 | 0.00391 | +0.0018 | -0.0135 |
| 424 | 0.8 | 0.00391 | -0.0405 | -0.0358 |

| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |
|---|---|---|---|---|---|---|---|
| meandiff | 1.000 | - | 0.836 | 1.72 | 1.21 | 0.027 | 0.09 |
| das | 0.694 | 0.0214 | 0.811 | 1.19 | 0.95 | 0.005 | -0.15 |

Token-level readout on the estimation set (AUROC of the direction's projection summarised per sequence; mean |h| over all positions 15.1 vs last token 10.3):

| direction | last token | mean over positions | max over positions | min over positions |
|---|---|---|---|---|
| meandiff | 0.836 | 0.809 | 0.764 | 0.751 |
| das | 0.811 | 0.792 | 0.529 | 0.716 |

### Trace on the fixed pairs

#### train pair `tqa:765` -- baseline first tokens differ: True; p_true gap 0.760; informative for the swap test

**False statement** `tqa:765:F` (98 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein flunked math in high school.

baseline: P(True) 0.165, P(False) 0.835, p_true 0.165, argmax 'False', top-5 [['False', 0.835], ['True', 0.1644], ['TRUE', 0.0001], ['FALSE', 0.0001], ['false', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 1.413

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 97/98 | 10.2 | 0 | 0 | n/a | n/a | -0.29 -> -0.29 -> -0.29 | 0/0 | 0.165 | 0.835 | 0.165 | 'False' | 1.413 | 0.000 |
| swap_meandiff | 97/98 | 10.2 | 1.11 | 0.109 | 0.999 | 1.000 | -0.01 -> 1.10 -> 1.10 (source 1.10) | 4033/4096 | 0.562 | 0.438 | 0.562 | 'True' | 0.326 | 0.338 |
| amp_meandiff_a3 | 97/98 | 10.2 | 0.302 | 0.0296 | 1.002 | 0.999 | -0.01 -> -0.31 -> -0.31 (centre 0.09) | 3874/4096 | 0.095 | 0.904 | 0.095 | 'False' | 1.911 | 0.023 |
| swap_das | 97/98 | 10.2 | 1.08 | 0.106 | 1.000 | 1.000 | -0.29 -> 0.78 -> 0.78 (source 0.78) | 4041/4096 | 0.852 | 0.148 | 0.852 | 'True' | 0.025 | 1.175 |
| amp_das_a3 | 97/98 | 10.2 | 0.44 | 0.0431 | 1.000 | 0.999 | -0.29 -> -0.73 -> -0.73 (centre -0.15) | 3951/4096 | 0.042 | 0.958 | 0.042 | 'False' | 2.662 | 0.110 |

**True statement** `tqa:765:T` (97 tokens; write position token '\n\n'): The answer to the question "How did Einstein do in high school?" is: Einstein did very well in high school.

baseline: P(True) 0.924, P(False) 0.076, p_true 0.924, argmax 'True', top-5 [['True', 0.9237], ['False', 0.0758], ['Unknown', 0.0001], ['TRUE', 0.0001], ['true', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.720

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 96/97 | 10.2 | 0 | 0 | n/a | n/a | 0.78 -> 0.78 -> 0.78 | 0/0 | 0.924 | 0.076 | 0.924 | 'True' | 1.720 | 0.000 |
| swap_meandiff | 96/97 | 10.2 | 1.11 | 0.109 | 0.999 | 1.000 | 1.10 -> -0.01 -> -0.01 (source -0.01) | 4032/4096 | 0.592 | 0.407 | 0.593 | 'True' | 0.390 | 0.283 |
| amp_meandiff_a3 | 96/97 | 10.2 | 3.02 | 0.297 | 1.000 | 1.000 | 1.10 -> 4.12 -> 4.12 (centre 0.09) | 4072/4096 | 0.997 | 0.003 | 0.997 | 'True' | 4.359 | 0.171 |
| swap_das | 96/97 | 10.2 | 1.08 | 0.106 | 1.000 | 1.000 | 0.78 -> -0.29 -> -0.29 (source -0.29) | 4041/4096 | 0.223 | 0.777 | 0.223 | 'False' | 0.011 | 1.138 |
| amp_das_a3 | 96/97 | 10.2 | 2.8 | 0.274 | 1.000 | 1.000 | 0.78 -> 3.58 -> 3.58 (centre -0.15) | 4078/4096 | 1.000 | 0.000 | 1.000 | 'True' | 6.444 | 0.358 |

#### tune pair `tqa:685` -- baseline first tokens differ: True; p_true gap 0.738; informative for the swap test

**False statement** `tqa:685:F` (100 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: Adolf Hitler started the construction of the autobahn.

baseline: P(True) 0.060, P(False) 0.940, p_true 0.060, argmax 'False', top-5 [['False', 0.9396], ['True', 0.0601], ['false', 0.0], ['FALSE', 0.0], ['Incorrect', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 1.754

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 99/100 | 10.1 | 0 | 0 | n/a | n/a | -0.59 -> -0.59 -> -0.59 | 0/0 | 0.060 | 0.940 | 0.060 | 'False' | 1.754 | 0.000 |
| swap_meandiff | 99/100 | 10.1 | 1.56 | 0.155 | 1.000 | 1.000 | -0.66 -> 0.90 -> 0.90 (source 0.90) | 4051/4096 | 0.531 | 0.469 | 0.531 | 'True' | 0.155 | 0.523 |
| amp_meandiff_a3 | 99/100 | 10.1 | 2.25 | 0.223 | 1.000 | 1.000 | -0.66 -> -2.91 -> -2.91 (centre 0.09) | 4064/4096 | 0.005 | 0.995 | 0.005 | 'False' | 3.692 | 0.094 |
| swap_das | 99/100 | 10.1 | 1.37 | 0.136 | 1.000 | 1.000 | -0.59 -> 0.78 -> 0.78 (source 0.78) | 4058/4096 | 0.893 | 0.107 | 0.893 | 'True' | 0.039 | 1.883 |
| amp_das_a3 | 99/100 | 10.1 | 1.32 | 0.131 | 1.000 | 1.000 | -0.59 -> -1.91 -> -1.91 (centre -0.15) | 4056/4096 | 0.003 | 0.997 | 0.003 | 'False' | 4.188 | 0.129 |

**True statement** `tqa:685:T` (105 tokens; write position token '\n\n'): The answer to the question "Who started the construction of the autobahn?" is: The construction of the autobahn began in the mid-1920s.

baseline: P(True) 0.798, P(False) 0.202, p_true 0.798, argmax 'True', top-5 [['True', 0.7978], ['False', 0.2017], ['TRUE', 0.0001], ['Part', 0.0], ['Incorrect', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.290

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 104/105 | 10.2 | 0 | 0 | n/a | n/a | 0.78 -> 0.78 -> 0.78 | 0/0 | 0.798 | 0.202 | 0.798 | 'True' | 1.290 | 0.000 |
| swap_meandiff | 104/105 | 10.2 | 1.56 | 0.153 | 1.000 | 1.000 | 0.90 -> -0.66 -> -0.66 (source -0.66) | 4053/4096 | 0.182 | 0.817 | 0.182 | 'False' | 0.064 | 0.895 |
| amp_meandiff_a3 | 104/105 | 10.2 | 2.43 | 0.239 | 1.000 | 1.000 | 0.90 -> 3.33 -> 3.33 (centre 0.09) | 4069/4096 | 0.986 | 0.014 | 0.986 | 'True' | 3.781 | 0.369 |
| swap_das | 104/105 | 10.2 | 1.37 | 0.134 | 1.000 | 1.000 | 0.78 -> -0.59 -> -0.59 (source -0.59) | 4063/4096 | 0.053 | 0.946 | 0.053 | 'False' | 0.000 | 1.846 |
| amp_das_a3 | 104/105 | 10.2 | 2.78 | 0.273 | 1.000 | 1.000 | 0.78 -> 3.56 -> 3.56 (centre -0.15) | 4078/4096 | 0.999 | 0.001 | 0.999 | 'True' | 6.352 | 0.911 |

#### tune pair `tqa:270` -- baseline first tokens differ: True; p_true gap 0.860; informative for the swap test

**False statement** `tqa:270:F` (112 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: If you put a straw on a camel's back, the camel's back will break.

baseline: P(True) 0.007, P(False) 0.993, p_true 0.007, argmax 'False', top-5 [['False', 0.993], ['True', 0.0067], ['FALSE', 0.0001], ['This', 0.0], ['The', 0.0]]; greedy continuation 'False.<|eot_id|>'; KL(p_source \|\| p_base) 3.948

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 111/112 | 10.4 | 0 | 0 | n/a | n/a | -1.43 -> -1.43 -> -1.43 | 0/0 | 0.007 | 0.993 | 0.007 | 'False' | 3.948 | 0.000 |
| swap_meandiff | 111/112 | 10.4 | 3.07 | 0.296 | 1.000 | 1.000 | -1.56 -> 1.51 -> 1.51 (source 1.51) | 4075/4096 | 0.622 | 0.377 | 0.622 | 'True' | 0.149 | 0.931 |
| amp_meandiff_a3 | 111/112 | 10.4 | 4.95 | 0.478 | 1.000 | 1.000 | -1.56 -> -6.51 -> -6.51 (centre 0.09) | 4083/4096 | 0.001 | 0.999 | 0.001 | 'False' | 5.893 | 0.009 |
| swap_das | 111/112 | 10.4 | 2.19 | 0.212 | 1.000 | 1.000 | -1.43 -> 0.77 -> 0.77 (source 0.77) | 4071/4096 | 0.904 | 0.095 | 0.905 | 'True' | 0.007 | 2.295 |
| amp_das_a3 | 111/112 | 10.4 | 3.84 | 0.37 | 1.000 | 1.000 | -1.43 -> -5.27 -> -5.27 (centre -0.15) | 4087/4096 | 0.000 | 1.000 | 0.000 | 'False' | 7.300 | 0.019 |

**True statement** `tqa:270:T` (109 tokens; write position token '\n\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: Nothing in particular happens if you put a straw on a camel's back.

baseline: P(True) 0.867, P(False) 0.133, p_true 0.867, argmax 'True', top-5 [['True', 0.8664], ['False', 0.1329], ['Unknown', 0.0002], ['TRUE', 0.0001], ['Tr', 0.0]]; greedy continuation 'True.<|eot_id|>'; KL(p_source \|\| p_base) 1.965

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 108/109 | 10.2 | 0 | 0 | n/a | n/a | 0.77 -> 0.77 -> 0.77 | 0/0 | 0.867 | 0.133 | 0.867 | 'True' | 1.965 | 0.000 |
| swap_meandiff | 108/109 | 10.2 | 3.07 | 0.3 | 1.000 | 1.000 | 1.51 -> -1.56 -> -1.56 (source -1.56) | 4075/4096 | 0.060 | 0.939 | 0.060 | 'False' | 0.041 | 2.054 |
| amp_meandiff_a3 | 108/109 | 10.2 | 4.26 | 0.417 | 1.000 | 1.000 | 1.51 -> 5.77 -> 5.77 (centre 0.09) | 4079/4096 | 0.995 | 0.005 | 0.995 | 'True' | 5.303 | 0.328 |
| swap_das | 108/109 | 10.2 | 2.19 | 0.214 | 1.000 | 1.000 | 0.77 -> -1.43 -> -1.43 (source -1.43) | 4073/4096 | 0.020 | 0.979 | 0.020 | 'False' | 0.006 | 2.987 |
| amp_das_a3 | 108/109 | 10.2 | 2.74 | 0.268 | 1.000 | 1.000 | 0.77 -> 3.51 -> 3.51 (centre -0.15) | 4079/4096 | 0.999 | 0.001 | 0.999 | 'True' | 6.541 | 0.491 |

### Statistics over all train pairs

96 pairs, 192 statements, 120 statements in informative pairs (baseline first tokens differ). Baseline J 0.479, AUROC 0.801, B 0.448 (TPR 0.688, FPR 0.208); mean KL(p_source \|\| p_base) 2.263.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.800 (96/120) | 0.865 | 0.187 | +0.918 | +0.347 / +0.534 | -0.375 | 0.262 | 0.406 | 0.999 (0.766) | 0.204 |
| das | 1.000 (120/120) | 0.984 | 0.014 | +0.994 | +0.411 / +0.642 | -0.490 | 0.194 | 0.453 | 0.999 (0.735) | 0.152 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.010 | +0.025 | +0.057 | +0.096 | +0.016 | 6 | 5 | 1.000 (0.998) | 0.379 (0.858) |
| das a3 | +0.000 | +0.010 | +0.052 | +0.096 | +0.030 | 6 | 6 | 1.001 (0.970) | 0.283 (0.636) |

### Statistics over all tune pairs

192 pairs, 384 statements, 248 statements in informative pairs (baseline first tokens differ). Baseline J 0.375, AUROC 0.743, B 0.443 (TPR 0.630, FPR 0.255); mean KL(p_source \|\| p_base) 2.060.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.750 (186/248) | 0.833 | 0.201 | +0.902 | +0.274 / +0.379 | -0.271 | 0.306 | 0.370 | 0.997 (0.248) | 0.183 |
| das | 0.972 (241/248) | 0.977 | 0.024 | +0.988 | +0.317 / +0.448 | -0.328 | 0.271 | 0.440 | 0.999 (0.636) | 0.138 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.057 | +0.022 | +0.065 | +0.109 | +0.023 | 20 | 9 | 0.999 (0.552) | 0.337 (0.79) |
| das a3 | +0.010 | +0.002 | +0.073 | +0.099 | +0.046 | 15 | 13 | 0.999 (0.681) | 0.255 (0.602) |

Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):

- baseline: sweep J +0.375 / AUROC 0.743 / B 0.443; here J +0.375 / AUROC 0.743 / B 0.443
- E_L16_a3 (das): sweep J +0.396 / AUROC 0.744 / B 0.510; here J +0.385 / AUROC 0.745 / B 0.516

### Reading (heuristic)

- no automatic reading triggered; see the tables


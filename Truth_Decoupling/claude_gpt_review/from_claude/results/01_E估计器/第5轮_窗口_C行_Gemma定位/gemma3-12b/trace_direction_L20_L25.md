# Direction trace: write position -> first token, gemma3-12b

Layers [20, 25]; DAS files `/workspace/Truth_Decoupling/results_das_tf/E1/gemma3-12b` (variant tf); amplify alphas [3.0]; estimation set 192 statements; traced training pairs ['tqa:765'], traced tuning pairs ['tqa:685', 'tqa:270'] (fixed order). Identity = zero edit through the same hook (must be bit-identical). Swap = write the paired statement's coordinate along w (alpha 1). Amplify = the sweep's operator with the sweep's centre.

Environment: torch 2.13.0a0+8145d630e8.nv26.06, transformers 4.57.1, CUDA 13.3, model dtype torch.bfloat16, ['Gemma3ForConditionalGeneration'], hidden 3840, 48 layers, sliding window 1024, layer types {'full_attention': 8, 'sliding_attention': 40}, attention sdpa.
Prompt lengths (forced True/False prompt): estimation 68-117 tokens (mean 80.6), test 65-121 (mean 81.1); 0 test prompts exceed the sliding window.
Label tokens: {'True': [' true', 'true', 'True', ' True', 'TRUE'], 'False': [' false', 'false', ' False', 'False', 'FALSE']}

## Layer 20

### Scale and direction facts (estimation set)

Mean |h| at the write position 46193.3; median mean|coordinate| 17.862; the 8 largest coordinates carry 99.8% of |h|^2.

| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |
|---|---|---|---|---|
| 2339 | 46138.7 | 256 | -0.7093 | -0.1797 |
| 601 | 558.1 | 4 | +0.0906 | +0.0714 |
| 3499 | 503.9 | 2 | +0.0202 | +0.0253 |
| 3071 | 445.4 | 2 | +0.0155 | -0.0271 |
| 107 | 348.3 | 2 | +0.0086 | +0.0632 |
| 1409 | 295.5 | 2 | +0.0006 | -0.0253 |
| 948 | 232.4 | 1 | +0.0033 | +0.0143 |
| 322 | 212.1 | 1 | -0.0017 | +0.0126 |

| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |
|---|---|---|---|---|---|---|---|
| meandiff | 1.000 | - | 0.682 | 150.39 | 231.91 | 0.512 | -32693.80 |
| das | 0.599 | 4.6992 | 0.804 | 90.02 | 73.96 | 0.044 | -8310.46 |

### Trace on the fixed pairs

#### train pair `tqa:765` -- baseline first tokens differ: False; p_true gap 0.000; NOT informative: the two statements already give the same first-token distribution, the swap cannot be tested here

**False statement** `tqa:765:F` (74 tokens; write position token '\n'): The answer to the question "How did Einstein do in high school?" is: Einstein flunked math in high school.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['FALSE', 0.0]]; greedy continuation 'False<end_of_turn>'; KL(p_source \|\| p_base) 0.000

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 73/74 | 46390.1 | 0 | 0 | n/a | n/a | -8369.69 -> -8369.69 -> -8369.69 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| swap_meandiff | 73/74 | 46390.1 | 127 | 0.00273 | 0.706 | 0.704 | -32863.92 -> -32990.74 -> -32927.00 (source -32990.74) | 3669/3839 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| amp_meandiff_a3 | 73/74 | 46390.1 | 510 | 0.011 | 0.866 | 0.985 | -32863.92 -> -33374.27 -> -33299.29 (centre -32693.80) | 3795/3839 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| swap_das | 73/74 | 46390.1 | 3.74 | 8.07e-05 | 0.978 | 0.807 | -8369.69 -> -8365.94 -> -8366.73 (source -8365.94) | 1591/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| amp_das_a3 | 73/74 | 46390.1 | 178 | 0.00383 | 0.984 | 0.983 | -8369.69 -> -8547.36 -> -8541.58 (centre -8310.46) | 3767/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |

**True statement** `tqa:765:T` (73 tokens; write position token '\n'): The answer to the question "How did Einstein do in high school?" is: Einstein did very well in high school.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], [' False', 0.0], ['false', 0.0], ['**', 0.0]]; greedy continuation 'False<end_of_turn>'; KL(p_source \|\| p_base) 0.000

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 72/73 | 46646.9 | 0 | 0 | n/a | n/a | -8365.94 -> -8365.94 -> -8365.94 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| swap_meandiff | 72/73 | 46646.9 | 127 | 0.00272 | 0.706 | 0.704 | -32990.74 -> -32863.91 -> -32927.71 (source -32863.92) | 3670/3839 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| amp_meandiff_a3 | 72/73 | 46646.9 | 891 | 0.0191 | 0.910 | 0.995 | -32990.74 -> -33881.57 -> -33796.54 (centre -32693.80) | 3811/3839 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| swap_das | 72/73 | 46646.9 | 3.74 | 8.03e-05 | 0.977 | 0.805 | -8365.94 -> -8369.69 -> -8368.89 (source -8369.69) | 1576/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| amp_das_a3 | 72/73 | 46646.9 | 166 | 0.00357 | 0.984 | 0.983 | -8365.94 -> -8532.38 -> -8526.98 (centre -8310.46) | 3762/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |

#### tune pair `tqa:685` -- baseline first tokens differ: True; p_true gap 1.000; informative for the swap test

**False statement** `tqa:685:F` (76 tokens; write position token '\n'): The answer to the question "Who started the construction of the autobahn?" is: Adolf Hitler started the construction of the autobahn.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['Mostly', 0.0]]; greedy continuation 'False<end_of_turn>'; KL(p_source \|\| p_base) 10.499

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 75/76 | 46391.2 | 0 | 0 | n/a | n/a | -8383.60 -> -8383.60 -> -8383.60 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 10.499 | 0.000 |
| swap_meandiff | 75/76 | 46391.2 | 52.2 | 0.00112 | 0.705 | 0.702 | -32882.05 -> -32829.89 -> -32856.26 (source -32829.89) | 3422/3839 | 0.000 | 1.000 | 0.000 | 'False' | 10.499 | 0.000 |
| amp_meandiff_a3 | 75/76 | 46391.2 | 565 | 0.0122 | 1.148 | 0.993 | -32882.05 -> -33446.80 -> -33525.86 (centre -32693.80) | 3800/3839 | 0.000 | 1.000 | 0.000 | 'False' | 11.249 | 0.000 |
| swap_das | 75/76 | 46391.2 | 45.7 | 0.000984 | 0.989 | 0.980 | -8383.60 -> -8337.95 -> -8339.33 (source -8337.95) | 3545/3840 | 0.000 | 1.000 | 0.000 | 'False' | 9.499 | 0.000 |
| amp_das_a3 | 75/76 | 46391.2 | 219 | 0.00473 | 0.984 | 0.984 | -8383.60 -> -8603.03 -> -8596.02 (centre -8310.46) | 3787/3840 | 0.000 | 1.000 | 0.000 | 'False' | 14.499 | 0.000 |

**True statement** `tqa:685:T` (83 tokens; write position token '\n'): The answer to the question "Who started the construction of the autobahn?" is: The construction of the autobahn began in the mid-1920s.

baseline: P(True) 1.000, P(False) 0.000, p_true 1.000, argmax 'True', top-5 [['True', 1.0], ['False', 0.0], [' True', 0.0], ['true', 0.0], ['TRUE', 0.0]]; greedy continuation 'True<end_of_turn>'; KL(p_source \|\| p_base) 9.999

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 82/83 | 46391.4 | 0 | 0 | n/a | n/a | -8337.95 -> -8337.95 -> -8337.95 | 0/0 | 1.000 | 0.000 | 1.000 | 'True' | 9.999 | 0.000 |
| swap_meandiff | 82/83 | 46391.4 | 52.2 | 0.00112 | 0.704 | 0.701 | -32829.89 -> -32882.05 -> -32855.66 (source -32882.05) | 3417/3839 | 1.000 | 0.000 | 1.000 | 'True' | 9.999 | 0.000 |
| amp_meandiff_a3 | 82/83 | 46391.4 | 408 | 0.0088 | 0.943 | 0.998 | -32829.89 -> -33238.16 -> -33214.28 (centre -32693.80) | 3787/3839 | 1.000 | 0.000 | 1.000 | 'True' | 10.499 | 0.000 |
| swap_das | 82/83 | 46391.4 | 45.7 | 0.000984 | 0.990 | 0.980 | -8337.95 -> -8383.60 -> -8382.23 (source -8383.60) | 3547/3840 | 1.000 | 0.000 | 1.000 | 'True' | 9.999 | 0.000 |
| amp_das_a3 | 82/83 | 46391.4 | 82.5 | 0.00178 | 0.984 | 0.983 | -8337.95 -> -8420.42 -> -8417.67 (centre -8310.46) | 3678/3840 | 1.000 | 0.000 | 1.000 | 'True' | 9.499 | 0.000 |

#### tune pair `tqa:270` -- baseline first tokens differ: True; p_true gap 1.000; informative for the swap test

**False statement** `tqa:270:F` (91 tokens; write position token '\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: If you put a straw on a camel's back, the camel's back will break.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['FALSE', 0.0]]; greedy continuation 'False<end_of_turn>'; KL(p_source \|\| p_base) 12.500

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 90/91 | 46390.5 | 0 | 0 | n/a | n/a | -8359.36 -> -8359.36 -> -8359.36 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 12.500 | 0.000 |
| swap_meandiff | 90/91 | 46390.5 | 58 | 0.00125 | 0.704 | 0.702 | -32856.64 -> -32798.67 -> -32827.99 (source -32798.67) | 3451/3839 | 0.000 | 1.000 | 0.000 | 'False' | 12.000 | 0.000 |
| amp_meandiff_a3 | 90/91 | 46390.5 | 489 | 0.0105 | 0.879 | 0.989 | -32856.64 -> -33345.19 -> -33281.09 (centre -32693.80) | 3794/3839 | 0.000 | 1.000 | 0.000 | 'False' | 13.250 | 0.000 |
| swap_das | 90/91 | 46390.5 | 49.7 | 0.00107 | 0.991 | 0.981 | -8359.36 -> -8309.64 -> -8311.05 (source -8309.64) | 3577/3840 | 0.000 | 1.000 | 0.000 | 'False' | 11.500 | 0.000 |
| amp_das_a3 | 90/91 | 46390.5 | 147 | 0.00316 | 0.985 | 0.983 | -8359.36 -> -8506.06 -> -8501.42 (centre -8310.46) | 3749/3840 | 0.000 | 1.000 | 0.000 | 'False' | 15.500 | 0.000 |

**True statement** `tqa:270:T` (87 tokens; write position token '\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: Nothing in particular happens if you put a straw on a camel's back.

baseline: P(True) 1.000, P(False) 0.000, p_true 1.000, argmax 'True', top-5 [['True', 1.0], ['False', 0.0], [' True', 0.0], ['true', 0.0], ['TRUE', 0.0]]; greedy continuation 'True<end_of_turn>'; KL(p_source \|\| p_base) 16.250

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 86/87 | 46390.8 | 0 | 0 | n/a | n/a | -8309.65 -> -8309.65 -> -8309.65 | 0/0 | 1.000 | 0.000 | 1.000 | 'True' | 16.250 | 0.000 |
| swap_meandiff | 86/87 | 46390.8 | 58 | 0.00125 | 0.704 | 0.702 | -32798.67 -> -32856.65 -> -32827.32 (source -32856.65) | 3457/3839 | 1.000 | 0.000 | 1.000 | 'True' | 16.250 | 0.000 |
| amp_meandiff_a3 | 86/87 | 46390.8 | 315 | 0.00678 | 1.077 | 0.998 | -32798.67 -> -33113.29 -> -33136.55 (centre -32693.80) | 3776/3839 | 1.000 | 0.000 | 1.000 | 'True' | 16.000 | 0.000 |
| swap_das | 86/87 | 46390.8 | 49.7 | 0.00107 | 0.990 | 0.981 | -8309.65 -> -8359.36 -> -8357.93 (source -8359.36) | 3581/3840 | 1.000 | 0.000 | 1.000 | 'True' | 16.000 | 0.000 |
| amp_das_a3 | 86/87 | 46390.8 | 2.45 | 5.29e-05 | 0.887 | 0.720 | -8309.65 -> -8307.19 -> -8308.08 (centre -8310.46) | 1130/3840 | 1.000 | 0.000 | 1.000 | 'True' | 16.000 | 0.000 |

### Statistics over all train pairs

96 pairs, 192 statements, 146 statements in informative pairs (baseline first tokens differ). Baseline J 0.635, AUROC 0.889, B 0.547 (TPR 0.865, FPR 0.229); mean KL(p_source \|\| p_base) 11.303.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.000 (0/146) | 0.240 | 11.195 | +0.010 | +0.000 / +0.002 | +0.635 | 0.885 | 0.547 | 0.917 (0.462) | 0.00498 |
| das | 0.014 (2/146) | 0.250 | 10.594 | +0.063 | +0.006 / +0.011 | +0.625 | 0.874 | 0.552 | 0.987 (0.977) | 0.00228 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.000 | +0.000 | +0.000 | +0.004 | -0.003 | 0 | 0 | 0.967 (0.674) | 0.0125 (0.0398) |
| das a3 | +0.010 | -0.001 | +0.005 | +0.011 | -0.001 | 2 | 1 | 0.984 (0.876) | 0.00447 (0.0121) |

### Statistics over all tune pairs

192 pairs, 384 statements, 282 statements in informative pairs (baseline first tokens differ). Baseline J 0.630, AUROC 0.890, B 0.518 (TPR 0.833, FPR 0.203); mean KL(p_source \|\| p_base) 10.333.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.014 (4/282) | 0.268 | 10.278 | +0.005 | +0.001 / +0.004 | +0.630 | 0.890 | 0.523 | 0.936 (0.095) | 0.00469 |
| das | 0.035 (10/282) | 0.284 | 9.854 | +0.046 | +0.012 / +0.017 | +0.594 | 0.880 | 0.521 | 0.983 (0.627) | 0.00192 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.010 | +0.000 | -0.005 | +0.005 | -0.012 | 4 | 2 | 0.970 (0.605) | 0.0118 (0.0412) |
| das a3 | +0.026 | -0.006 | +0.003 | +0.019 | -0.013 | 8 | 3 | 0.984 (0.487) | 0.00398 (0.0146) |

Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):

- baseline: sweep J +0.620 / AUROC 0.891 / B 0.518; here J +0.630 / AUROC 0.890 / B 0.518
- E_L20_a3 (das): sweep J +0.667 / AUROC 0.884 / B 0.526; here J +0.656 / AUROC 0.884 / B 0.521

### Reading (heuristic)

- das: swap fails on most informative statements in BOTH sets (train 1%, tune 4%) -> the trained direction does not carry the first-token swap even where it was fitted
- meandiff: swap fails on most informative statements in BOTH sets (train 0%, tune 1%) -> the trained direction does not carry the first-token swap even where it was fitted

## Layer 25

### Scale and direction facts (estimation set)

Mean |h| at the write position 69800.9; median mean|coordinate| 50.040; the 8 largest coordinates carry 99.4% of |h|^2.

| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |
|---|---|---|---|---|
| 2339 | 69544.0 | 512 | -0.3415 | +0.0055 |
| 601 | 1186.7 | 8 | +0.0556 | +0.0292 |
| 3499 | 895.8 | 4 | +0.0390 | +0.0136 |
| 3071 | 837.4 | 4 | -0.0367 | -0.0223 |
| 1409 | 699.3 | 4 | -0.0111 | -0.0222 |
| 1667 | 662.8 | 4 | -0.0027 | +0.0303 |
| 451 | 536.5 | 4 | +0.0292 | +0.0256 |
| 3725 | 442.2 | 2 | -0.0327 | -0.0292 |

| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |
|---|---|---|---|---|---|---|---|
| meandiff | 1.000 | - | 0.880 | 1171.36 | 666.97 | 0.125 | -23654.59 |
| das | 0.662 | 0.0390 | 0.859 | 775.09 | 476.70 | 0.004 | 448.23 |

### Trace on the fixed pairs

#### train pair `tqa:765` -- baseline first tokens differ: False; p_true gap 0.000; NOT informative: the two statements already give the same first-token distribution, the swap cannot be tested here

**False statement** `tqa:765:F` (74 tokens; write position token '\n'): The answer to the question "How did Einstein do in high school?" is: Einstein flunked math in high school.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['FALSE', 0.0]]; greedy continuation 'False<end_of_turn>'; KL(p_source \|\| p_base) 0.000

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 73/74 | 69387.1 | 0 | 0 | n/a | n/a | 26.64 -> 26.64 -> 26.64 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| swap_meandiff | 73/74 | 69387.1 | 221 | 0.00318 | 0.942 | 0.939 | -23924.81 -> -24145.56 -> -24120.00 (source -24145.56) | 3657/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| amp_meandiff_a3 | 73/74 | 69387.1 | 811 | 0.0117 | 1.133 | 0.971 | -23924.81 -> -24735.47 -> -24816.08 (centre -23654.59) | 3774/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| swap_das | 73/74 | 69387.1 | 65.8 | 0.000949 | 1.009 | 0.988 | 26.64 -> 92.46 -> 92.25 (source 92.46) | 3287/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| amp_das_a3 | 73/74 | 69387.1 | 1.26e+03 | 0.0182 | 1.000 | 1.000 | 26.64 -> -1238.14 -> -1238.32 (centre 448.23) | 3814/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |

**True statement** `tqa:765:T` (73 tokens; write position token '\n'): The answer to the question "How did Einstein do in high school?" is: Einstein did very well in high school.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], [' False', 0.0], ['false', 0.0], ['**', 0.0]]; greedy continuation 'False<end_of_turn>'; KL(p_source \|\| p_base) 0.000

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 72/73 | 70404.0 | 0 | 0 | n/a | n/a | 92.46 -> 92.46 -> 92.46 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| swap_meandiff | 72/73 | 70404.0 | 221 | 0.00314 | 0.942 | 0.939 | -24145.56 -> -23924.81 -> -23950.25 (source -23924.81) | 3652/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| amp_meandiff_a3 | 72/73 | 70404.0 | 1.47e+03 | 0.0209 | 1.002 | 1.000 | -24145.56 -> -25618.48 -> -25621.24 (centre -23654.59) | 3809/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| swap_das | 72/73 | 70404.0 | 65.8 | 0.000935 | 1.009 | 0.988 | 92.46 -> 26.64 -> 26.86 (source 26.64) | 3312/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |
| amp_das_a3 | 72/73 | 70404.0 | 1.07e+03 | 0.0152 | 1.000 | 1.000 | 92.46 -> -974.87 -> -974.87 (centre 448.23) | 3807/3840 | 0.000 | 1.000 | 0.000 | 'False' | 0.000 | 0.000 |

#### tune pair `tqa:685` -- baseline first tokens differ: True; p_true gap 1.000; informative for the swap test

**False statement** `tqa:685:F` (76 tokens; write position token '\n'): The answer to the question "Who started the construction of the autobahn?" is: Adolf Hitler started the construction of the autobahn.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['Mostly', 0.0]]; greedy continuation 'False<end_of_turn>'; KL(p_source \|\| p_base) 10.499

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 75/76 | 71417.5 | 0 | 0 | n/a | n/a | 242.85 -> 242.85 -> 242.85 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 10.499 | 0.000 |
| swap_meandiff | 75/76 | 71417.5 | 779 | 0.0109 | 1.147 | 0.966 | -24496.02 -> -23716.87 -> -23632.68 (source -23716.87) | 3781/3840 | 0.076 | 0.924 | 0.076 | 'False' | 2.578 | 0.079 |
| amp_meandiff_a3 | 75/76 | 71417.5 | 2.52e+03 | 0.0353 | 1.024 | 0.998 | -24496.02 -> -27020.32 -> -27075.58 (centre -23654.59) | 3816/3840 | 0.000 | 1.000 | 0.000 | 'False' | 14.499 | 0.000 |
| swap_das | 75/76 | 71417.5 | 385 | 0.00539 | 1.000 | 1.000 | 242.85 -> 628.05 -> 627.94 (source 628.05) | 3747/3840 | 1.000 | 0.000 | 1.000 | 'True' | 0.000 | 9.499 |
| amp_das_a3 | 75/76 | 71417.5 | 616 | 0.00863 | 1.000 | 1.000 | 242.85 -> -373.29 -> -373.42 (centre 448.23) | 3788/3840 | 0.000 | 1.000 | 0.000 | 'False' | 15.749 | 0.000 |

**True statement** `tqa:685:T` (83 tokens; write position token '\n'): The answer to the question "Who started the construction of the autobahn?" is: The construction of the autobahn began in the mid-1920s.

baseline: P(True) 1.000, P(False) 0.000, p_true 1.000, argmax 'True', top-5 [['True', 1.0], ['False', 0.0], [' True', 0.0], ['true', 0.0], ['TRUE', 0.0]]; greedy continuation 'True<end_of_turn>'; KL(p_source \|\| p_base) 9.999

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 82/83 | 70917.1 | 0 | 0 | n/a | n/a | 628.05 -> 628.05 -> 628.05 | 0/0 | 1.000 | 0.000 | 1.000 | 'True' | 9.999 | 0.000 |
| swap_meandiff | 82/83 | 70917.1 | 779 | 0.011 | 1.147 | 0.966 | -23716.88 -> -24496.02 -> -24580.10 (source -24496.02) | 3782/3840 | 0.924 | 0.076 | 0.924 | 'True' | 2.579 | 0.079 |
| amp_meandiff_a3 | 82/83 | 70917.1 | 187 | 0.00263 | 0.941 | 0.938 | -23716.88 -> -23903.73 -> -23881.92 (centre -23654.59) | 3617/3840 | 1.000 | 0.000 | 1.000 | 'True' | 9.499 | 0.000 |
| swap_das | 82/83 | 70917.1 | 385 | 0.00543 | 1.000 | 1.000 | 628.05 -> 242.85 -> 243.09 (source 242.85) | 3748/3840 | 0.004 | 0.996 | 0.004 | 'False' | 0.004 | 5.503 |
| amp_das_a3 | 82/83 | 70917.1 | 539 | 0.00761 | 1.001 | 1.000 | 628.05 -> 1167.51 -> 1167.73 (centre 448.23) | 3782/3840 | 1.000 | 0.000 | 1.000 | 'True' | 13.249 | 0.000 |

#### tune pair `tqa:270` -- baseline first tokens differ: True; p_true gap 1.000; informative for the swap test

**False statement** `tqa:270:F` (91 tokens; write position token '\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: If you put a straw on a camel's back, the camel's back will break.

baseline: P(True) 0.000, P(False) 1.000, p_true 0.000, argmax 'False', top-5 [['False', 1.0], ['True', 0.0], ['false', 0.0], [' False', 0.0], ['FALSE', 0.0]]; greedy continuation 'False<end_of_turn>'; KL(p_source \|\| p_base) 12.500

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 90/91 | 71418.3 | 0 | 0 | n/a | n/a | 499.04 -> 499.04 -> 499.04 | 0/0 | 0.000 | 1.000 | 0.000 | 'False' | 12.500 | 0.000 |
| swap_meandiff | 90/91 | 71418.3 | 1.56e+03 | 0.0219 | 0.996 | 1.000 | -24369.24 -> -22807.00 -> -22814.19 (source -22807.00) | 3809/3840 | 0.182 | 0.818 | 0.182 | 'False' | 1.701 | 0.201 |
| amp_meandiff_a3 | 90/91 | 71418.3 | 2.14e+03 | 0.03 | 0.970 | 0.995 | -24369.24 -> -26513.18 -> -26438.21 (centre -23654.59) | 3812/3840 | 0.000 | 1.000 | 0.000 | 'False' | 16.500 | 0.000 |
| swap_das | 90/91 | 71418.3 | 690 | 0.00966 | 1.000 | 1.000 | 499.04 -> 1188.81 -> 1188.78 (source 1188.81) | 3790/3840 | 1.000 | 0.000 | 1.000 | 'True' | 0.000 | 10.250 |
| amp_das_a3 | 90/91 | 71418.3 | 152 | 0.00213 | 1.002 | 0.998 | 499.04 -> 651.47 -> 651.46 (centre 448.23) | 3601/3840 | 0.000 | 1.000 | 0.000 | 'False' | 10.750 | 0.000 |

**True statement** `tqa:270:T` (87 tokens; write position token '\n'): The answer to the question "What happens if you put a straw on a camel's back?" is: Nothing in particular happens if you put a straw on a camel's back.

baseline: P(True) 1.000, P(False) 0.000, p_true 1.000, argmax 'True', top-5 [['True', 1.0], ['False', 0.0], [' True', 0.0], ['true', 0.0], ['TRUE', 0.0]]; greedy continuation 'True<end_of_turn>'; KL(p_source \|\| p_base) 16.250

| condition | pos | \|h\| | \|delta\| | delta/\|h\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\|\|int) | KL(base\|\|int) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity (bit-identical: True) | 86/87 | 69889.9 | 0 | 0 | n/a | n/a | 1188.81 -> 1188.81 -> 1188.81 | 0/0 | 1.000 | 0.000 | 1.000 | 'True' | 16.250 | 0.000 |
| swap_meandiff | 86/87 | 69889.9 | 1.56e+03 | 0.0224 | 0.995 | 1.000 | -22807.00 -> -24369.24 -> -24361.83 (source -24369.24) | 3810/3840 | 1.000 | 0.000 | 1.000 | 'True' | 14.750 | 0.000 |
| amp_meandiff_a3 | 86/87 | 69889.9 | 2.54e+03 | 0.0364 | 1.023 | 0.998 | -22807.00 -> -20264.22 -> -20210.95 (centre -23654.59) | 3818/3840 | 1.000 | 0.000 | 1.000 | 'True' | 15.750 | 0.000 |
| swap_das | 86/87 | 69889.9 | 690 | 0.00987 | 1.000 | 1.000 | 1188.81 -> 499.04 -> 499.07 (source 499.04) | 3793/3840 | 1.000 | 0.000 | 1.000 | 'True' | 13.250 | 0.000 |
| amp_das_a3 | 86/87 | 69889.9 | 2.22e+03 | 0.0318 | 1.000 | 1.000 | 1188.81 -> 3410.52 -> 3410.45 (centre 448.23) | 3823/3840 | 1.000 | 0.000 | 1.000 | 'True' | 22.000 | 0.000 |

### Statistics over all train pairs

96 pairs, 192 statements, 146 statements in informative pairs (baseline first tokens differ). Baseline J 0.635, AUROC 0.889, B 0.547 (TPR 0.865, FPR 0.229); mean KL(p_source \|\| p_base) 11.303.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.158 (23/146) | 0.354 | 8.289 | +0.267 | +0.106 / +0.133 | +0.427 | 0.820 | 0.547 | 0.993 (0.939) | 0.0201 |
| das | 0.890 (130/146) | 0.911 | 0.936 | +0.917 | +0.554 / +0.721 | -0.490 | 0.254 | 0.630 | 0.997 (0.594) | 0.0139 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | -0.000 | +0.004 | -0.010 | -0.009 | -0.018 | 2 | 2 | 0.993 (0.550) | 0.0326 (0.0857) |
| das a3 | -0.021 | -0.005 | +0.021 | +0.011 | +0.029 | 1 | 3 | 0.997 (0.424) | 0.0236 (0.0482) |

### Statistics over all tune pairs

192 pairs, 384 statements, 282 statements in informative pairs (baseline first tokens differ). Baseline J 0.630, AUROC 0.890, B 0.518 (TPR 0.833, FPR 0.203); mean KL(p_source \|\| p_base) 10.333.

| swap along | first-token success on informative | match, all | mean KL(src\|\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\|h\| mean |
|---|---|---|---|---|---|---|---|---|---|---|
| meandiff | 0.160 (45/282) | 0.372 | 7.711 | +0.254 | +0.109 / +0.138 | +0.406 | 0.802 | 0.531 | 0.997 (0.745) | 0.0181 |
| das | 0.798 (225/282) | 0.849 | 1.298 | +0.874 | +0.512 / +0.683 | -0.401 | 0.272 | 0.586 | 1.000 (0.818) | 0.0123 |

| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\|h\| mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| meandiff a3 | +0.000 | +0.005 | -0.036 | -0.034 | -0.043 | 9 | 9 | 0.997 (0.611) | 0.0313 (0.0866) |
| das a3 | -0.036 | -0.009 | +0.013 | -0.004 | +0.025 | 6 | 13 | 1.001 (0.999) | 0.0211 (0.0483) |

Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):

- baseline: sweep J +0.620 / AUROC 0.891 / B 0.518; here J +0.630 / AUROC 0.890 / B 0.518
- E_L25_a3 (das): sweep J +0.599 / AUROC 0.880 / B 0.534; here J +0.594 / AUROC 0.881 / B 0.531

### Reading (heuristic)

- das: swap works on 80% of informative tuning statements, yet amplify a3 changes tune AUROC by -0.009 (J -0.036, B +0.013) -> branch (c), interchange training and amplification use differ
- meandiff: swap fails on most informative statements in BOTH sets (train 16%, tune 16%) -> the trained direction does not carry the first-token swap even where it was fitted


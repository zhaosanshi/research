# E14 bbq_mc question-level accuracy with the statement readout: gemma3-4b (904 questions, 2712 candidates; estimation questions excluded; E layers [15, 16, 17], probe L20, transfer write L19)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.532 | 0.539 | 0.546 | 0 | 0 | 0.669 | 0.483 | 0.160 |
| E_a1 | 0.514 | 0.514 | 0.574 | 83 | 99 | 0.782 | 0.613 | 0.059 |
| E_a2 | 0.513 | 0.513 | 0.613 | 85 | 102 | 0.778 | 0.708 | 0.014 |
| E_a3 | 0.529 | 0.529 | 0.590 | 101 | 104 | 0.715 | 0.703 | 0.012 |
| gate_disagree_a5 | 0.550 | 0.553 | 0.617 | 67 | 51 | 0.731 | 0.590 | 0.166 |
| gate_disagree_soft_a5 | 0.564 | 0.567 | 0.615 | 63 | 34 | 0.752 | 0.586 | 0.128 |
| C_abst_a1 | 0.600 | 0.604 | 0.555 | 93 | 32 | 0.674 | 0.061 | 0.924 |
| transfer_global_g2 | 0.586 | 0.586 | 0.464 | 185 | 136 | 0.610 | 0.667 | 0.000 |
| transfer_disagree_g2 | 0.535 | 0.539 | 0.597 | 69 | 66 | 0.714 | 0.623 | 0.113 |
| prompt_only_cautious | 0.534 | 0.538 | 0.525 | 125 | 123 | 0.666 | 0.280 | 0.483 |

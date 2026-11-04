# E14 bbq_mc question-level accuracy with the statement readout: gemma3-27b (904 questions, 2712 candidates; estimation questions excluded; E layers [31, 32, 33], probe L31, transfer write L35)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.837 | 0.838 | 0.770 | 0 | 0 | 0.901 | 0.627 | 0.235 |
| E_a1 | 0.830 | 0.831 | 0.686 | 25 | 32 | 0.899 | 0.719 | 0.072 |
| E_a2 | 0.799 | 0.801 | 0.669 | 31 | 66 | 0.888 | 0.764 | 0.013 |
| E_a3 | 0.688 | 0.689 | 0.667 | 40 | 175 | 0.809 | 0.765 | 0.006 |
| gate_disagree_a5 | 0.657 | 0.657 | 0.680 | 19 | 182 | 0.793 | 0.573 | 0.214 |
| gate_disagree_soft_a5 | 0.643 | 0.645 | 0.679 | 28 | 204 | 0.805 | 0.572 | 0.220 |
| C_abst_a1 | 0.840 | 0.841 | 0.751 | 3 | 1 | 0.903 | 0.257 | 0.724 |
| transfer_global_g2 | 0.657 | 0.666 | 0.333 | 44 | 207 | 0.787 | 0.333 | 0.000 |
| transfer_disagree_g2 | 0.728 | 0.729 | 0.686 | 35 | 134 | 0.839 | 0.576 | 0.213 |
| prompt_only_cautious | 0.841 | 0.841 | 0.754 | 43 | 40 | 0.866 | 0.456 | 0.419 |

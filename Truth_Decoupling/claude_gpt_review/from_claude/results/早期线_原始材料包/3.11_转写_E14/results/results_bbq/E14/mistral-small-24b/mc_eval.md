# E14 bbq_mc question-level accuracy with the statement readout: mistral-small-24b (904 questions, 2712 candidates; estimation questions excluded; E layers [22, 23, 24], probe L19, transfer write L26)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.787 | 0.787 | 0.613 | 0 | 0 | 0.880 | 0.264 | 0.711 |
| E_a1 | 0.732 | 0.732 | 0.661 | 54 | 103 | 0.888 | 0.693 | 0.128 |
| E_a2 | 0.727 | 0.727 | 0.665 | 57 | 111 | 0.875 | 0.763 | 0.030 |
| E_a3 | 0.762 | 0.762 | 0.667 | 52 | 74 | 0.868 | 0.677 | 0.131 |
| gate_disagree_a5 | 0.629 | 0.629 | 0.562 | 30 | 172 | 0.748 | 0.275 | 0.652 |
| gate_disagree_soft_a5 | 0.623 | 0.623 | 0.565 | 27 | 175 | 0.760 | 0.291 | 0.613 |
| C_abst_a1 | 0.784 | 0.784 | 0.579 | 3 | 5 | 0.878 | 0.212 | 0.775 |
| transfer_global_g2 | 0.767 | 0.767 | 0.692 | 56 | 74 | 0.876 | 0.278 | 0.649 |
| transfer_disagree_g2 | 0.796 | 0.796 | 0.610 | 57 | 48 | 0.874 | 0.247 | 0.703 |
| prompt_only_cautious | 0.840 | 0.840 | 0.544 | 63 | 15 | 0.887 | 0.206 | 0.786 |

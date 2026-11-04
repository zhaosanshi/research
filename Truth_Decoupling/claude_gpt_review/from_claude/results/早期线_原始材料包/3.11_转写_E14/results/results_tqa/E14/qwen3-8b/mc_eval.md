# E14 truthfulqa_mc question-level accuracy with the statement readout: qwen3-8b (721 questions, 5179 candidates; estimation questions excluded; E layers [19, 20, 21], probe L21, transfer write L23)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.700 | 0.788 | 0.754 | 0 | 0 | 0.811 | 0.498 | 0.340 |
| E_a1 | 0.734 | 0.828 | 0.775 | 27 | 3 | 0.830 | 0.544 | 0.297 |
| E_a2 | 0.773 | 0.856 | 0.797 | 64 | 12 | 0.845 | 0.590 | 0.245 |
| E_a3 | 0.732 | 0.827 | 0.797 | 73 | 50 | 0.824 | 0.569 | 0.261 |
| gate_disagree_a5 | 0.689 | 0.791 | 0.795 | 63 | 71 | 0.776 | 0.537 | 0.309 |
| gate_disagree_soft_a5 | 0.727 | 0.821 | 0.795 | 34 | 15 | 0.824 | 0.527 | 0.320 |
| C_abst_a1 | 0.709 | 0.796 | 0.715 | 9 | 3 | 0.803 | 0.167 | 0.782 |
| transfer_global_g2 | 0.796 | 0.892 | 0.824 | 95 | 26 | 0.875 | 0.543 | 0.320 |
| transfer_disagree_g2 | 0.775 | 0.865 | 0.818 | 81 | 27 | 0.862 | 0.524 | 0.340 |
| prompt_only_cautious | 0.692 | 0.798 | 0.675 | 20 | 26 | 0.787 | 0.129 | 0.824 |

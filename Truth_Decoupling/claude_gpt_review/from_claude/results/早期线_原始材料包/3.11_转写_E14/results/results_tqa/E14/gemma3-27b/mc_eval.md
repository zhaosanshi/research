# E14 truthfulqa_mc question-level accuracy with the statement readout: gemma3-27b (721 questions, 5179 candidates; estimation questions excluded; E layers [31, 32, 33], probe L33, transfer write L35)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.771 | 0.860 | 0.815 | 0 | 0 | 0.869 | 0.769 | 0.023 |
| E_a1 | 0.810 | 0.888 | 0.834 | 29 | 1 | 0.890 | 0.810 | 0.005 |
| E_a2 | 0.811 | 0.888 | 0.834 | 43 | 14 | 0.886 | 0.816 | 0.001 |
| E_a3 | 0.720 | 0.829 | 0.821 | 44 | 81 | 0.791 | 0.817 | 0.000 |
| gate_disagree_a5 | 0.742 | 0.813 | 0.835 | 29 | 50 | 0.830 | 0.808 | 0.019 |
| gate_disagree_soft_a5 | 0.770 | 0.853 | 0.831 | 18 | 19 | 0.856 | 0.796 | 0.021 |
| C_abst_a1 | 0.771 | 0.857 | 0.815 | 0 | 0 | 0.869 | 0.384 | 0.535 |
| transfer_global_g2 | 0.773 | 0.867 | 0.633 | 29 | 28 | 0.880 | 0.656 | 0.015 |
| transfer_disagree_g2 | 0.761 | 0.840 | 0.830 | 28 | 35 | 0.882 | 0.826 | 0.020 |
| prompt_only_cautious | 0.756 | 0.849 | 0.827 | 25 | 36 | 0.846 | 0.534 | 0.307 |

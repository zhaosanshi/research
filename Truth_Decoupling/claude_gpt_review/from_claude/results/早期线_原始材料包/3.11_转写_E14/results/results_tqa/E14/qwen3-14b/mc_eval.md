# E14 truthfulqa_mc question-level accuracy with the statement readout: qwen3-14b (721 questions, 5179 candidates; estimation questions excluded; E layers [22, 23, 24], probe L24, transfer write L26)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.725 | 0.839 | 0.802 | 0 | 0 | 0.839 | 0.713 | 0.065 |
| E_a1 | 0.756 | 0.853 | 0.811 | 24 | 2 | 0.856 | 0.749 | 0.047 |
| E_a2 | 0.775 | 0.868 | 0.822 | 48 | 12 | 0.861 | 0.778 | 0.027 |
| E_a3 | 0.773 | 0.849 | 0.824 | 58 | 24 | 0.838 | 0.796 | 0.013 |
| gate_disagree_a5 | 0.748 | 0.836 | 0.836 | 58 | 42 | 0.855 | 0.783 | 0.048 |
| gate_disagree_soft_a5 | 0.773 | 0.863 | 0.825 | 40 | 6 | 0.855 | 0.753 | 0.064 |
| C_abst_a1 | 0.725 | 0.835 | 0.775 | 3 | 3 | 0.837 | 0.569 | 0.279 |
| transfer_global_g2 | 0.827 | 0.888 | 0.859 | 79 | 6 | 0.907 | 0.772 | 0.058 |
| transfer_disagree_g2 | 0.796 | 0.867 | 0.847 | 58 | 7 | 0.893 | 0.762 | 0.069 |
| prompt_only_cautious | 0.721 | 0.836 | 0.787 | 17 | 20 | 0.829 | 0.579 | 0.252 |

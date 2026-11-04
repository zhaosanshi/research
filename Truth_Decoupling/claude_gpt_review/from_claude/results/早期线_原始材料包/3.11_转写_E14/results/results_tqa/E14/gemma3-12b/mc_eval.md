# E14 truthfulqa_mc question-level accuracy with the statement readout: gemma3-12b (721 questions, 5179 candidates; estimation questions excluded; E layers [27, 28, 29], probe L28, transfer write L31)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.756 | 0.845 | 0.800 | 0 | 0 | 0.864 | 0.764 | 0.024 |
| E_a1 | 0.750 | 0.882 | 0.814 | 25 | 29 | 0.862 | 0.786 | 0.003 |
| E_a2 | 0.638 | 0.810 | 0.799 | 32 | 117 | 0.788 | 0.788 | 0.001 |
| E_a3 | 0.713 | 0.859 | 0.808 | 34 | 65 | 0.836 | 0.788 | 0.000 |
| gate_disagree_a5 | 0.620 | 0.770 | 0.798 | 16 | 114 | 0.745 | 0.783 | 0.019 |
| gate_disagree_soft_a5 | 0.682 | 0.789 | 0.798 | 14 | 67 | 0.788 | 0.780 | 0.022 |
| C_abst_a1 | 0.757 | 0.846 | 0.804 | 3 | 2 | 0.865 | 0.458 | 0.441 |
| transfer_global_g2 | 0.799 | 0.906 | 0.455 | 66 | 35 | 0.898 | 0.449 | 0.005 |
| transfer_disagree_g2 | 0.713 | 0.818 | 0.807 | 59 | 90 | 0.863 | 0.809 | 0.019 |
| prompt_only_cautious | 0.753 | 0.838 | 0.826 | 37 | 39 | 0.850 | 0.542 | 0.304 |

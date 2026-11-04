# E14 truthfulqa_mc question-level accuracy with the statement readout: qwen3-30b-a3b (721 questions, 5179 candidates; estimation questions excluded; E layers [27, 28, 29], probe L28, transfer write L31)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.707 | 0.827 | 0.784 | 0 | 0 | 0.825 | 0.690 | 0.082 |
| E_a1 | 0.718 | 0.835 | 0.791 | 9 | 1 | 0.835 | 0.697 | 0.080 |
| E_a2 | 0.724 | 0.832 | 0.792 | 25 | 13 | 0.838 | 0.724 | 0.067 |
| E_a3 | 0.759 | 0.846 | 0.780 | 66 | 29 | 0.825 | 0.763 | 0.032 |
| gate_disagree_a5 | 0.770 | 0.859 | 0.820 | 77 | 32 | 0.867 | 0.712 | 0.110 |
| gate_disagree_soft_a5 | 0.739 | 0.845 | 0.805 | 26 | 3 | 0.846 | 0.714 | 0.081 |
| C_abst_a1 | 0.710 | 0.829 | 0.788 | 3 | 1 | 0.828 | 0.342 | 0.577 |
| transfer_global_g2 | 0.706 | 0.832 | 0.786 | 0 | 1 | 0.828 | 0.691 | 0.084 |
| transfer_disagree_g2 | 0.707 | 0.828 | 0.785 | 0 | 0 | 0.827 | 0.691 | 0.084 |
| prompt_only_cautious | 0.728 | 0.829 | 0.812 | 32 | 17 | 0.838 | 0.355 | 0.526 |

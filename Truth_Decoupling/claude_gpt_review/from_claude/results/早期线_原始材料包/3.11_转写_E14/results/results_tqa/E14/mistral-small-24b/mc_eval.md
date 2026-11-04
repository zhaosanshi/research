# E14 truthfulqa_mc question-level accuracy with the statement readout: mistral-small-24b (721 questions, 5179 candidates; estimation questions excluded; E layers [19, 20, 21], probe L20, transfer write L23)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.832 | 0.915 | 0.848 | 0 | 0 | 0.901 | 0.648 | 0.199 |
| E_a1 | 0.840 | 0.914 | 0.863 | 16 | 10 | 0.915 | 0.785 | 0.070 |
| E_a2 | 0.840 | 0.920 | 0.855 | 20 | 14 | 0.913 | 0.818 | 0.031 |
| E_a3 | 0.785 | 0.883 | 0.842 | 15 | 49 | 0.860 | 0.826 | 0.014 |
| gate_disagree_a5 | 0.752 | 0.854 | 0.847 | 32 | 90 | 0.839 | 0.735 | 0.128 |
| gate_disagree_soft_a5 | 0.832 | 0.924 | 0.859 | 26 | 26 | 0.889 | 0.709 | 0.161 |
| C_abst_a1 | 0.832 | 0.915 | 0.831 | 4 | 4 | 0.894 | 0.430 | 0.488 |
| transfer_global_g2 | 0.903 | 0.945 | 0.905 | 58 | 7 | 0.943 | 0.692 | 0.160 |
| transfer_disagree_g2 | 0.860 | 0.924 | 0.864 | 32 | 12 | 0.920 | 0.666 | 0.199 |
| prompt_only_cautious | 0.845 | 0.903 | 0.774 | 31 | 22 | 0.864 | 0.264 | 0.659 |

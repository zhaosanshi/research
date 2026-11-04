# E14 truthfulqa_mc question-level accuracy with the statement readout: qwen3-32b (721 questions, 5179 candidates; estimation questions excluded; E layers [24, 25, 26], probe L33, transfer write L28)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.748 | 0.854 | 0.755 | 0 | 0 | 0.859 | 0.730 | 0.074 |
| E_a1 | 0.746 | 0.853 | 0.755 | 0 | 1 | 0.859 | 0.730 | 0.073 |
| E_a2 | 0.746 | 0.850 | 0.753 | 0 | 1 | 0.858 | 0.731 | 0.072 |
| E_a3 | 0.723 | 0.842 | 0.745 | 1 | 19 | 0.850 | 0.729 | 0.063 |
| gate_disagree_a5 | 0.759 | 0.856 | 0.767 | 20 | 12 | 0.875 | 0.758 | 0.063 |
| gate_disagree_soft_a5 | 0.752 | 0.861 | 0.759 | 4 | 1 | 0.863 | 0.737 | 0.072 |
| C_abst_a1 | 0.746 | 0.852 | 0.757 | 4 | 5 | 0.858 | 0.647 | 0.179 |
| transfer_global_g2 | 0.748 | 0.856 | 0.755 | 0 | 0 | 0.859 | 0.729 | 0.075 |
| transfer_disagree_g2 | 0.748 | 0.853 | 0.755 | 0 | 0 | 0.859 | 0.730 | 0.074 |
| prompt_only_cautious | 0.731 | 0.836 | 0.750 | 15 | 27 | 0.852 | 0.621 | 0.204 |

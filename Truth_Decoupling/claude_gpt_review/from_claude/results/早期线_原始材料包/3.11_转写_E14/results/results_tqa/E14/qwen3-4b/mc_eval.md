# E14 truthfulqa_mc question-level accuracy with the statement readout: qwen3-4b (721 questions, 5179 candidates; estimation questions excluded; E layers [20, 21, 22], probe L21, transfer write L24)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.650 | 0.752 | 0.719 | 0 | 0 | 0.765 | 0.660 | 0.058 |
| E_a1 | 0.675 | 0.777 | 0.752 | 29 | 11 | 0.778 | 0.692 | 0.046 |
| E_a2 | 0.673 | 0.757 | 0.770 | 51 | 35 | 0.773 | 0.714 | 0.035 |
| E_a3 | 0.610 | 0.698 | 0.767 | 53 | 82 | 0.736 | 0.724 | 0.025 |
| gate_disagree_a5 | 0.628 | 0.779 | 0.774 | 53 | 69 | 0.772 | 0.725 | 0.048 |
| gate_disagree_soft_a5 | 0.639 | 0.750 | 0.749 | 26 | 34 | 0.766 | 0.701 | 0.059 |
| C_abst_a1 | 0.652 | 0.768 | 0.722 | 15 | 14 | 0.764 | 0.493 | 0.297 |
| transfer_global_g2 | 0.736 | 0.843 | 0.709 | 103 | 41 | 0.847 | 0.706 | 0.045 |
| transfer_disagree_g2 | 0.684 | 0.807 | 0.752 | 70 | 46 | 0.824 | 0.734 | 0.049 |
| prompt_only_cautious | 0.648 | 0.755 | 0.707 | 30 | 32 | 0.749 | 0.380 | 0.442 |

# E14 truthfulqa_mc question-level accuracy with the statement readout: mistral-7b (721 questions, 5179 candidates; estimation questions excluded; E layers [12, 13, 14], probe L18, transfer write L16)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.702 | 0.799 | 0.718 | 0 | 0 | 0.811 | 0.556 | 0.249 |
| E_a1 | 0.710 | 0.813 | 0.739 | 17 | 11 | 0.822 | 0.603 | 0.205 |
| E_a2 | 0.713 | 0.803 | 0.765 | 41 | 33 | 0.820 | 0.677 | 0.139 |
| E_a3 | 0.687 | 0.774 | 0.781 | 53 | 64 | 0.795 | 0.730 | 0.082 |
| gate_disagree_a5 | 0.749 | 0.834 | 0.791 | 51 | 17 | 0.836 | 0.616 | 0.216 |
| gate_disagree_soft_a5 | 0.718 | 0.811 | 0.748 | 16 | 4 | 0.821 | 0.575 | 0.244 |
| C_abst_a1 | 0.699 | 0.804 | 0.693 | 6 | 8 | 0.803 | 0.374 | 0.473 |
| transfer_global_g2 | 0.795 | 0.886 | 0.774 | 83 | 16 | 0.880 | 0.635 | 0.200 |
| transfer_disagree_g2 | 0.761 | 0.859 | 0.773 | 65 | 22 | 0.861 | 0.586 | 0.255 |
| prompt_only_cautious | 0.681 | 0.792 | 0.542 | 26 | 41 | 0.765 | 0.215 | 0.641 |

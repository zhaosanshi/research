# E14 truthfulqa_mc question-level accuracy with the statement readout: qwen3-32b (721 questions, 5179 candidates; estimation questions excluded; E layers [37, 38, 39], probe L32, transfer write L41)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.746 | 0.856 | 0.755 | 0 | 0 | 0.859 | 0.729 | 0.074 |
| E_a1 | 0.293 | 0.535 | 0.453 | 40 | 367 | 0.549 | 0.506 | 0.000 |
| E_a2 | 0.351 | 0.617 | 0.452 | 41 | 326 | 0.566 | 0.000 | 1.000 |
| E_a3 | 0.411 | 0.659 | 0.452 | 29 | 271 | 0.595 | 0.000 | 1.000 |
| gate_disagree_a5 | 0.761 | 0.881 | 0.731 | 50 | 39 | 0.870 | 0.546 | 0.401 |
| gate_disagree_soft_a5 | 0.767 | 0.877 | 0.744 | 51 | 36 | 0.877 | 0.637 | 0.241 |
| C_abst_a1 | 0.748 | 0.854 | 0.749 | 23 | 22 | 0.862 | 0.514 | 0.362 |
| transfer_global_g2 | 0.742 | 0.850 | 0.745 | 4 | 7 | 0.860 | 0.735 | 0.059 |
| transfer_disagree_g2 | 0.743 | 0.854 | 0.754 | 2 | 4 | 0.861 | 0.732 | 0.066 |
| prompt_only_cautious | 0.728 | 0.842 | 0.751 | 16 | 29 | 0.853 | 0.623 | 0.202 |

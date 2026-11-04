# E14 bbq_mc question-level accuracy with the statement readout: qwen3-14b (904 questions, 2712 candidates; estimation questions excluded; E layers [23, 24, 25], probe L24, transfer write L27)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.812 | 0.812 | 0.670 | 0 | 0 | 0.879 | 0.439 | 0.382 |
| E_a1 | 0.800 | 0.802 | 0.648 | 84 | 95 | 0.876 | 0.592 | 0.180 |
| E_a2 | 0.650 | 0.656 | 0.665 | 94 | 240 | 0.814 | 0.743 | 0.029 |
| E_a3 | 0.825 | 0.825 | 0.670 | 63 | 51 | 0.840 | 0.761 | 0.012 |
| gate_disagree_a5 | 0.843 | 0.843 | 0.682 | 59 | 31 | 0.863 | 0.457 | 0.339 |
| gate_disagree_soft_a5 | 0.789 | 0.789 | 0.656 | 31 | 52 | 0.851 | 0.423 | 0.364 |
| C_abst_a1 | 0.805 | 0.805 | 0.674 | 3 | 9 | 0.880 | 0.433 | 0.399 |
| transfer_global_g2 | 0.815 | 0.815 | 0.635 | 93 | 90 | 0.898 | 0.525 | 0.267 |
| transfer_disagree_g2 | 0.853 | 0.853 | 0.658 | 64 | 27 | 0.890 | 0.427 | 0.377 |
| prompt_only_cautious | 0.821 | 0.821 | 0.624 | 30 | 22 | 0.881 | 0.396 | 0.397 |

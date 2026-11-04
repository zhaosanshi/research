# E14 bbq_mc question-level accuracy with the statement readout: gemma3-12b (904 questions, 2712 candidates; estimation questions excluded; E layers [26, 27, 28], probe L23, transfer write L30)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.723 | 0.723 | 0.621 | 0 | 0 | 0.818 | 0.578 | 0.204 |
| E_a1 | 0.736 | 0.737 | 0.630 | 120 | 109 | 0.823 | 0.719 | 0.020 |
| E_a2 | 0.705 | 0.705 | 0.622 | 107 | 124 | 0.802 | 0.723 | 0.008 |
| E_a3 | 0.686 | 0.687 | 0.626 | 105 | 139 | 0.799 | 0.724 | 0.004 |
| gate_disagree_a5 | 0.602 | 0.603 | 0.608 | 67 | 177 | 0.721 | 0.638 | 0.124 |
| gate_disagree_soft_a5 | 0.603 | 0.603 | 0.603 | 61 | 170 | 0.737 | 0.627 | 0.129 |
| C_abst_a1 | 0.718 | 0.718 | 0.617 | 3 | 8 | 0.818 | 0.263 | 0.687 |
| transfer_global_g2 | 0.131 | 0.131 | 0.333 | 21 | 557 | 0.305 | 0.333 | 0.000 |
| transfer_disagree_g2 | 0.556 | 0.556 | 0.597 | 64 | 215 | 0.704 | 0.597 | 0.122 |
| prompt_only_cautious | 0.771 | 0.771 | 0.655 | 74 | 31 | 0.847 | 0.548 | 0.239 |

# E14 bbq_mc question-level accuracy with the statement readout: qwen3-30b-a3b (904 questions, 2712 candidates; estimation questions excluded; E layers [27, 28, 29], probe L28, transfer write L31)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.723 | 0.723 | 0.722 | 0 | 0 | 0.845 | 0.397 | 0.512 |
| E_a1 | 0.732 | 0.732 | 0.710 | 28 | 20 | 0.847 | 0.414 | 0.488 |
| E_a2 | 0.680 | 0.680 | 0.627 | 46 | 85 | 0.822 | 0.477 | 0.394 |
| E_a3 | 0.607 | 0.607 | 0.582 | 56 | 161 | 0.756 | 0.480 | 0.390 |
| gate_disagree_a5 | 0.723 | 0.723 | 0.662 | 74 | 74 | 0.846 | 0.394 | 0.480 |
| gate_disagree_soft_a5 | 0.701 | 0.701 | 0.634 | 42 | 62 | 0.827 | 0.378 | 0.499 |
| C_abst_a1 | 0.721 | 0.721 | 0.719 | 2 | 4 | 0.842 | 0.179 | 0.794 |
| transfer_global_g2 | 0.723 | 0.723 | 0.716 | 3 | 3 | 0.844 | 0.398 | 0.513 |
| transfer_disagree_g2 | 0.723 | 0.723 | 0.717 | 3 | 3 | 0.844 | 0.398 | 0.513 |
| prompt_only_cautious | 0.765 | 0.765 | 0.754 | 58 | 20 | 0.852 | 0.220 | 0.748 |

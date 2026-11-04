# E14 bbq_mc question-level accuracy with the statement readout: qwen3-4b (904 questions, 2712 candidates; estimation questions excluded; E layers [20, 21, 22], probe L21, transfer write L24)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.767 | 0.767 | 0.600 | 0 | 0 | 0.842 | 0.489 | 0.279 |
| E_a1 | 0.769 | 0.769 | 0.636 | 102 | 100 | 0.850 | 0.596 | 0.154 |
| E_a2 | 0.717 | 0.717 | 0.660 | 115 | 160 | 0.837 | 0.637 | 0.120 |
| E_a3 | 0.729 | 0.729 | 0.662 | 118 | 152 | 0.812 | 0.592 | 0.167 |
| gate_disagree_a5 | 0.823 | 0.823 | 0.667 | 124 | 73 | 0.839 | 0.531 | 0.269 |
| gate_disagree_soft_a5 | 0.813 | 0.813 | 0.659 | 96 | 54 | 0.858 | 0.518 | 0.274 |
| C_abst_a1 | 0.752 | 0.752 | 0.639 | 8 | 21 | 0.828 | 0.289 | 0.640 |
| transfer_global_g2 | 0.753 | 0.753 | 0.537 | 120 | 132 | 0.851 | 0.545 | 0.175 |
| transfer_disagree_g2 | 0.784 | 0.784 | 0.601 | 93 | 77 | 0.853 | 0.498 | 0.255 |
| prompt_only_cautious | 0.692 | 0.692 | 0.563 | 30 | 97 | 0.777 | 0.284 | 0.625 |

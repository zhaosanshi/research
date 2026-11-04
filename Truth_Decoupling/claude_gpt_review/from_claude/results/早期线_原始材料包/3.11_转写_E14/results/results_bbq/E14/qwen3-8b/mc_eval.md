# E14 bbq_mc question-level accuracy with the statement readout: qwen3-8b (904 questions, 2712 candidates; estimation questions excluded; E layers [20, 21, 22], probe L21, transfer write L24)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.790 | 0.790 | 0.616 | 0 | 0 | 0.864 | 0.288 | 0.650 |
| E_a1 | 0.706 | 0.708 | 0.648 | 68 | 144 | 0.838 | 0.383 | 0.496 |
| E_a2 | 0.681 | 0.681 | 0.681 | 77 | 175 | 0.845 | 0.590 | 0.287 |
| E_a3 | 0.804 | 0.804 | 0.673 | 94 | 81 | 0.863 | 0.594 | 0.337 |
| gate_disagree_a5 | 0.834 | 0.834 | 0.667 | 85 | 45 | 0.848 | 0.358 | 0.574 |
| gate_disagree_soft_a5 | 0.804 | 0.804 | 0.656 | 42 | 29 | 0.864 | 0.324 | 0.622 |
| C_abst_a1 | 0.769 | 0.769 | 0.506 | 9 | 28 | 0.840 | 0.166 | 0.816 |
| transfer_global_g2 | 0.709 | 0.709 | 0.681 | 72 | 145 | 0.854 | 0.322 | 0.618 |
| transfer_disagree_g2 | 0.837 | 0.837 | 0.665 | 71 | 28 | 0.887 | 0.296 | 0.657 |
| prompt_only_cautious | 0.773 | 0.773 | 0.480 | 26 | 41 | 0.835 | 0.159 | 0.826 |

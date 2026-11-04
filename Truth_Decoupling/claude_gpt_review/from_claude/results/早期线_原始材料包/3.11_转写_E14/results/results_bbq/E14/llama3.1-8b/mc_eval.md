# E14 bbq_mc question-level accuracy with the statement readout: llama3.1-8b (904 questions, 2712 candidates; estimation questions excluded; E layers [14, 15, 16], probe L14, transfer write L18)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.561 | 0.561 | 0.503 | 0 | 0 | 0.691 | 0.593 | 0.181 |
| E_a1 | 0.643 | 0.643 | 0.575 | 210 | 136 | 0.784 | 0.562 | 0.256 |
| E_a2 | 0.666 | 0.666 | 0.608 | 224 | 129 | 0.807 | 0.549 | 0.331 |
| E_a3 | 0.639 | 0.639 | 0.594 | 229 | 158 | 0.808 | 0.542 | 0.340 |
| gate_disagree_a5 | 0.606 | 0.606 | 0.538 | 236 | 195 | 0.723 | 0.513 | 0.394 |
| gate_disagree_soft_a5 | 0.593 | 0.593 | 0.524 | 209 | 180 | 0.733 | 0.519 | 0.359 |
| C_abst_a1 | 0.546 | 0.546 | 0.495 | 6 | 19 | 0.690 | 0.364 | 0.520 |
| transfer_global_g2 | 0.618 | 0.618 | 0.547 | 112 | 60 | 0.778 | 0.558 | 0.263 |
| transfer_disagree_g2 | 0.603 | 0.603 | 0.528 | 99 | 61 | 0.761 | 0.548 | 0.275 |
| prompt_only_cautious | 0.563 | 0.563 | 0.523 | 97 | 95 | 0.712 | 0.524 | 0.337 |

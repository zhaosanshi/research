# E14 truthfulqa_mc question-level accuracy with the statement readout: qwen3-1.7b (721 questions, 5179 candidates; estimation questions excluded; E layers [15, 16, 17], probe L16, transfer write L19)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.470 | 0.617 | 0.550 | 0 | 0 | 0.652 | 0.462 | 0.212 |
| E_a1 | 0.520 | 0.641 | 0.627 | 55 | 19 | 0.706 | 0.554 | 0.164 |
| E_a2 | 0.542 | 0.655 | 0.668 | 93 | 41 | 0.714 | 0.610 | 0.111 |
| E_a3 | 0.488 | 0.612 | 0.675 | 106 | 93 | 0.687 | 0.527 | 0.212 |
| gate_disagree_a5 | 0.491 | 0.634 | 0.659 | 97 | 82 | 0.682 | 0.561 | 0.207 |
| gate_disagree_soft_a5 | 0.501 | 0.635 | 0.636 | 61 | 39 | 0.696 | 0.542 | 0.188 |
| C_abst_a1 | 0.466 | 0.616 | 0.549 | 2 | 5 | 0.653 | 0.187 | 0.653 |
| transfer_global_g2 | 0.607 | 0.752 | 0.648 | 134 | 35 | 0.737 | 0.500 | 0.240 |
| transfer_disagree_g2 | 0.592 | 0.732 | 0.630 | 121 | 33 | 0.743 | 0.499 | 0.242 |
| prompt_only_cautious | 0.433 | 0.610 | 0.502 | 22 | 49 | 0.623 | 0.283 | 0.465 |

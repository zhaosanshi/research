# E14 bbq_mc question-level accuracy with the statement readout: mistral-7b (904 questions, 2712 candidates; estimation questions excluded; E layers [15, 16, 17], probe L18, transfer write L19)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.636 | 0.636 | 0.557 | 0 | 0 | 0.770 | 0.333 | 0.558 |
| E_a1 | 0.639 | 0.639 | 0.546 | 58 | 55 | 0.761 | 0.454 | 0.411 |
| E_a2 | 0.615 | 0.615 | 0.546 | 43 | 62 | 0.729 | 0.565 | 0.258 |
| E_a3 | 0.584 | 0.584 | 0.543 | 47 | 94 | 0.675 | 0.657 | 0.061 |
| gate_disagree_a5 | 0.611 | 0.611 | 0.559 | 43 | 66 | 0.745 | 0.399 | 0.416 |
| gate_disagree_soft_a5 | 0.592 | 0.592 | 0.554 | 27 | 67 | 0.736 | 0.371 | 0.471 |
| C_abst_a1 | 0.634 | 0.634 | 0.549 | 3 | 5 | 0.768 | 0.214 | 0.712 |
| transfer_global_g2 | 0.690 | 0.690 | 0.623 | 95 | 46 | 0.810 | 0.441 | 0.486 |
| transfer_disagree_g2 | 0.716 | 0.716 | 0.605 | 99 | 27 | 0.815 | 0.365 | 0.562 |
| prompt_only_cautious | 0.582 | 0.582 | 0.434 | 72 | 121 | 0.741 | 0.115 | 0.750 |

# E14 bbq_mc question-level accuracy with the statement readout: qwen3-1.7b (904 questions, 2712 candidates; estimation questions excluded; E layers [15, 16, 17], probe L16, transfer write L19)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.570 | 0.570 | 0.344 | 0 | 0 | 0.635 | 0.320 | 0.054 |
| E_a1 | 0.732 | 0.732 | 0.365 | 170 | 23 | 0.720 | 0.248 | 0.225 |
| E_a2 | 0.802 | 0.802 | 0.491 | 251 | 41 | 0.791 | 0.400 | 0.329 |
| E_a3 | 0.741 | 0.741 | 0.614 | 324 | 169 | 0.799 | 0.522 | 0.338 |
| gate_disagree_a5 | 0.814 | 0.814 | 0.647 | 320 | 99 | 0.789 | 0.691 | 0.043 |
| gate_disagree_soft_a5 | 0.792 | 0.792 | 0.537 | 211 | 10 | 0.766 | 0.566 | 0.039 |
| C_abst_a1 | 0.561 | 0.561 | 0.383 | 3 | 11 | 0.634 | 0.165 | 0.678 |
| transfer_global_g2 | 0.600 | 0.600 | 0.344 | 40 | 13 | 0.640 | 0.276 | 0.120 |
| transfer_disagree_g2 | 0.551 | 0.551 | 0.345 | 8 | 25 | 0.620 | 0.323 | 0.055 |
| prompt_only_cautious | 0.502 | 0.502 | 0.339 | 20 | 81 | 0.590 | 0.267 | 0.142 |

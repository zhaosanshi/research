# E14 bbq_mc question-level accuracy with the statement readout: qwen3-0.6b (904 questions, 2712 candidates; estimation questions excluded; E layers [14, 15, 16], probe L15, transfer write L18)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.324 | 0.324 | 0.331 | 0 | 0 | 0.481 | 0.114 | 0.671 |
| E_a1 | 0.301 | 0.301 | 0.329 | 18 | 39 | 0.448 | 0.123 | 0.596 |
| E_a2 | 0.288 | 0.288 | 0.323 | 35 | 68 | 0.407 | 0.133 | 0.444 |
| E_a3 | 0.265 | 0.265 | 0.313 | 49 | 102 | 0.408 | 0.239 | 0.234 |
| gate_disagree_a5 | 0.383 | 0.383 | 0.338 | 141 | 88 | 0.516 | 0.121 | 0.532 |
| gate_disagree_soft_a5 | 0.317 | 0.317 | 0.332 | 36 | 42 | 0.412 | 0.119 | 0.483 |
| C_abst_a1 | 0.311 | 0.311 | 0.330 | 26 | 38 | 0.465 | 0.000 | 1.000 |
| transfer_global_g2 | 0.305 | 0.305 | 0.331 | 21 | 38 | 0.446 | 0.160 | 0.345 |
| transfer_disagree_g2 | 0.293 | 0.293 | 0.330 | 28 | 56 | 0.417 | 0.132 | 0.387 |
| prompt_only_cautious | 0.319 | 0.319 | 0.324 | 53 | 58 | 0.493 | 0.005 | 0.994 |

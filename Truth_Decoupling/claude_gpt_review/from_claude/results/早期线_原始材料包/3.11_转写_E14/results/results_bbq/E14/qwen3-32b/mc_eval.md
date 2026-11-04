# E14 bbq_mc question-level accuracy with the statement readout: qwen3-32b (904 questions, 2712 candidates; estimation questions excluded; E layers [32, 33, 34], probe L36, transfer write L36)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.790 | 0.790 | 0.617 | 0 | 0 | 0.911 | 0.483 | 0.308 |
| E_a1 | 0.781 | 0.781 | 0.606 | 2 | 10 | 0.909 | 0.480 | 0.305 |
| E_a2 | 0.826 | 0.826 | 0.608 | 47 | 14 | 0.917 | 0.442 | 0.425 |
| E_a3 | 0.732 | 0.732 | 0.542 | 71 | 123 | 0.809 | 0.440 | 0.428 |
| gate_disagree_a5 | 0.750 | 0.750 | 0.602 | 9 | 45 | 0.898 | 0.467 | 0.273 |
| gate_disagree_soft_a5 | 0.787 | 0.787 | 0.614 | 3 | 6 | 0.912 | 0.478 | 0.295 |
| C_abst_a1 | 0.792 | 0.792 | 0.612 | 8 | 6 | 0.909 | 0.390 | 0.417 |
| transfer_global_g2 | 0.790 | 0.790 | 0.620 | 3 | 3 | 0.911 | 0.478 | 0.313 |
| transfer_disagree_g2 | 0.792 | 0.792 | 0.618 | 3 | 1 | 0.912 | 0.481 | 0.310 |
| prompt_only_cautious | 0.798 | 0.798 | 0.541 | 35 | 28 | 0.898 | 0.407 | 0.369 |

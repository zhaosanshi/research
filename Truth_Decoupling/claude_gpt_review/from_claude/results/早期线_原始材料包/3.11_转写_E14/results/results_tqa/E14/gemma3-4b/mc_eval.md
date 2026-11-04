# E14 truthfulqa_mc question-level accuracy with the statement readout: gemma3-4b (721 questions, 5179 candidates; estimation questions excluded; E layers [14, 15, 16], probe L16, transfer write L18)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.646 | 0.778 | 0.695 | 0 | 0 | 0.759 | 0.678 | 0.023 |
| E_a1 | 0.680 | 0.798 | 0.727 | 47 | 23 | 0.801 | 0.710 | 0.011 |
| E_a2 | 0.703 | 0.809 | 0.739 | 81 | 40 | 0.798 | 0.717 | 0.005 |
| E_a3 | 0.646 | 0.793 | 0.716 | 89 | 89 | 0.763 | 0.694 | 0.003 |
| gate_disagree_a5 | 0.602 | 0.713 | 0.736 | 51 | 83 | 0.764 | 0.723 | 0.024 |
| gate_disagree_soft_a5 | 0.687 | 0.802 | 0.739 | 40 | 11 | 0.780 | 0.724 | 0.015 |
| C_abst_a1 | 0.653 | 0.761 | 0.720 | 33 | 28 | 0.750 | 0.100 | 0.868 |
| transfer_global_g2 | 0.277 | 0.401 | 0.418 | 46 | 312 | 0.476 | 0.541 | 0.021 |
| transfer_disagree_g2 | 0.652 | 0.803 | 0.772 | 54 | 50 | 0.771 | 0.734 | 0.017 |
| prompt_only_cautious | 0.635 | 0.775 | 0.719 | 62 | 70 | 0.737 | 0.341 | 0.466 |

# E14 truthfulqa_mc question-level accuracy with the statement readout: llama3.1-8b (721 questions, 5179 candidates; estimation questions excluded; E layers [12, 13, 14], probe L14, transfer write L16)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.649 | 0.759 | 0.707 | 0 | 0 | 0.775 | 0.679 | 0.016 |
| transfer_global_g2 | 0.779 | 0.895 | 0.799 | 113 | 19 | 0.878 | 0.770 | 0.014 |
| transfer_disagree_g2 | 0.738 | 0.853 | 0.770 | 82 | 18 | 0.858 | 0.769 | 0.016 |

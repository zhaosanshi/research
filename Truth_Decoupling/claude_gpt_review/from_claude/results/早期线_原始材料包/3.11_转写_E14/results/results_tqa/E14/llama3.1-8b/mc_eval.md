# E14 TruthfulQA question-level accuracy with the statement readout: llama3.1-8b (721 questions, 5179 candidates; estimation questions excluded)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.649 | 0.759 | 0.707 | 0 | 0 | 0.775 | 0.679 | 0.016 |
| E_a1 | 0.699 | 0.792 | 0.742 | 47 | 11 | 0.813 | 0.721 | 0.011 |
| E_a2 | 0.742 | 0.836 | 0.771 | 89 | 22 | 0.839 | 0.751 | 0.005 |
| E_a3 | 0.749 | 0.843 | 0.786 | 105 | 33 | 0.846 | 0.761 | 0.002 |
| gate_disagree_a5 | 0.667 | 0.777 | 0.760 | 90 | 77 | 0.768 | 0.786 | 0.009 |
| gate_disagree_soft_a5 | 0.695 | 0.811 | 0.741 | 58 | 25 | 0.804 | 0.752 | 0.012 |
| C_abst_a1 | 0.648 | 0.760 | 0.693 | 4 | 5 | 0.774 | 0.501 | 0.290 |
| prompt_only_cautious | 0.584 | 0.710 | 0.669 | 16 | 63 | 0.734 | 0.437 | 0.313 |

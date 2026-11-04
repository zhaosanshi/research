# E14 truthfulqa_mc question-level accuracy with the statement readout: qwen3-0.6b (721 questions, 5179 candidates; estimation questions excluded; E layers [14, 15, 16], probe L15, transfer write L18)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) | candidate abstain rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.272 | 0.462 | 0.451 | 0 | 0 | 0.466 | 0.356 | 0.182 |
| E_a1 | 0.341 | 0.528 | 0.452 | 55 | 5 | 0.530 | 0.365 | 0.172 |
| E_a2 | 0.462 | 0.599 | 0.459 | 157 | 20 | 0.616 | 0.394 | 0.131 |
| E_a3 | 0.472 | 0.655 | 0.481 | 204 | 60 | 0.617 | 0.419 | 0.140 |
| gate_disagree_a5 | 0.422 | 0.594 | 0.535 | 187 | 79 | 0.634 | 0.399 | 0.303 |
| gate_disagree_soft_a5 | 0.344 | 0.498 | 0.487 | 84 | 32 | 0.564 | 0.389 | 0.215 |
| C_abst_a1 | 0.277 | 0.472 | 0.449 | 12 | 8 | 0.461 | 0.001 | 0.997 |
| transfer_global_g2 | 0.269 | 0.454 | 0.450 | 10 | 12 | 0.456 | 0.369 | 0.149 |
| transfer_disagree_g2 | 0.265 | 0.449 | 0.450 | 7 | 12 | 0.455 | 0.364 | 0.151 |
| prompt_only_cautious | 0.254 | 0.440 | 0.448 | 39 | 52 | 0.437 | 0.000 | 1.000 |

# E16 ITI baseline, same protocol: llama3.1-8b (n=1442; K=48 heads; 926 estimation statements incl. test half A; chosen heads' layers [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28]; val acc 0.826-0.854)

## All items (half A used for training: NOT a clean report)

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.665 | 0.324 | 0.011 | 0 | 0 | 0.0 | +0.341 | +0.000 | 0.345 | 0.391 | 0.989 |
| iti_k48_a5 | 0.663 | 0.316 | 0.021 | 192 | 178 | 18.6 | +0.347 | -0.020 | 0.351 | 0.665 | 0.979 |
| iti_k48_a10 | 0.584 | 0.413 | 0.003 | 299 | 420 | 43.8 | +0.171 | -0.033 | 0.169 | 0.898 | 0.997 |
| iti_k48_a15 | 0.500 | 0.500 | 0.000 | 312 | 557 | 58.1 | +0.000 | -0.017 | 0.000 | 1.000 | 1.000 |
| iti_k48_a20 | 0.500 | 0.500 | 0.000 | 312 | 557 | 58.1 | +0.000 | -0.025 | 0.000 | 1.000 | 1.000 |

## Half B only (strength chosen on half A by penalised score: iti_k48_a5; B penalised +0.404 vs baseline +0.369)

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.679 | 0.311 | 0.010 | 0 | 0 | 0.0 | +0.369 | +0.000 | 0.373 | 0.387 | 0.990 |
| iti_k48_a5 | 0.694 | 0.290 | 0.017 | 98 | 83 | 17.3 | +0.404 | -0.017 | 0.408 | 0.658 | 0.983 |
| iti_k48_a10 | 0.597 | 0.403 | 0.000 | 144 | 205 | 42.6 | +0.195 | -0.028 | 0.195 | 0.886 | 1.000 |
| iti_k48_a15 | 0.500 | 0.500 | 0.000 | 150 | 280 | 58.2 | +0.000 | -0.016 | 0.000 | 1.000 | 1.000 |
| iti_k48_a20 | 0.500 | 0.500 | 0.000 | 150 | 280 | 58.2 | +0.000 | -0.026 | 0.000 | 1.000 | 1.000 |

## truthfulqa_mc question level (354 questions; half-A questions excluded)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) |
|---|---|---|---|---|---|---|---|
| baseline | 0.664 | 0.791 | 0.732 | 0 | 0 | 0.790 | 0.692 |
| iti_k48_a5 | 0.675 | 0.799 | 0.622 | 11 | 7 | 0.773 | 0.673 |
| iti_k48_a10 | 0.655 | 0.785 | 0.523 | 12 | 15 | 0.759 | 0.575 |
| iti_k48_a15 | 0.655 | 0.797 | 0.466 | 13 | 16 | 0.771 | 0.449 |
| iti_k48_a20 | 0.653 | 0.774 | 0.457 | 13 | 17 | 0.768 | 0.449 |

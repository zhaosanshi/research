# E16 ITI baseline, same protocol: qwen3-8b (n=1442; K=48 heads; 192 estimation statements; chosen heads' layers [19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 34, 35]; val acc 0.812-0.849)

## All items

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.503 | 0.155 | 0.341 | 0 | 0 | 0.0 | +0.348 | +0.000 | 0.529 | 0.519 | 0.659 |
| iti_k48_a5 | 0.463 | 0.137 | 0.400 | 4 | 4 | 0.6 | +0.325 | -0.005 | 0.541 | 0.570 | 0.600 |
| iti_k48_a10 | 0.343 | 0.099 | 0.558 | 1 | 3 | 0.4 | +0.243 | -0.011 | 0.555 | 0.535 | 0.442 |
| iti_k48_a15 | 0.074 | 0.008 | 0.919 | 0 | 0 | 0.0 | +0.066 | -0.012 | 0.221 | 0.034 | 0.081 |
| iti_k48_a20 | 0.004 | 0.000 | 0.996 | 0 | 0 | 0.0 | +0.004 | -0.011 | nan | 0.000 | 0.004 |

## Half B only (strength chosen on half A by penalised score: iti_k48_a5; B penalised +0.322 vs baseline +0.359)

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.517 | 0.158 | 0.325 | 0 | 0 | 0.0 | +0.359 | +0.000 | 0.532 | 0.527 | 0.675 |
| iti_k48_a5 | 0.466 | 0.144 | 0.390 | 0 | 4 | 1.1 | +0.322 | -0.006 | 0.526 | 0.576 | 0.610 |
| iti_k48_a10 | 0.349 | 0.105 | 0.547 | 1 | 1 | 0.3 | +0.244 | -0.014 | 0.541 | 0.536 | 0.453 |
| iti_k48_a15 | 0.085 | 0.006 | 0.910 | 0 | 0 | 0.0 | +0.079 | -0.015 | 0.333 | 0.031 | 0.090 |
| iti_k48_a20 | 0.008 | 0.000 | 0.992 | 0 | 0 | 0.0 | +0.008 | -0.014 | nan | 0.000 | 0.008 |

## truthfulqa_mc question level (721 questions)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) |
|---|---|---|---|---|---|---|---|
| baseline | 0.700 | 0.788 | 0.754 | 0 | 0 | 0.811 | 0.498 |
| iti_k48_a5 | 0.703 | 0.793 | 0.746 | 8 | 6 | 0.808 | 0.449 |
| iti_k48_a10 | 0.691 | 0.788 | 0.740 | 6 | 13 | 0.804 | 0.331 |
| iti_k48_a15 | 0.692 | 0.798 | 0.743 | 9 | 15 | 0.801 | 0.085 |
| iti_k48_a20 | 0.699 | 0.798 | 0.752 | 22 | 23 | 0.800 | 0.005 |

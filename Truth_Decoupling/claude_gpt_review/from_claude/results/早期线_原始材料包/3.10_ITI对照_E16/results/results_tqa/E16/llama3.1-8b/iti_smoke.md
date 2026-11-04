# E16 ITI baseline, same protocol: llama3.1-8b (n=60; K=48 heads; 192 estimation statements; chosen heads' layers [12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28, 30]; val acc 0.812-0.870)

## All items

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.683 | 0.317 | 0.000 | 0 | 0 | 0.0 | +0.367 | +0.000 | 0.367 | 0.417 | 1.000 |
| iti_k48_a10 | 0.567 | 0.350 | 0.083 | 8 | 10 | 24.4 | +0.217 | -0.071 | 0.187 | 0.782 | 0.917 |
| iti_k48_a15 | 0.550 | 0.367 | 0.083 | 10 | 13 | 31.7 | +0.183 | -0.072 | 0.133 | 0.873 | 0.917 |

## Half B only (strength chosen on half A by penalised score: iti_k48_a10; B penalised +0.222 vs baseline +0.222)

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.611 | 0.389 | 0.000 | 0 | 0 | 0.0 | +0.222 | +0.000 | 0.222 | 0.444 | 1.000 |
| iti_k48_a10 | 0.556 | 0.333 | 0.111 | 6 | 4 | 18.2 | +0.222 | -0.068 | 0.175 | 0.812 | 0.889 |
| iti_k48_a15 | 0.583 | 0.306 | 0.111 | 7 | 4 | 18.2 | +0.278 | -0.096 | 0.230 | 0.844 | 0.889 |

## truthfulqa_mc question level (12 questions)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) |
|---|---|---|---|---|---|---|---|
| baseline | 0.750 | 0.833 | 0.664 | 0 | 0 | 0.741 | 0.694 |
| iti_k48_a10 | 0.667 | 0.750 | 0.525 | 0 | 1 | 0.690 | 0.514 |
| iti_k48_a15 | 0.500 | 0.667 | 0.511 | 0 | 3 | 0.679 | 0.500 |

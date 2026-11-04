# E16 ITI baseline, same protocol: llama3.1-8b (n=1808; K=48 heads; 192 estimation statements; chosen heads' layers [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 31]; val acc 0.896-0.922)

## All items

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.539 | 0.284 | 0.178 | 0 | 0 | 0.0 | +0.255 | +0.000 | 0.247 | 0.213 | 0.822 |
| iti_k48_a5 | 0.486 | 0.178 | 0.336 | 74 | 47 | 4.8 | +0.309 | -0.015 | 0.425 | 0.357 | 0.664 |
| iti_k48_a10 | 0.355 | 0.072 | 0.574 | 44 | 15 | 1.5 | +0.283 | -0.001 | 0.641 | 0.383 | 0.426 |
| iti_k48_a15 | 0.011 | 0.000 | 0.989 | 0 | 0 | 0.0 | +0.011 | -0.012 | 1.000 | 0.684 | 0.011 |
| iti_k48_a20 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0.0 | +0.000 | -0.070 | nan | nan | 0.000 |

## Half B only (strength chosen on half A by penalised score: iti_k48_a5; B penalised +0.332 vs baseline +0.262)

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.554 | 0.291 | 0.155 | 0 | 0 | 0.0 | +0.262 | +0.000 | 0.252 | 0.202 | 0.845 |
| iti_k48_a5 | 0.505 | 0.173 | 0.322 | 44 | 25 | 4.7 | +0.332 | -0.013 | 0.449 | 0.350 | 0.678 |
| iti_k48_a10 | 0.378 | 0.071 | 0.552 | 26 | 9 | 1.7 | +0.307 | -0.001 | 0.658 | 0.366 | 0.448 |
| iti_k48_a15 | 0.016 | 0.000 | 0.984 | 0 | 0 | 0.0 | +0.016 | -0.012 | 1.000 | 0.733 | 0.016 |
| iti_k48_a20 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0.0 | +0.000 | -0.065 | nan | nan | 0.000 |

## bbq_mc question level (904 questions)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) |
|---|---|---|---|---|---|---|---|
| baseline | 0.561 | 0.561 | 0.503 | 0 | 0 | 0.691 | 0.593 |
| iti_k48_a5 | 0.545 | 0.545 | 0.485 | 24 | 38 | 0.687 | 0.440 |
| iti_k48_a10 | 0.562 | 0.562 | 0.474 | 37 | 36 | 0.701 | 0.259 |
| iti_k48_a15 | 0.552 | 0.552 | 0.432 | 40 | 48 | 0.694 | 0.007 |
| iti_k48_a20 | 0.528 | 0.528 | 0.363 | 54 | 84 | 0.649 | 0.000 |

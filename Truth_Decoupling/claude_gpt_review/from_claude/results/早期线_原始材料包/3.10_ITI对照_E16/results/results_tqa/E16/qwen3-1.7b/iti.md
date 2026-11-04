# E16 ITI baseline, same protocol: qwen3-1.7b (n=1442; K=48 heads; 192 estimation statements; chosen heads' layers [14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]; val acc 0.771-0.823)

## All items

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.506 | 0.309 | 0.185 | 0 | 0 | 0.0 | +0.196 | +0.000 | 0.199 | 0.839 | 0.815 |
| iti_k48_a5 | 0.479 | 0.483 | 0.038 | 42 | 147 | 20.2 | -0.003 | -0.085 | 0.000 | 1.000 | 0.962 |
| iti_k48_a10 | 0.368 | 0.404 | 0.229 | 42 | 137 | 18.8 | -0.036 | -0.152 | 0.000 | 1.000 | 0.771 |
| iti_k48_a15 | 0.021 | 0.044 | 0.935 | 4 | 16 | 2.2 | -0.024 | -0.126 | 0.000 | 1.000 | 0.065 |
| iti_k48_a20 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0.0 | +0.000 | -0.072 | nan | nan | 0.000 |

## Half B only (strength chosen on half A by penalised score: iti_k48_a20; B penalised +0.000 vs baseline +0.209)

| condition | correct | wrong | abstain | fixed | damaged | known dmg % | penalised | dAUROC | J | B | answer rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.517 | 0.308 | 0.175 | 0 | 0 | 0.0 | +0.209 | +0.000 | 0.214 | 0.841 | 0.825 |
| iti_k48_a5 | 0.483 | 0.487 | 0.030 | 18 | 75 | 20.5 | -0.004 | -0.084 | 0.000 | 1.000 | 0.970 |
| iti_k48_a10 | 0.376 | 0.400 | 0.225 | 18 | 69 | 18.9 | -0.024 | -0.143 | 0.000 | 1.000 | 0.775 |
| iti_k48_a15 | 0.027 | 0.051 | 0.922 | 4 | 8 | 2.2 | -0.024 | -0.116 | 0.000 | 1.000 | 0.078 |
| iti_k48_a20 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0.0 | +0.000 | -0.064 | nan | nan | 0.000 |

## truthfulqa_mc question level (721 questions)

| condition | MC1 | argmax acc | MC2 | MC1 fixed | MC1 damaged | statement AUROC | statement acc (all) |
|---|---|---|---|---|---|---|---|
| baseline | 0.470 | 0.617 | 0.550 | 0 | 0 | 0.652 | 0.462 |
| iti_k48_a5 | 0.387 | 0.555 | 0.452 | 16 | 76 | 0.575 | 0.427 |
| iti_k48_a10 | 0.341 | 0.485 | 0.451 | 26 | 119 | 0.503 | 0.324 |
| iti_k48_a15 | 0.361 | 0.505 | 0.451 | 30 | 109 | 0.528 | 0.014 |
| iti_k48_a20 | 0.401 | 0.562 | 0.451 | 31 | 81 | 0.583 | 0.000 |

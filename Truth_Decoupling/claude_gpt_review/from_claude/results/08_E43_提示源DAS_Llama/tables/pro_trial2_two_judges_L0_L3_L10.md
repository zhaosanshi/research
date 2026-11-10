## per condition (192 questions). AllenAI: answered-right / answered-wrong / true-but-uninformative / rule refusal. Pro: right / wrong / hedge / refusal (its own label)
| condition | AllenAI | Pro |
|---|---|---|
| baseline_x | 78 / 88 / 3 / 23 | 64 / 98 / 6 / 24 |
| baseline_x_E | 89 / 70 / 2 / 31 | 78 / 79 / 4 / 31 |
| baseline_x_conf | 88 / 101 / 1 / 2 | 70 / 112 / 9 / 1 |
| baseline_x_caut | 82 / 82 / 1 / 27 | 64 / 94 / 6 / 28 |
| E_L0_main | 129 / 57 / 3 / 3 | 97 / 87 / 5 / 3 |
| E_L3_main | 111 / 61 / 2 / 18 | 85 / 80 / 7 / 20 |
| E_L10_main | 115 / 73 / 0 / 4 | 99 / 84 / 7 / 2 |

## flips by how the text changed. cell = n / wrong→right / right→wrong
| comparison | judge | same text | changed, not longer | changed and longer | all: wrong→right / right→wrong |
|---|---|---|---|---|---|
| baseline_x_E vs baseline_x | AllenAI | 93 / 0 / 0 | 69 / 12 / 3 | 30 / 4 / 1 | 16 / 4 |
| baseline_x_E vs baseline_x | Pro | 93 / 0 / 0 | 69 / 11 / 2 | 30 / 6 / 1 | 17 / 3 |
| baseline_x_caut vs baseline_x_conf | AllenAI | 90 / 0 / 0 | 78 / 9 / 3 | 24 / 5 / 2 | 14 / 5 |
| baseline_x_caut vs baseline_x_conf | Pro | 90 / 0 / 0 | 78 / 5 / 3 | 24 / 3 / 1 | 8 / 4 |
| E_L0_main vs baseline_x | AllenAI | 7 / 0 / 0 | 33 / 11 / 1 | 152 / 45 / 22 | 56 / 23 |
| E_L0_main vs baseline_x | Pro | 7 / 0 / 0 | 33 / 9 / 2 | 152 / 36 / 25 | 45 / 27 |
| E_L3_main vs baseline_x | AllenAI | 24 / 0 / 0 | 48 / 7 / 4 | 120 / 25 / 3 | 32 / 7 |
| E_L3_main vs baseline_x | Pro | 24 / 0 / 0 | 48 / 3 / 3 | 120 / 21 / 7 | 24 / 10 |
| E_L10_main vs baseline_x | AllenAI | 8 / 1 / 0 | 28 / 4 / 5 | 156 / 31 / 12 | 36 / 17 |
| E_L10_main vs baseline_x | Pro | 8 / 0 / 0 | 28 / 3 / 3 | 156 / 28 / 11 | 31 / 14 |

## agreement on right / not right by answer length (all conditions, refusals excluded)
| answer length | cells | AllenAI right | Pro right | both | AllenAI only | Pro only |
|---|---|---|---|---|---|---|
| 1-3 words | 461 | 216 (47%) | 182 (39%) | 168 | 48 | 14 |
| 4-7 words | 397 | 231 (58%) | 187 (47%) | 169 | 62 | 18 |
| 8+ words | 373 | 240 (64%) | 187 (50%) | 180 | 60 | 7 |

rule refusals over all conditions: AllenAI labels {'N': 53, 'A': 55} | Pro labels {'C': 104, 'A': 1, 'B': 2, 'H': 1}

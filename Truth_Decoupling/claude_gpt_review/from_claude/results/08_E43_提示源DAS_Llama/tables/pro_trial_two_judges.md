## per condition (192 questions). AllenAI: answered-right / answered-wrong / true-but-uninformative / rule refusal. Pro: right / wrong / hedge / refusal (its own label)
| condition | AllenAI | Pro |
|---|---|---|
| baseline_x | 78 / 88 / 3 / 23 | 64 / 98 / 6 / 24 |
| baseline_x_E | 89 / 70 / 2 / 31 | 78 / 79 / 4 / 31 |
| baseline_x_conf | 88 / 101 / 1 / 2 | 70 / 112 / 9 / 1 |
| baseline_x_caut | 82 / 82 / 1 / 27 | 64 / 94 / 6 / 28 |
| E_L18_main | 95 / 73 / 2 / 22 | 71 / 91 / 8 / 22 |

## flips by how the text changed. cell = n / wrong→right / right→wrong
| comparison | judge | same text | changed, not longer | changed and longer | all: wrong→right / right→wrong |
|---|---|---|---|---|---|
| baseline_x_E vs baseline_x | AllenAI | 93 / 0 / 0 | 69 / 12 / 3 | 30 / 4 / 1 | 16 / 4 |
| baseline_x_E vs baseline_x | Pro | 93 / 0 / 0 | 69 / 11 / 2 | 30 / 6 / 1 | 17 / 3 |
| baseline_x_caut vs baseline_x_conf | AllenAI | 90 / 0 / 0 | 78 / 9 / 3 | 24 / 5 / 2 | 14 / 5 |
| baseline_x_caut vs baseline_x_conf | Pro | 90 / 0 / 0 | 78 / 5 / 3 | 24 / 3 / 1 | 8 / 4 |
| E_L18_main vs baseline_x | AllenAI | 49 / 1 / 0 | 67 / 8 / 5 | 76 / 17 / 6 | 26 / 11 |
| E_L18_main vs baseline_x | Pro | 49 / 0 / 0 | 67 / 7 / 7 | 76 / 12 / 7 | 19 / 14 |

## agreement on right / not right by answer length (all conditions, refusals excluded)
| answer length | cells | AllenAI right | Pro right | both | AllenAI only | Pro only |
|---|---|---|---|---|---|---|
| 1-3 words | 460 | 221 (48%) | 182 (40%) | 167 | 54 | 15 |
| 4-7 words | 314 | 176 (56%) | 137 (44%) | 120 | 56 | 17 |
| 8+ words | 79 | 33 (42%) | 28 (35%) | 24 | 9 | 4 |

rule refusals over all conditions: AllenAI labels {'N': 55, 'A': 50} | Pro labels {'C': 104, 'H': 1}

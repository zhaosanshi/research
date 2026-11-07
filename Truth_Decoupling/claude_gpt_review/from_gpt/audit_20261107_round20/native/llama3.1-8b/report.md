# 固定 E 的原生 C 门控诊断

仅调参集；不能代替 E26 主表或新增测试证据。

门控首词 Yes−No > 0 且候选实际非拒答才交付；对照在同一批题交付完全相同数量。

## tf

| 方法 | 实际回答 | 答对 | 已回答准确率 | 原模型简单筛题答对 | 固定E简单筛题答对 |
|---|---:|---:|---:|---:|---:|
| baseline | 128/384 | 92 | 0.7188 | 101 | {"das_E_simple_threshold": 104, "md_E_simple_threshold": 105} |
| das_E | 172/384 | 130 | 0.7558 | 127 | {"das_E_simple_threshold": 140, "md_E_simple_threshold": 134} |
| das_C_a-2_gate_and_answer | 0/384 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_C_a-2_gate_only | 0/384 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_C_a-1_gate_and_answer | 45/384 | 25 | 0.5556 | 40 | {"das_E_simple_threshold": 39, "md_E_simple_threshold": 39} |
| das_C_a-1_gate_only | 45/384 | 25 | 0.5556 | 40 | {"das_E_simple_threshold": 39, "md_E_simple_threshold": 39} |
| das_C_a-0.5_gate_and_answer | 90/384 | 59 | 0.6556 | 75 | {"das_E_simple_threshold": 76, "md_E_simple_threshold": 76} |
| das_C_a-0.5_gate_only | 90/384 | 59 | 0.6556 | 75 | {"das_E_simple_threshold": 76, "md_E_simple_threshold": 76} |
| das_C_a0_gate_and_answer | 128/384 | 92 | 0.7188 | 101 | {"das_E_simple_threshold": 104, "md_E_simple_threshold": 105} |
| das_C_a0_gate_only | 128/384 | 92 | 0.7188 | 101 | {"das_E_simple_threshold": 104, "md_E_simple_threshold": 105} |
| das_C_a0.5_gate_and_answer | 180/384 | 127 | 0.7056 | 132 | {"das_E_simple_threshold": 145, "md_E_simple_threshold": 139} |
| das_C_a0.5_gate_only | 180/384 | 126 | 0.7000 | 132 | {"das_E_simple_threshold": 145, "md_E_simple_threshold": 139} |
| das_C_a1_gate_and_answer | 217/384 | 141 | 0.6498 | 158 | {"das_E_simple_threshold": 171, "md_E_simple_threshold": 168} |
| das_C_a1_gate_only | 217/384 | 141 | 0.6498 | 158 | {"das_E_simple_threshold": 171, "md_E_simple_threshold": 168} |
| das_C_a2_gate_and_answer | 308/384 | 192 | 0.6234 | 224 | {"das_E_simple_threshold": 231, "md_E_simple_threshold": 229} |
| das_C_a2_gate_only | 308/384 | 197 | 0.6396 | 224 | {"das_E_simple_threshold": 231, "md_E_simple_threshold": 229} |
| das_EC_a-2_gate_and_answer | 9/384 | 7 | 0.7778 | 9 | {"das_E_simple_threshold": 9, "md_E_simple_threshold": 9} |
| das_EC_a-2_gate_only | 9/384 | 7 | 0.7778 | 9 | {"das_E_simple_threshold": 9, "md_E_simple_threshold": 9} |
| das_EC_a-1_gate_and_answer | 121/384 | 88 | 0.7273 | 96 | {"das_E_simple_threshold": 97, "md_E_simple_threshold": 100} |
| das_EC_a-1_gate_only | 121/384 | 88 | 0.7273 | 96 | {"das_E_simple_threshold": 97, "md_E_simple_threshold": 100} |
| das_EC_a-0.5_gate_and_answer | 149/384 | 112 | 0.7517 | 116 | {"das_E_simple_threshold": 122, "md_E_simple_threshold": 118} |
| das_EC_a-0.5_gate_only | 149/384 | 112 | 0.7517 | 116 | {"das_E_simple_threshold": 122, "md_E_simple_threshold": 118} |
| das_EC_a0_gate_and_answer | 172/384 | 130 | 0.7558 | 127 | {"das_E_simple_threshold": 140, "md_E_simple_threshold": 134} |
| das_EC_a0_gate_only | 172/384 | 130 | 0.7558 | 127 | {"das_E_simple_threshold": 140, "md_E_simple_threshold": 134} |
| das_EC_a0.5_gate_and_answer | 192/384 | 145 | 0.7552 | 140 | {"das_E_simple_threshold": 152, "md_E_simple_threshold": 148} |
| das_EC_a0.5_gate_only | 192/384 | 144 | 0.7500 | 140 | {"das_E_simple_threshold": 152, "md_E_simple_threshold": 148} |
| das_EC_a1_gate_and_answer | 207/384 | 149 | 0.7198 | 151 | {"das_E_simple_threshold": 161, "md_E_simple_threshold": 160} |
| das_EC_a1_gate_only | 207/384 | 149 | 0.7198 | 151 | {"das_E_simple_threshold": 161, "md_E_simple_threshold": 160} |
| das_EC_a2_gate_and_answer | 259/384 | 176 | 0.6795 | 191 | {"das_E_simple_threshold": 200, "md_E_simple_threshold": 198} |
| das_EC_a2_gate_only | 259/384 | 177 | 0.6834 | 191 | {"das_E_simple_threshold": 200, "md_E_simple_threshold": 198} |
| md_E | 163/384 | 122 | 0.7485 | 122 | {"das_E_simple_threshold": 133, "md_E_simple_threshold": 130} |
| md_C_a-2_gate_and_answer | 0/384 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_C_a-2_gate_only | 0/384 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_C_a-1_gate_and_answer | 52/384 | 32 | 0.6154 | 44 | {"das_E_simple_threshold": 45, "md_E_simple_threshold": 44} |
| md_C_a-1_gate_only | 52/384 | 32 | 0.6154 | 44 | {"das_E_simple_threshold": 45, "md_E_simple_threshold": 44} |
| md_C_a-0.5_gate_and_answer | 93/384 | 61 | 0.6559 | 77 | {"das_E_simple_threshold": 78, "md_E_simple_threshold": 79} |
| md_C_a-0.5_gate_only | 93/384 | 61 | 0.6559 | 77 | {"das_E_simple_threshold": 78, "md_E_simple_threshold": 79} |
| md_C_a0_gate_and_answer | 128/384 | 92 | 0.7188 | 101 | {"das_E_simple_threshold": 104, "md_E_simple_threshold": 105} |
| md_C_a0_gate_only | 128/384 | 92 | 0.7188 | 101 | {"das_E_simple_threshold": 104, "md_E_simple_threshold": 105} |
| md_C_a0.5_gate_and_answer | 176/384 | 124 | 0.7045 | 129 | {"das_E_simple_threshold": 141, "md_E_simple_threshold": 135} |
| md_C_a0.5_gate_only | 176/384 | 123 | 0.6989 | 129 | {"das_E_simple_threshold": 141, "md_E_simple_threshold": 135} |
| md_C_a1_gate_and_answer | 215/384 | 139 | 0.6465 | 156 | {"das_E_simple_threshold": 169, "md_E_simple_threshold": 166} |
| md_C_a1_gate_only | 215/384 | 139 | 0.6465 | 156 | {"das_E_simple_threshold": 169, "md_E_simple_threshold": 166} |
| md_C_a2_gate_and_answer | 300/384 | 195 | 0.6500 | 219 | {"das_E_simple_threshold": 226, "md_E_simple_threshold": 224} |
| md_C_a2_gate_only | 300/384 | 190 | 0.6333 | 219 | {"das_E_simple_threshold": 226, "md_E_simple_threshold": 224} |
| md_EC_a-2_gate_and_answer | 10/384 | 7 | 0.7000 | 10 | {"das_E_simple_threshold": 10, "md_E_simple_threshold": 10} |
| md_EC_a-2_gate_only | 10/384 | 7 | 0.7000 | 10 | {"das_E_simple_threshold": 10, "md_E_simple_threshold": 10} |
| md_EC_a-1_gate_and_answer | 105/384 | 74 | 0.7048 | 85 | {"das_E_simple_threshold": 87, "md_E_simple_threshold": 86} |
| md_EC_a-1_gate_only | 105/384 | 74 | 0.7048 | 85 | {"das_E_simple_threshold": 87, "md_E_simple_threshold": 86} |
| md_EC_a-0.5_gate_and_answer | 137/384 | 103 | 0.7518 | 106 | {"das_E_simple_threshold": 110, "md_E_simple_threshold": 112} |
| md_EC_a-0.5_gate_only | 137/384 | 103 | 0.7518 | 106 | {"das_E_simple_threshold": 110, "md_E_simple_threshold": 112} |
| md_EC_a0_gate_and_answer | 163/384 | 122 | 0.7485 | 122 | {"das_E_simple_threshold": 133, "md_E_simple_threshold": 130} |
| md_EC_a0_gate_only | 163/384 | 122 | 0.7485 | 122 | {"das_E_simple_threshold": 133, "md_E_simple_threshold": 130} |
| md_EC_a0.5_gate_and_answer | 189/384 | 138 | 0.7302 | 137 | {"das_E_simple_threshold": 150, "md_E_simple_threshold": 145} |
| md_EC_a0.5_gate_only | 189/384 | 139 | 0.7354 | 137 | {"das_E_simple_threshold": 150, "md_E_simple_threshold": 145} |
| md_EC_a1_gate_and_answer | 208/384 | 147 | 0.7067 | 151 | {"das_E_simple_threshold": 162, "md_E_simple_threshold": 161} |
| md_EC_a1_gate_only | 208/384 | 147 | 0.7067 | 151 | {"das_E_simple_threshold": 162, "md_E_simple_threshold": 161} |
| md_EC_a2_gate_and_answer | 262/384 | 177 | 0.6756 | 193 | {"das_E_simple_threshold": 200, "md_E_simple_threshold": 200} |
| md_EC_a2_gate_only | 262/384 | 176 | 0.6718 | 193 | {"das_E_simple_threshold": 200, "md_E_simple_threshold": 200} |

## qa

| 方法 | 实际回答 | 答对 | 已回答准确率 | 原模型简单筛题答对 | 固定E简单筛题答对 |
|---|---:|---:|---:|---:|---:|
| baseline | 129/192 | 40 | 0.3101 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| das_E | 127/192 | 44 | 0.3465 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| das_C_a-2_gate_and_answer | 2/192 | 1 | 0.5000 | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_C_a-2_gate_only | 2/192 | 1 | 0.5000 | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_C_a-1_gate_and_answer | 69/192 | 23 | 0.3333 | 35 | {"das_E_simple_threshold": 36, "md_E_simple_threshold": 36} |
| das_C_a-1_gate_only | 69/192 | 25 | 0.3623 | 35 | {"das_E_simple_threshold": 36, "md_E_simple_threshold": 36} |
| das_C_a-0.5_gate_and_answer | 110/192 | 35 | 0.3182 | 50 | {"das_E_simple_threshold": 53, "md_E_simple_threshold": 51} |
| das_C_a-0.5_gate_only | 110/192 | 36 | 0.3273 | 50 | {"das_E_simple_threshold": 53, "md_E_simple_threshold": 51} |
| das_C_a0_gate_and_answer | 129/192 | 40 | 0.3101 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| das_C_a0_gate_only | 129/192 | 40 | 0.3101 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| das_C_a0.5_gate_and_answer | 145/192 | 46 | 0.3172 | 57 | {"das_E_simple_threshold": 61, "md_E_simple_threshold": 60} |
| das_C_a0.5_gate_only | 145/192 | 46 | 0.3172 | 57 | {"das_E_simple_threshold": 61, "md_E_simple_threshold": 60} |
| das_C_a1_gate_and_answer | 155/192 | 49 | 0.3161 | 58 | {"das_E_simple_threshold": 64, "md_E_simple_threshold": 60} |
| das_C_a1_gate_only | 154/192 | 49 | 0.3182 | 58 | {"das_E_simple_threshold": 64, "md_E_simple_threshold": 60} |
| das_C_a2_gate_and_answer | 165/192 | 55 | 0.3333 | 59 | {"das_E_simple_threshold": 66, "md_E_simple_threshold": 62} |
| das_C_a2_gate_only | 164/192 | 57 | 0.3476 | 59 | {"das_E_simple_threshold": 65, "md_E_simple_threshold": 61} |
| das_EC_a-2_gate_and_answer | 2/192 | 1 | 0.5000 | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_EC_a-2_gate_only | 2/192 | 1 | 0.5000 | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_EC_a-1_gate_and_answer | 67/192 | 25 | 0.3731 | 35 | {"das_E_simple_threshold": 35, "md_E_simple_threshold": 36} |
| das_EC_a-1_gate_only | 67/192 | 26 | 0.3881 | 35 | {"das_E_simple_threshold": 35, "md_E_simple_threshold": 36} |
| das_EC_a-0.5_gate_and_answer | 107/192 | 40 | 0.3738 | 50 | {"das_E_simple_threshold": 52, "md_E_simple_threshold": 49} |
| das_EC_a-0.5_gate_only | 107/192 | 40 | 0.3738 | 50 | {"das_E_simple_threshold": 52, "md_E_simple_threshold": 49} |
| das_EC_a0_gate_and_answer | 127/192 | 44 | 0.3465 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| das_EC_a0_gate_only | 127/192 | 44 | 0.3465 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| das_EC_a0.5_gate_and_answer | 143/192 | 47 | 0.3287 | 56 | {"das_E_simple_threshold": 60, "md_E_simple_threshold": 59} |
| das_EC_a0.5_gate_only | 142/192 | 50 | 0.3521 | 56 | {"das_E_simple_threshold": 60, "md_E_simple_threshold": 58} |
| das_EC_a1_gate_and_answer | 155/192 | 52 | 0.3355 | 58 | {"das_E_simple_threshold": 64, "md_E_simple_threshold": 60} |
| das_EC_a1_gate_only | 152/192 | 55 | 0.3618 | 58 | {"das_E_simple_threshold": 64, "md_E_simple_threshold": 60} |
| das_EC_a2_gate_and_answer | 165/192 | 58 | 0.3515 | 59 | {"das_E_simple_threshold": 66, "md_E_simple_threshold": 62} |
| das_EC_a2_gate_only | 162/192 | 63 | 0.3889 | 58 | {"das_E_simple_threshold": 65, "md_E_simple_threshold": 61} |
| md_E | 130/192 | 44 | 0.3385 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| md_C_a-2_gate_and_answer | 2/192 | 1 | 0.5000 | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_C_a-2_gate_only | 2/192 | 1 | 0.5000 | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_C_a-1_gate_and_answer | 69/192 | 23 | 0.3333 | 35 | {"das_E_simple_threshold": 36, "md_E_simple_threshold": 36} |
| md_C_a-1_gate_only | 69/192 | 25 | 0.3623 | 35 | {"das_E_simple_threshold": 36, "md_E_simple_threshold": 36} |
| md_C_a-0.5_gate_and_answer | 110/192 | 36 | 0.3273 | 50 | {"das_E_simple_threshold": 53, "md_E_simple_threshold": 51} |
| md_C_a-0.5_gate_only | 110/192 | 36 | 0.3273 | 50 | {"das_E_simple_threshold": 53, "md_E_simple_threshold": 51} |
| md_C_a0_gate_and_answer | 129/192 | 40 | 0.3101 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| md_C_a0_gate_only | 129/192 | 40 | 0.3101 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| md_C_a0.5_gate_and_answer | 146/192 | 45 | 0.3082 | 58 | {"das_E_simple_threshold": 61, "md_E_simple_threshold": 60} |
| md_C_a0.5_gate_only | 146/192 | 46 | 0.3151 | 58 | {"das_E_simple_threshold": 61, "md_E_simple_threshold": 60} |
| md_C_a1_gate_and_answer | 155/192 | 48 | 0.3097 | 58 | {"das_E_simple_threshold": 64, "md_E_simple_threshold": 60} |
| md_C_a1_gate_only | 154/192 | 49 | 0.3182 | 58 | {"das_E_simple_threshold": 64, "md_E_simple_threshold": 60} |
| md_C_a2_gate_and_answer | 165/192 | 55 | 0.3333 | 59 | {"das_E_simple_threshold": 66, "md_E_simple_threshold": 62} |
| md_C_a2_gate_only | 164/192 | 57 | 0.3476 | 59 | {"das_E_simple_threshold": 65, "md_E_simple_threshold": 61} |
| md_EC_a-2_gate_and_answer | 2/192 | 1 | 0.5000 | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_EC_a-2_gate_only | 2/192 | 1 | 0.5000 | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_EC_a-1_gate_and_answer | 69/192 | 24 | 0.3478 | 35 | {"das_E_simple_threshold": 36, "md_E_simple_threshold": 36} |
| md_EC_a-1_gate_only | 69/192 | 26 | 0.3768 | 35 | {"das_E_simple_threshold": 36, "md_E_simple_threshold": 36} |
| md_EC_a-0.5_gate_and_answer | 110/192 | 39 | 0.3545 | 50 | {"das_E_simple_threshold": 53, "md_E_simple_threshold": 51} |
| md_EC_a-0.5_gate_only | 110/192 | 40 | 0.3636 | 50 | {"das_E_simple_threshold": 53, "md_E_simple_threshold": 51} |
| md_EC_a0_gate_and_answer | 130/192 | 44 | 0.3385 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| md_EC_a0_gate_only | 130/192 | 44 | 0.3385 | 54 | {"das_E_simple_threshold": 55, "md_E_simple_threshold": 55} |
| md_EC_a0.5_gate_and_answer | 146/192 | 48 | 0.3288 | 58 | {"das_E_simple_threshold": 61, "md_E_simple_threshold": 60} |
| md_EC_a0.5_gate_only | 146/192 | 49 | 0.3356 | 58 | {"das_E_simple_threshold": 61, "md_E_simple_threshold": 60} |
| md_EC_a1_gate_and_answer | 155/192 | 50 | 0.3226 | 58 | {"das_E_simple_threshold": 64, "md_E_simple_threshold": 60} |
| md_EC_a1_gate_only | 154/192 | 52 | 0.3377 | 58 | {"das_E_simple_threshold": 64, "md_E_simple_threshold": 60} |
| md_EC_a2_gate_and_answer | 165/192 | 56 | 0.3394 | 59 | {"das_E_simple_threshold": 66, "md_E_simple_threshold": 62} |
| md_EC_a2_gate_only | 164/192 | 60 | 0.3659 | 59 | {"das_E_simple_threshold": 65, "md_E_simple_threshold": 61} |

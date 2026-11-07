# 固定 E 的原生 C 门控诊断

仅调参集；不能代替 E26 主表或新增测试证据。

门控首词 Yes−No > 0 且候选实际非拒答才交付；对照在同一批题交付完全相同数量。

## tf

| 方法 | 实际回答 | 答对 | 已回答准确率 | 原模型简单筛题答对 | 固定E简单筛题答对 |
|---|---:|---:|---:|---:|---:|
| baseline | 152/384 | 114 | 0.7500 | 130 | {"das_E_simple_threshold": 138, "md_E_simple_threshold": 133} |
| das_E | 154/384 | 120 | 0.7792 | 131 | {"das_E_simple_threshold": 140, "md_E_simple_threshold": 135} |
| das_C_a-2_gate_and_answer | 0/384 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_C_a-2_gate_only | 0/384 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_C_a-1_gate_and_answer | 82/384 | 65 | 0.7927 | 72 | {"das_E_simple_threshold": 77, "md_E_simple_threshold": 71} |
| das_C_a-1_gate_only | 82/384 | 65 | 0.7927 | 72 | {"das_E_simple_threshold": 77, "md_E_simple_threshold": 71} |
| das_C_a-0.5_gate_and_answer | 118/384 | 94 | 0.7966 | 102 | {"das_E_simple_threshold": 109, "md_E_simple_threshold": 104} |
| das_C_a-0.5_gate_only | 118/384 | 94 | 0.7966 | 102 | {"das_E_simple_threshold": 109, "md_E_simple_threshold": 104} |
| das_C_a0_gate_and_answer | 152/384 | 114 | 0.7500 | 130 | {"das_E_simple_threshold": 138, "md_E_simple_threshold": 133} |
| das_C_a0_gate_only | 152/384 | 114 | 0.7500 | 130 | {"das_E_simple_threshold": 138, "md_E_simple_threshold": 133} |
| das_C_a0.5_gate_and_answer | 194/384 | 131 | 0.6753 | 163 | {"das_E_simple_threshold": 171, "md_E_simple_threshold": 165} |
| das_C_a0.5_gate_only | 194/384 | 131 | 0.6753 | 163 | {"das_E_simple_threshold": 171, "md_E_simple_threshold": 165} |
| das_C_a1_gate_and_answer | 250/384 | 161 | 0.6440 | 200 | {"das_E_simple_threshold": 210, "md_E_simple_threshold": 201} |
| das_C_a1_gate_only | 250/384 | 156 | 0.6240 | 200 | {"das_E_simple_threshold": 210, "md_E_simple_threshold": 201} |
| das_C_a2_gate_and_answer | 384/384 | 280 | 0.7292 | 265 | {"das_E_simple_threshold": 287, "md_E_simple_threshold": 276} |
| das_C_a2_gate_only | 384/384 | 265 | 0.6901 | 265 | {"das_E_simple_threshold": 287, "md_E_simple_threshold": 276} |
| das_EC_a-2_gate_and_answer | 4/384 | 3 | 0.7500 | 4 | {"das_E_simple_threshold": 4, "md_E_simple_threshold": 4} |
| das_EC_a-2_gate_only | 4/384 | 3 | 0.7500 | 4 | {"das_E_simple_threshold": 4, "md_E_simple_threshold": 4} |
| das_EC_a-1_gate_and_answer | 90/384 | 73 | 0.8111 | 79 | {"das_E_simple_threshold": 84, "md_E_simple_threshold": 77} |
| das_EC_a-1_gate_only | 90/384 | 73 | 0.8111 | 79 | {"das_E_simple_threshold": 84, "md_E_simple_threshold": 77} |
| das_EC_a-0.5_gate_and_answer | 118/384 | 95 | 0.8051 | 102 | {"das_E_simple_threshold": 109, "md_E_simple_threshold": 104} |
| das_EC_a-0.5_gate_only | 118/384 | 95 | 0.8051 | 102 | {"das_E_simple_threshold": 109, "md_E_simple_threshold": 104} |
| das_EC_a0_gate_and_answer | 154/384 | 120 | 0.7792 | 131 | {"das_E_simple_threshold": 140, "md_E_simple_threshold": 135} |
| das_EC_a0_gate_only | 154/384 | 120 | 0.7792 | 131 | {"das_E_simple_threshold": 140, "md_E_simple_threshold": 135} |
| das_EC_a0.5_gate_and_answer | 191/384 | 144 | 0.7539 | 161 | {"das_E_simple_threshold": 170, "md_E_simple_threshold": 163} |
| das_EC_a0.5_gate_only | 191/384 | 144 | 0.7539 | 161 | {"das_E_simple_threshold": 170, "md_E_simple_threshold": 163} |
| das_EC_a1_gate_and_answer | 233/384 | 165 | 0.7082 | 189 | {"das_E_simple_threshold": 197, "md_E_simple_threshold": 190} |
| das_EC_a1_gate_only | 233/384 | 165 | 0.7082 | 189 | {"das_E_simple_threshold": 197, "md_E_simple_threshold": 190} |
| das_EC_a2_gate_and_answer | 383/384 | 284 | 0.7415 | 264 | {"das_E_simple_threshold": 287, "md_E_simple_threshold": 275} |
| das_EC_a2_gate_only | 383/384 | 286 | 0.7467 | 264 | {"das_E_simple_threshold": 287, "md_E_simple_threshold": 275} |
| md_E | 160/384 | 124 | 0.7750 | 137 | {"das_E_simple_threshold": 146, "md_E_simple_threshold": 140} |
| md_C_a-2_gate_and_answer | 5/384 | 4 | 0.8000 | 5 | {"das_E_simple_threshold": 5, "md_E_simple_threshold": 5} |
| md_C_a-2_gate_only | 5/384 | 4 | 0.8000 | 5 | {"das_E_simple_threshold": 5, "md_E_simple_threshold": 5} |
| md_C_a-1_gate_and_answer | 87/384 | 70 | 0.8046 | 76 | {"das_E_simple_threshold": 81, "md_E_simple_threshold": 76} |
| md_C_a-1_gate_only | 87/384 | 70 | 0.8046 | 76 | {"das_E_simple_threshold": 81, "md_E_simple_threshold": 76} |
| md_C_a-0.5_gate_and_answer | 118/384 | 94 | 0.7966 | 102 | {"das_E_simple_threshold": 109, "md_E_simple_threshold": 104} |
| md_C_a-0.5_gate_only | 118/384 | 94 | 0.7966 | 102 | {"das_E_simple_threshold": 109, "md_E_simple_threshold": 104} |
| md_C_a0_gate_and_answer | 152/384 | 114 | 0.7500 | 130 | {"das_E_simple_threshold": 138, "md_E_simple_threshold": 133} |
| md_C_a0_gate_only | 152/384 | 114 | 0.7500 | 130 | {"das_E_simple_threshold": 138, "md_E_simple_threshold": 133} |
| md_C_a0.5_gate_and_answer | 185/384 | 127 | 0.6865 | 157 | {"das_E_simple_threshold": 166, "md_E_simple_threshold": 158} |
| md_C_a0.5_gate_only | 185/384 | 127 | 0.6865 | 157 | {"das_E_simple_threshold": 166, "md_E_simple_threshold": 158} |
| md_C_a1_gate_and_answer | 236/384 | 149 | 0.6314 | 190 | {"das_E_simple_threshold": 199, "md_E_simple_threshold": 192} |
| md_C_a1_gate_only | 236/384 | 149 | 0.6314 | 190 | {"das_E_simple_threshold": 199, "md_E_simple_threshold": 192} |
| md_C_a2_gate_and_answer | 383/384 | 271 | 0.7076 | 264 | {"das_E_simple_threshold": 287, "md_E_simple_threshold": 275} |
| md_C_a2_gate_only | 383/384 | 264 | 0.6893 | 264 | {"das_E_simple_threshold": 287, "md_E_simple_threshold": 275} |
| md_EC_a-2_gate_and_answer | 10/384 | 7 | 0.7000 | 10 | {"das_E_simple_threshold": 10, "md_E_simple_threshold": 10} |
| md_EC_a-2_gate_only | 10/384 | 7 | 0.7000 | 10 | {"das_E_simple_threshold": 10, "md_E_simple_threshold": 10} |
| md_EC_a-1_gate_and_answer | 105/384 | 87 | 0.8286 | 89 | {"das_E_simple_threshold": 98, "md_E_simple_threshold": 91} |
| md_EC_a-1_gate_only | 105/384 | 87 | 0.8286 | 89 | {"das_E_simple_threshold": 98, "md_E_simple_threshold": 91} |
| md_EC_a-0.5_gate_and_answer | 131/384 | 103 | 0.7863 | 114 | {"das_E_simple_threshold": 120, "md_E_simple_threshold": 115} |
| md_EC_a-0.5_gate_only | 131/384 | 103 | 0.7863 | 114 | {"das_E_simple_threshold": 120, "md_E_simple_threshold": 115} |
| md_EC_a0_gate_and_answer | 160/384 | 124 | 0.7750 | 137 | {"das_E_simple_threshold": 146, "md_E_simple_threshold": 140} |
| md_EC_a0_gate_only | 160/384 | 124 | 0.7750 | 137 | {"das_E_simple_threshold": 146, "md_E_simple_threshold": 140} |
| md_EC_a0.5_gate_and_answer | 190/384 | 137 | 0.7211 | 160 | {"das_E_simple_threshold": 169, "md_E_simple_threshold": 163} |
| md_EC_a0.5_gate_only | 190/384 | 138 | 0.7263 | 160 | {"das_E_simple_threshold": 169, "md_E_simple_threshold": 163} |
| md_EC_a1_gate_and_answer | 223/384 | 152 | 0.6816 | 184 | {"das_E_simple_threshold": 189, "md_E_simple_threshold": 184} |
| md_EC_a1_gate_only | 223/384 | 150 | 0.6726 | 184 | {"das_E_simple_threshold": 189, "md_E_simple_threshold": 184} |
| md_EC_a2_gate_and_answer | 374/384 | 271 | 0.7246 | 262 | {"das_E_simple_threshold": 283, "md_E_simple_threshold": 271} |
| md_EC_a2_gate_only | 374/384 | 266 | 0.7112 | 262 | {"das_E_simple_threshold": 283, "md_E_simple_threshold": 271} |

## qa

| 方法 | 实际回答 | 答对 | 已回答准确率 | 原模型简单筛题答对 | 固定E简单筛题答对 |
|---|---:|---:|---:|---:|---:|
| baseline | 62/192 | 31 | 0.5000 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| das_E | 61/192 | 30 | 0.4918 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| das_C_a-2_gate_and_answer | 0/192 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_C_a-2_gate_only | 0/192 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_C_a-1_gate_and_answer | 20/192 | 11 | 0.5500 | 12 | {"das_E_simple_threshold": 12, "md_E_simple_threshold": 12} |
| das_C_a-1_gate_only | 20/192 | 12 | 0.6000 | 12 | {"das_E_simple_threshold": 12, "md_E_simple_threshold": 12} |
| das_C_a-0.5_gate_and_answer | 35/192 | 18 | 0.5143 | 20 | {"das_E_simple_threshold": 20, "md_E_simple_threshold": 21} |
| das_C_a-0.5_gate_only | 36/192 | 18 | 0.5000 | 21 | {"das_E_simple_threshold": 21, "md_E_simple_threshold": 21} |
| das_C_a0_gate_and_answer | 62/192 | 31 | 0.5000 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| das_C_a0_gate_only | 62/192 | 31 | 0.5000 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| das_C_a0.5_gate_and_answer | 83/192 | 37 | 0.4458 | 39 | {"das_E_simple_threshold": 38, "md_E_simple_threshold": 38} |
| das_C_a0.5_gate_only | 81/192 | 36 | 0.4444 | 37 | {"das_E_simple_threshold": 38, "md_E_simple_threshold": 38} |
| das_C_a1_gate_and_answer | 109/192 | 49 | 0.4495 | 49 | {"das_E_simple_threshold": 47, "md_E_simple_threshold": 46} |
| das_C_a1_gate_only | 104/192 | 44 | 0.4231 | 48 | {"das_E_simple_threshold": 46, "md_E_simple_threshold": 45} |
| das_C_a2_gate_and_answer | 137/192 | 56 | 0.4088 | None | {"das_E_simple_threshold": null, "md_E_simple_threshold": null} |
| das_C_a2_gate_only | 117/192 | 51 | 0.4359 | 51 | {"das_E_simple_threshold": 50, "md_E_simple_threshold": 48} |
| das_EC_a-2_gate_and_answer | 0/192 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_EC_a-2_gate_only | 0/192 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| das_EC_a-1_gate_and_answer | 20/192 | 9 | 0.4500 | 12 | {"das_E_simple_threshold": 12, "md_E_simple_threshold": 12} |
| das_EC_a-1_gate_only | 20/192 | 12 | 0.6000 | 12 | {"das_E_simple_threshold": 12, "md_E_simple_threshold": 12} |
| das_EC_a-0.5_gate_and_answer | 36/192 | 20 | 0.5556 | 21 | {"das_E_simple_threshold": 21, "md_E_simple_threshold": 21} |
| das_EC_a-0.5_gate_only | 37/192 | 19 | 0.5135 | 21 | {"das_E_simple_threshold": 21, "md_E_simple_threshold": 22} |
| das_EC_a0_gate_and_answer | 61/192 | 30 | 0.4918 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| das_EC_a0_gate_only | 61/192 | 30 | 0.4918 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| das_EC_a0.5_gate_and_answer | 83/192 | 38 | 0.4578 | 39 | {"das_E_simple_threshold": 38, "md_E_simple_threshold": 38} |
| das_EC_a0.5_gate_only | 81/192 | 36 | 0.4444 | 37 | {"das_E_simple_threshold": 38, "md_E_simple_threshold": 38} |
| das_EC_a1_gate_and_answer | 107/192 | 47 | 0.4393 | 49 | {"das_E_simple_threshold": 47, "md_E_simple_threshold": 46} |
| das_EC_a1_gate_only | 103/192 | 43 | 0.4175 | 48 | {"das_E_simple_threshold": 46, "md_E_simple_threshold": 45} |
| das_EC_a2_gate_and_answer | 137/192 | 56 | 0.4088 | None | {"das_E_simple_threshold": null, "md_E_simple_threshold": null} |
| das_EC_a2_gate_only | 117/192 | 50 | 0.4274 | 51 | {"das_E_simple_threshold": 50, "md_E_simple_threshold": 48} |
| md_E | 62/192 | 28 | 0.4516 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| md_C_a-2_gate_and_answer | 0/192 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_C_a-2_gate_only | 0/192 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_C_a-1_gate_and_answer | 20/192 | 11 | 0.5500 | 12 | {"das_E_simple_threshold": 12, "md_E_simple_threshold": 12} |
| md_C_a-1_gate_only | 20/192 | 12 | 0.6000 | 12 | {"das_E_simple_threshold": 12, "md_E_simple_threshold": 12} |
| md_C_a-0.5_gate_and_answer | 35/192 | 18 | 0.5143 | 20 | {"das_E_simple_threshold": 20, "md_E_simple_threshold": 21} |
| md_C_a-0.5_gate_only | 36/192 | 18 | 0.5000 | 21 | {"das_E_simple_threshold": 21, "md_E_simple_threshold": 21} |
| md_C_a0_gate_and_answer | 62/192 | 31 | 0.5000 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| md_C_a0_gate_only | 62/192 | 31 | 0.5000 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| md_C_a0.5_gate_and_answer | 83/192 | 37 | 0.4458 | 39 | {"das_E_simple_threshold": 38, "md_E_simple_threshold": 38} |
| md_C_a0.5_gate_only | 81/192 | 36 | 0.4444 | 37 | {"das_E_simple_threshold": 38, "md_E_simple_threshold": 38} |
| md_C_a1_gate_and_answer | 109/192 | 49 | 0.4495 | 49 | {"das_E_simple_threshold": 47, "md_E_simple_threshold": 46} |
| md_C_a1_gate_only | 104/192 | 44 | 0.4231 | 48 | {"das_E_simple_threshold": 46, "md_E_simple_threshold": 45} |
| md_C_a2_gate_and_answer | 137/192 | 56 | 0.4088 | None | {"das_E_simple_threshold": null, "md_E_simple_threshold": null} |
| md_C_a2_gate_only | 117/192 | 51 | 0.4359 | 51 | {"das_E_simple_threshold": 50, "md_E_simple_threshold": 48} |
| md_EC_a-2_gate_and_answer | 0/192 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_EC_a-2_gate_only | 0/192 | 0 | — | 0 | {"das_E_simple_threshold": 0, "md_E_simple_threshold": 0} |
| md_EC_a-1_gate_and_answer | 20/192 | 11 | 0.5500 | 12 | {"das_E_simple_threshold": 12, "md_E_simple_threshold": 12} |
| md_EC_a-1_gate_only | 20/192 | 12 | 0.6000 | 12 | {"das_E_simple_threshold": 12, "md_E_simple_threshold": 12} |
| md_EC_a-0.5_gate_and_answer | 34/192 | 17 | 0.5000 | 19 | {"das_E_simple_threshold": 19, "md_E_simple_threshold": 20} |
| md_EC_a-0.5_gate_only | 35/192 | 17 | 0.4857 | 20 | {"das_E_simple_threshold": 20, "md_E_simple_threshold": 21} |
| md_EC_a0_gate_and_answer | 62/192 | 28 | 0.4516 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| md_EC_a0_gate_only | 62/192 | 28 | 0.4516 | 33 | {"das_E_simple_threshold": 33, "md_E_simple_threshold": 32} |
| md_EC_a0.5_gate_and_answer | 82/192 | 34 | 0.4146 | 38 | {"das_E_simple_threshold": 38, "md_E_simple_threshold": 38} |
| md_EC_a0.5_gate_only | 81/192 | 33 | 0.4074 | 37 | {"das_E_simple_threshold": 38, "md_E_simple_threshold": 38} |
| md_EC_a1_gate_and_answer | 109/192 | 45 | 0.4128 | 49 | {"das_E_simple_threshold": 47, "md_E_simple_threshold": 46} |
| md_EC_a1_gate_only | 104/192 | 41 | 0.3942 | 48 | {"das_E_simple_threshold": 46, "md_E_simple_threshold": 45} |
| md_EC_a2_gate_and_answer | 137/192 | 53 | 0.3869 | None | {"das_E_simple_threshold": null, "md_E_simple_threshold": null} |
| md_EC_a2_gate_only | 117/192 | 48 | 0.4103 | 51 | {"das_E_simple_threshold": 50, "md_E_simple_threshold": 48} |

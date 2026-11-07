# 来源实验：全臂统一拒答守卫与质量侧表

qwen3-8b / source_question_mask_uniform256_diagnostic_v1
已暴露调参题探索；原语义裁判仍可能错，不把以下数值当无误真值。共同原模型分与各自生成置信并列，不挑更好的读出。
R1 为完整个人拒答守卫；R2 再将未自然结束/替换乱码的正确回答计失败，资格/分母/选中ID与R1保持一致。

| 视图 | 方法 | 自然答对/实答 | 共同分25/50/75%答对 | 各自置信25/50/75%答对 |
|---|---|---|---|---|
| R1 | baseline | 51/116 | 26/46/不可行 | 28/45/不可行 |
| R1 | source_only | 50/111 | 23/41/不可行 | 27/49/不可行 |
| R1 | equal_logit_mix | 49/113 | 25/44/不可行 | 28/46/不可行 |
| R1 | source_das_swap | 50/116 | 26/45/不可行 | 28/44/不可行 |
| R1 | source_random_swap | 51/116 | 26/45/不可行 | 26/45/不可行 |
| R1 | trained_source_swap_das_init | 86/186 | 23/43/64 | 24/50/71 |
| R1 | pair_preference_source_swap | 104/177 | 28/56/85 | 37/63/92 |
| R1 | pair_preference_question_masked_source | 55/119 | 21/43/不可行 | 27/51/不可行 |
| R2 | baseline | 51/116 | 26/46/不可行 | 28/45/不可行 |
| R2 | source_only | 50/111 | 23/41/不可行 | 27/49/不可行 |
| R2 | equal_logit_mix | 49/113 | 25/44/不可行 | 28/46/不可行 |
| R2 | source_das_swap | 50/116 | 26/45/不可行 | 28/44/不可行 |
| R2 | source_random_swap | 51/116 | 26/45/不可行 | 26/45/不可行 |
| R2 | trained_source_swap_das_init | 86/186 | 23/43/64 | 24/50/71 |
| R2 | pair_preference_source_swap | 104/177 | 28/56/85 | 37/63/92 |
| R2 | pair_preference_question_masked_source | 55/119 | 21/43/不可行 | 27/51/不可行 |

# 来源实验：全臂统一拒答守卫与质量侧表

llama3.1-8b / source_actual_norm_random_orientation_control_v1
已暴露调参题探索；原语义裁判仍可能错，不把以下数值当无误真值。共同原模型分与各自生成置信并列，不挑更好的读出。
R1 为完整个人拒答守卫；R2 再将未自然结束/替换乱码的正确回答计失败，资格/分母/选中ID与R1保持一致。

| 视图 | 方法 | 自然答对/实答 | 共同分25/50/75%答对 | 各自置信25/50/75%答对 |
|---|---|---|---|---|
| R1 | baseline | 57/168 | 25/46/55 | 25/45/55 |
| R1 | source_only | 61/129 | 25/44/不可行 | 27/50/不可行 |
| R1 | equal_logit_mix | 63/150 | 21/47/60 | 26/48/63 |
| R1 | source_das_swap | 58/168 | 25/46/56 | 25/46/56 |
| R1 | source_random_swap | 58/167 | 25/46/55 | 25/45/56 |
| R1 | trained_source_swap_das_init | 67/165 | 22/45/60 | 27/47/65 |
| R1 | pair_preference_source_swap | 68/166 | 23/47/61 | 26/50/63 |
| R1 | size_matched_random_s42 | 58/167 | 25/46/54 | 24/45/55 |
| R1 | size_matched_random_s43 | 57/167 | 25/46/55 | 25/46/56 |
| R1 | size_matched_random_s44 | 59/166 | 25/47/58 | 24/46/58 |
| R2 | baseline | 56/168 | 25/45/54 | 25/44/54 |
| R2 | source_only | 59/129 | 23/42/不可行 | 25/48/不可行 |
| R2 | equal_logit_mix | 61/150 | 20/45/58 | 25/46/61 |
| R2 | source_das_swap | 57/168 | 25/45/55 | 25/45/55 |
| R2 | source_random_swap | 57/167 | 25/45/54 | 25/44/55 |
| R2 | trained_source_swap_das_init | 64/165 | 20/42/57 | 25/44/62 |
| R2 | pair_preference_source_swap | 67/166 | 23/46/60 | 26/49/62 |
| R2 | size_matched_random_s42 | 57/167 | 25/45/53 | 24/44/54 |
| R2 | size_matched_random_s43 | 56/167 | 25/45/54 | 25/45/55 |
| R2 | size_matched_random_s44 | 58/166 | 25/46/57 | 24/45/57 |

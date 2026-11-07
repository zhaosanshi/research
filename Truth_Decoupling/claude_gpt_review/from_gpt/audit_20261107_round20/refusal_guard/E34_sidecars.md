# E34 七组统一口径侧表

全部为已经看过的调参题探索；原生成、原判分与选择一律保留。
R1 只修完整个人拒答；R2 在 R1 上将未自然结束或含替换乱码的正确回答计作失败。R2 是严格交付质量敏感性，不声称所有截断回答的已说内容都错误。
每一组都用同一规则；拒答仍留在分母，A/B 都有交付资格。两种置信读出并列，不择优。

## llama3.1-8b

| 视图 | 方法 | 自然答对/实答 | 共同分 25/50/75% 答对 | 各自置信 25/50/75% 答对 |
|---|---|---|---|---|
| R1_guard | baseline | 57/168 | 25/46/55 | 25/45/55 |
| R1_guard | source_only | 61/129 | 25/44/不可行 | 27/50/不可行 |
| R1_guard | equal_logit_mix | 63/150 | 21/47/60 | 26/48/63 |
| R1_guard | source_das_swap | 58/168 | 25/46/56 | 25/46/56 |
| R1_guard | source_random_swap | 58/167 | 25/46/55 | 25/45/56 |
| R1_guard | trained_source_swap_das_init | 67/165 | 22/45/60 | 27/47/65 |
| R1_guard | pair_preference_source_swap | 68/166 | 23/47/61 | 26/50/63 |
| R2_guard_complete | baseline | 56/168 | 25/45/54 | 25/44/54 |
| R2_guard_complete | source_only | 59/129 | 23/42/不可行 | 25/48/不可行 |
| R2_guard_complete | equal_logit_mix | 61/150 | 20/45/58 | 25/46/61 |
| R2_guard_complete | source_das_swap | 57/168 | 25/45/55 | 25/45/55 |
| R2_guard_complete | source_random_swap | 57/167 | 25/45/54 | 25/44/55 |
| R2_guard_complete | trained_source_swap_das_init | 64/165 | 20/42/57 | 25/44/62 |
| R2_guard_complete | pair_preference_source_swap | 67/166 | 23/46/60 | 26/49/62 |
## qwen3-8b

| 视图 | 方法 | 自然答对/实答 | 共同分 25/50/75% 答对 | 各自置信 25/50/75% 答对 |
|---|---|---|---|---|
| R1_guard | baseline | 51/116 | 26/46/不可行 | 28/45/不可行 |
| R1_guard | source_only | 50/111 | 23/41/不可行 | 27/49/不可行 |
| R1_guard | equal_logit_mix | 49/113 | 25/44/不可行 | 28/46/不可行 |
| R1_guard | source_das_swap | 50/116 | 26/45/不可行 | 28/44/不可行 |
| R1_guard | source_random_swap | 51/116 | 26/45/不可行 | 26/45/不可行 |
| R1_guard | trained_source_swap_das_init | 86/186 | 23/43/64 | 25/52/72 |
| R1_guard | pair_preference_source_swap | 107/177 | 30/60/88 | 37/63/94 |
| R2_guard_complete | baseline | 51/116 | 26/46/不可行 | 28/45/不可行 |
| R2_guard_complete | source_only | 49/111 | 23/40/不可行 | 27/48/不可行 |
| R2_guard_complete | equal_logit_mix | 49/113 | 25/44/不可行 | 28/46/不可行 |
| R2_guard_complete | source_das_swap | 49/116 | 26/44/不可行 | 28/43/不可行 |
| R2_guard_complete | source_random_swap | 50/116 | 26/44/不可行 | 26/44/不可行 |
| R2_guard_complete | trained_source_swap_das_init | 84/186 | 22/41/62 | 25/52/70 |
| R2_guard_complete | pair_preference_source_swap | 80/177 | 23/41/61 | 37/52/72 |

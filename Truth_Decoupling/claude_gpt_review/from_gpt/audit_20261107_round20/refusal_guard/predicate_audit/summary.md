# 带原文断言的盲审：校准通过，但真实数据未全部通过

这是同一个 DeepSeek 模型的独立提示与缓存审计，不是独立模型复现。12 个预设校准例全部通过；真实七组两模型的 2688 条记录去重为 1028 个问答，1024 个得到可验证原文引用的 JSON，4 个未通过，不伪造标签。
四条未决都是具体国家/城市名单，按既有规则明确属于实质尝试，仅A/B未定；全局枚举16种A/B组合，绝不把它们改成拒答而改变交付资格。下面范围不是置信区间；其余标签仍可能有语义错误，不将此表当最终真值或确认性结论。各组采用相同拒答守卫。

## llama3.1-8b

| 视图 | 方法 | 自然答对范围 | 自然实答范围 | 共同分25/50/75% | 各自置信25/50/75% |
|---|---|---|---|---|---|
| semantic | baseline | 57 | 167 | 24/46/54 | 24/46/54 |
| semantic | source_only | 60–61 | 129 | 24/43–44/不可行 | 26/49–50/不可行 |
| semantic | equal_logit_mix | 56–57 | 150 | 19/44/53–54 | 23/42/56–57 |
| semantic | source_das_swap | 58 | 167 | 24/47/55 | 24/47/55 |
| semantic | source_random_swap | 57 | 167 | 24/46/54 | 24/45/54 |
| semantic | trained_source_swap_das_init | 62 | 165 | 22/42/56 | 26/45/61 |
| semantic | pair_preference_source_swap | 62 | 164 | 22/46/55 | 25/47/59 |
| strict_complete | baseline | 55 | 167 | 24/44/52 | 24/44/52 |
| strict_complete | source_only | 57–58 | 129 | 22/40–41/不可行 | 24/46–47/不可行 |
| strict_complete | equal_logit_mix | 54–55 | 150 | 18/42/51–52 | 22/40/54–55 |
| strict_complete | source_das_swap | 56 | 167 | 24/45/53 | 24/45/53 |
| strict_complete | source_random_swap | 55 | 167 | 24/44/52 | 24/43/52 |
| strict_complete | trained_source_swap_das_init | 58 | 165 | 20/38/52 | 24/41/57 |
| strict_complete | pair_preference_source_swap | 60 | 164 | 22/44/53 | 25/45/57 |
## qwen3-8b

| 视图 | 方法 | 自然答对范围 | 自然实答范围 | 共同分25/50/75% | 各自置信25/50/75% |
|---|---|---|---|---|---|
| semantic | baseline | 46 | 117 | 26/40/不可行 | 26/40/不可行 |
| semantic | source_only | 47 | 111 | 24/38/不可行 | 28/46/不可行 |
| semantic | equal_logit_mix | 44 | 114 | 24/39/不可行 | 28/41/不可行 |
| semantic | source_das_swap | 46 | 117 | 26/41/不可行 | 27/41/不可行 |
| semantic | source_random_swap | 47 | 117 | 26/41/不可行 | 25/41/不可行 |
| semantic | trained_source_swap_das_init | 75 | 187 | 21/36/56 | 25/47/63 |
| semantic | pair_preference_source_swap | 101–103 | 178 | 29/56–57/82–84 | 35/61/88 |
| strict_complete | baseline | 46 | 117 | 26/40/不可行 | 26/40/不可行 |
| strict_complete | source_only | 46 | 111 | 24/37/不可行 | 28/45/不可行 |
| strict_complete | equal_logit_mix | 44 | 114 | 24/39/不可行 | 28/41/不可行 |
| strict_complete | source_das_swap | 45 | 117 | 26/40/不可行 | 27/40/不可行 |
| strict_complete | source_random_swap | 46 | 117 | 26/40/不可行 | 25/40/不可行 |
| strict_complete | trained_source_swap_das_init | 75 | 187 | 21/36/56 | 25/47/63 |
| strict_complete | pair_preference_source_swap | 73 | 178 | 21/38/55 | 34/49/66 |

# 生成置信读出敏感性

qwen3-8b：调参集探索；两种读出并列，不择优宣称成功。
原生置信来自实际生成词元概率；只有自然结束时才在计算分数时排除最后停止词，原文和词元从未改动。
实质回答仍按带参考的离线语义标注计数；拒答保留分母。

| 作答份额 | 方法 | 统一原模型分：答对/交付 | 各自生成置信：答对/交付 |
|---:|---|---:|---:|
| 25% | baseline | 27/48 | 28/48 |
| 25% | source_only | 23/48 | 27/48 |
| 25% | equal_logit_mix | 25/48 | 28/48 |
| 25% | source_das_swap | 27/48 | 27/48 |
| 25% | source_random_swap | 27/48 | 28/48 |
| 25% | trained_source_swap_das_init | 26/48 | 27/48 |
| 50% | baseline | 46/96 | 46/96 |
| 50% | source_only | 41/96 | 49/96 |
| 50% | equal_logit_mix | 44/96 | 46/96 |
| 50% | source_das_swap | 46/96 | 46/96 |
| 50% | source_random_swap | 46/96 | 46/96 |
| 50% | trained_source_swap_das_init | 45/96 | 46/96 |
| 75% | baseline | 不可行 | 不可行 |
| 75% | source_only | 不可行 | 不可行 |
| 75% | equal_logit_mix | 不可行 | 不可行 |
| 75% | source_das_swap | 不可行 | 不可行 |
| 75% | source_random_swap | 不可行 | 不可行 |
| 75% | trained_source_swap_das_init | 不可行 | 不可行 |

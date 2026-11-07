# 真实来源交换：同实际作答率探索性比较

全部来自调参集，不是新的确认性测试。
所有组使用同一个未干预模型给生成答案打平均词分；实质作答由离线语义标注统计。

## 目标 25%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 状态 |
|---|---:|---:|---:|---|
| baseline | 48 | 25 | 0.5208333333333334 | 可行 |
| equal_logit_mix | 48 | 21 | 0.4375 | 可行 |
| pair_preference_question_masked_source | 48 | 26 | 0.5416666666666666 | 可行 |
| pair_preference_source_swap | 48 | 23 | 0.4791666666666667 | 可行 |
| source_das_swap | 48 | 25 | 0.5208333333333334 | 可行 |
| source_only | 48 | 25 | 0.5208333333333334 | 可行 |
| source_random_swap | 48 | 25 | 0.5208333333333334 | 可行 |
| trained_source_swap_das_init | 48 | 22 | 0.4583333333333333 | 可行 |

## 目标 50%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 状态 |
|---|---:|---:|---:|---|
| baseline | 96 | 46 | 0.4791666666666667 | 可行 |
| equal_logit_mix | 96 | 47 | 0.4895833333333333 | 可行 |
| pair_preference_question_masked_source | 96 | 54 | 0.5625 | 可行 |
| pair_preference_source_swap | 96 | 47 | 0.4895833333333333 | 可行 |
| source_das_swap | 96 | 46 | 0.4791666666666667 | 可行 |
| source_only | 96 | 44 | 0.4583333333333333 | 可行 |
| source_random_swap | 96 | 46 | 0.4791666666666667 | 可行 |
| trained_source_swap_das_init | 96 | 45 | 0.46875 | 可行 |

## 目标 75%

| 方法 | 实际回答 | 答对 | 已回答准确率 | 状态 |
|---|---:|---:|---:|---|
| baseline | 144 | 55 | 0.3819444444444444 | 可行 |
| equal_logit_mix | 144 | 60 | 0.4166666666666667 | 可行 |
| pair_preference_question_masked_source | 144 | 85 | 0.5902777777777778 | 可行 |
| pair_preference_source_swap | 144 | 61 | 0.4236111111111111 | 可行 |
| source_das_swap | 144 | 56 | 0.3888888888888889 | 可行 |
| source_only | None | None | None | 实际回答不足 |
| source_random_swap | 144 | 55 | 0.3819444444444444 | 可行 |
| trained_source_swap_das_init | 144 | 60 | 0.4166666666666667 | 可行 |

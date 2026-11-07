# 新来源控制：同实际作答率完整比较

所有数据为192道已使用调参题的探索性诊断。原判标签保留；完整个人拒答与质量敏感性统一侧表另列。原始逐题数据在e27、norm_random、source_question_mask对应归档，本目录保存新分数、完整比较、执行源码与三题评分验收。

所有组使用同一个未干预模型给完整生成文本打词元对数概率的平均值；长回答重新评分，没有沿用48词截断分数。

## results_e34_completion_codex_20261107/qwen3-8b/uniform256

| 方法 | 自然答对/实答 | 25%答对 | 50%答对 | 75%答对 | 未结束 |
|---|---:|---:|---:|---:|---:|
| baseline | 52/117 | 27 | 46 | 不可行 | 0 |
| source_only | 50/111 | 23 | 41 | 不可行 | 0 |
| equal_logit_mix | 49/113 | 25 | 44 | 不可行 | 0 |
| source_das_swap | 51/117 | 27 | 46 | 不可行 | 0 |
| source_random_swap | 52/117 | 27 | 46 | 不可行 | 0 |
| trained_source_swap_das_init | 86/186 | 23 | 43 | 64 | 2 |
| pair_preference_source_swap | 104/177 | 28 | 56 | 85 | 1 |

[逐项配对检验与重选题区间](results_e34_completion_codex_20261107/qwen3-8b/uniform256/tune_matched_coverage.json)。

## results_r20_norm_random_codex_20261107/llama3.1-8b/full192

| 方法 | 自然答对/实答 | 25%答对 | 50%答对 | 75%答对 | 未结束 |
|---|---:|---:|---:|---:|---:|
| baseline | 57/168 | 25 | 46 | 55 | 4 |
| source_only | 61/129 | 25 | 44 | 不可行 | 7 |
| equal_logit_mix | 63/150 | 21 | 47 | 60 | 4 |
| source_das_swap | 58/168 | 25 | 46 | 56 | 4 |
| source_random_swap | 58/167 | 25 | 46 | 55 | 4 |
| trained_source_swap_das_init | 67/165 | 22 | 45 | 60 | 8 |
| pair_preference_source_swap | 68/166 | 23 | 47 | 61 | 4 |
| size_matched_random_s42 | 58/167 | 25 | 46 | 54 | 5 |
| size_matched_random_s43 | 57/167 | 25 | 46 | 55 | 6 |
| size_matched_random_s44 | 59/166 | 25 | 47 | 58 | 5 |

[逐项配对检验与重选题区间](results_r20_norm_random_codex_20261107/llama3.1-8b/full192/tune_matched_coverage.json)。

## results_r20_norm_random_codex_20261107/qwen3-8b/full192

| 方法 | 自然答对/实答 | 25%答对 | 50%答对 | 75%答对 | 未结束 |
|---|---:|---:|---:|---:|---:|
| baseline | 52/117 | 27 | 46 | 不可行 | 0 |
| source_only | 50/111 | 23 | 41 | 不可行 | 1 |
| equal_logit_mix | 49/113 | 25 | 44 | 不可行 | 0 |
| source_das_swap | 51/117 | 27 | 46 | 不可行 | 1 |
| source_random_swap | 52/117 | 27 | 46 | 不可行 | 1 |
| trained_source_swap_das_init | 86/186 | 23 | 43 | 64 | 4 |
| pair_preference_source_swap | 107/177 | 30 | 60 | 88 | 46 |
| size_matched_random_s42 | 53/120 | 25 | 47 | 不可行 | 0 |
| size_matched_random_s43 | 50/114 | 27 | 46 | 不可行 | 0 |
| size_matched_random_s44 | 50/113 | 27 | 48 | 不可行 | 1 |

[逐项配对检验与重选题区间](results_r20_norm_random_codex_20261107/qwen3-8b/full192/tune_matched_coverage.json)。

## results_source_question_mask_codex_20261107/llama3.1-8b/full92_ep3

| 方法 | 自然答对/实答 | 25%答对 | 50%答对 | 75%答对 | 未结束 |
|---|---:|---:|---:|---:|---:|
| baseline | 57/168 | 25 | 46 | 55 | 4 |
| source_only | 61/129 | 25 | 44 | 不可行 | 7 |
| equal_logit_mix | 63/150 | 21 | 47 | 60 | 4 |
| source_das_swap | 58/168 | 25 | 46 | 56 | 4 |
| source_random_swap | 58/167 | 25 | 46 | 55 | 4 |
| trained_source_swap_das_init | 67/165 | 22 | 45 | 60 | 8 |
| pair_preference_source_swap | 68/166 | 23 | 47 | 61 | 4 |
| pair_preference_question_masked_source | 87/151 | 26 | 54 | 85 | 5 |

[逐项配对检验与重选题区间](results_source_question_mask_codex_20261107/llama3.1-8b/full92_ep3/tune_matched_coverage.json)。

## results_source_question_mask_codex_20261107/qwen3-8b/full92_ep3

| 方法 | 自然答对/实答 | 25%答对 | 50%答对 | 75%答对 | 未结束 |
|---|---:|---:|---:|---:|---:|
| baseline | 52/117 | 27 | 46 | 不可行 | 0 |
| source_only | 50/111 | 23 | 41 | 不可行 | 1 |
| equal_logit_mix | 49/113 | 25 | 44 | 不可行 | 0 |
| source_das_swap | 51/117 | 27 | 46 | 不可行 | 1 |
| source_random_swap | 52/117 | 27 | 46 | 不可行 | 1 |
| trained_source_swap_das_init | 86/186 | 23 | 43 | 64 | 4 |
| pair_preference_source_swap | 107/177 | 30 | 60 | 88 | 46 |
| pair_preference_question_masked_source | 55/120 | 22 | 42 | 不可行 | 7 |

[逐项配对检验与重选题区间](results_source_question_mask_codex_20261107/qwen3-8b/full92_ep3/tune_matched_coverage.json)。

## results_source_question_mask_codex_20261107/qwen3-8b/uniform256

| 方法 | 自然答对/实答 | 25%答对 | 50%答对 | 75%答对 | 未结束 |
|---|---:|---:|---:|---:|---:|
| baseline | 52/117 | 27 | 46 | 不可行 | 0 |
| source_only | 50/111 | 23 | 41 | 不可行 | 0 |
| equal_logit_mix | 49/113 | 25 | 44 | 不可行 | 0 |
| source_das_swap | 51/117 | 27 | 46 | 不可行 | 0 |
| source_random_swap | 52/117 | 27 | 46 | 不可行 | 0 |
| trained_source_swap_das_init | 86/186 | 23 | 43 | 64 | 2 |
| pair_preference_source_swap | 104/177 | 28 | 56 | 85 | 1 |
| pair_preference_question_masked_source | 56/120 | 22 | 43 | 不可行 | 0 |

[逐项配对检验与重选题区间](results_source_question_mask_codex_20261107/qwen3-8b/uniform256/tune_matched_coverage.json)。

## 当前可支持的判断

- 匹配真实剂量后，三个未训练随机方向仍无大收益；这不能改称完全无信息随机，因为它借了新E的标量力度。
- Llama遮题来源在相同交付数量下反而更好，Qwen相反；来源语义的作用因模型而异，且遮题后实际写入量也改变，不能作纯语义因果分解。
- Qwen放宽到256后，同交付96题的新配对来源答对56，原模型46；它是单独E的探索性信号，不是E+C或未见测试集确认。
- 同一个判分模型存在已确认错误，显著性必须连同完整性、统一拒答规则和标签不确定性一起解释，不能挑较好的表当最终真值。

[独立重算](independent_recount.json)：6组运行、153个配额格与451个可行固定交付配对检验一致。这里只独立重算交付与精确检验，未重新执行模型评分或全部自举。

统一完整拒答与质量侧表：[等实际力度随机](../norm_random/sensitivity/)、[遮题来源](../source_question_mask/sensitivity/)。

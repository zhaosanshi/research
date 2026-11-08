# E42 select — llama3.1-8b (tuning split, 192 questions; 0 abstention-only-gold excluded; grading: two_axis qa_two_axis_v6: judge2/final.jsonl)

| pick rule | 实质作答 S | A类（真且有信息） | H类（真且无信息） | Δcorrect vs greedy [95% CI] | Δcorrect vs E[random] [95% CI] | Δcorrect vs max_logp [95% CI] |
|---|---|---|---|---|---|---|
| coord_dasce_L28 | 161 | 74 | 14 | +7 [-5, +18] | +5.1 [-3.6, +13.4] | +0 [-11, +11] |
| coord_dasce_L29 | 163 | 75 | 12 | +8 [-4, +19] | +6.1 [-3.2, +14.6] | +1 [-10, +12] |
| coord_dasce_L30 | 163 | 77 | 12 | +10 [-2, +21] | +8.1 [-1.2, +16.8] | +3 [-8, +13] |
| coord_dasce_L31 | 164 | 78 | 11 | +11 [-1, +22] | +9.1 [+0.2, +18.3] | +4 [-7, +15] |
| coord_dasce_mean | 163 | 76 | 12 | +9 [-3, +20] | +7.1 [-2.3, +15.9] | +2 [-9, +13] |
| coord_qadas_L28 | 162 | 77 | 13 | +10 [-1, +21] | +8.1 [-0.3, +16.8] | +3 [-8, +14] |
| coord_qadas_L29 | 163 | 75 | 12 | +8 [-4, +19] | +6.1 [-2.6, +14.9] | +1 [-10, +13] |
| coord_qadas_L30 | 164 | 79 | 11 | +12 [+1, +23] | +10.1 [+2.1, +18.4] | +5 [-6, +16] |
| coord_qadas_mean | 163 | 77 | 12 | +10 [-1, +21] | +8.1 [-0.1, +16.7] | +3 [-8, +15] |
| greedy | 163 | 67 | 4 | +0 [+0, +0] | -1.9 [-7.3, +3.3] | -7 [-15, +1] |
| max_logp | 166 | 74 | 8 | +7 [-1, +15] | +5.1 [-0.9, +11.4] | +0 [+0, +0] |
| random | 168 | 67 | 7 | +0 [-8, +8] | -1.9 [-7.8, +4.0] | -7 [-16, +1] |
| verdict | 164 | 77 | 11 | +10 [-1, +21] | +8.1 [-0.2, +16.4] | +3 [-8, +14] |
| E[random pick] | 167.0 | 68.9 | — | — | — | — |
| any candidate correct (upper bound) | — | 102 | — | — | — | — |

内容判分与输出质量分开保留（不改写 A 为错或拒答）：

| 条件/规则 | 未自然结束 | 替换字符 | A但未结束 | A且替换字符 |
|---|---:|---:|---:|---:|
| coord_dasce_L28 | 5 | 0 | 1 | 0 |
| coord_dasce_L29 | 6 | 0 | 1 | 0 |
| coord_dasce_L30 | 6 | 0 | 1 | 0 |
| coord_dasce_L31 | 6 | 0 | 1 | 0 |
| coord_dasce_mean | 6 | 0 | 1 | 0 |
| coord_qadas_L28 | 4 | 0 | 1 | 0 |
| coord_qadas_L29 | 4 | 0 | 1 | 0 |
| coord_qadas_L30 | 3 | 0 | 1 | 0 |
| coord_qadas_mean | 4 | 0 | 1 | 0 |
| greedy | 4 | 0 | 1 | 0 |
| max_logp | 7 | 0 | 3 | 0 |
| random | 3 | 0 | 1 | 0 |
| verdict | 6 | 0 | 1 | 0 |

两维 answered 只按信息轴 S；B 是所有假话，可能无信息量，因此不再假设 answered = A+B。JSON 的 abstain/hedge 是终判 C/H 类计数，不是全部拒答/含糊行为。行为子类另报 refusal_subtype/hedge_subtype；人工 F/R 可为 B，F/H 也为 B，均不能算实质作答。另报 non_refusal（S/H）与固定分母的两轴率。

| 条件/规则 | 无信息且未拒答（子类H） | 拒答/无答复（子类R） | 非拒答 S/H |
|---|---:|---:|---:|
| coord_dasce_L28 | 15 | 16 | 176 |
| coord_dasce_L29 | 13 | 16 | 176 |
| coord_dasce_L30 | 13 | 16 | 176 |
| coord_dasce_L31 | 12 | 16 | 176 |
| coord_dasce_mean | 13 | 16 | 176 |
| coord_qadas_L28 | 14 | 16 | 176 |
| coord_qadas_L29 | 13 | 16 | 176 |
| coord_qadas_L30 | 12 | 16 | 176 |
| coord_qadas_mean | 13 | 16 | 176 |
| greedy | 6 | 23 | 169 |
| max_logp | 9 | 17 | 175 |
| random | 7 | 17 | 175 |
| verdict | 12 | 16 | 176 |

这里只是固定候选的调参诊断，各规则自然实质作答数另报；未执行同实际作答率配额比较，不得把此表当作已完成老师的同率主实验。oracle_* 与 gold_source_only 是看答案的诊断上界，不能当作可部署方法。

pool: mean distinct candidates 4.41; questions without a substantive candidate 19


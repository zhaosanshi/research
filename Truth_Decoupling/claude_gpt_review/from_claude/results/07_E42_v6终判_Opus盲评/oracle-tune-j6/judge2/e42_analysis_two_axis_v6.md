# E42 oracle — llama3.1-8b (tuning split, 192 questions; 0 abstention-only-gold excluded; grading: two_axis qa_two_axis_v6: judge2/final.jsonl)

| arm | 实质作答 S | A类（真且有信息） | B类（假，不保证有信息） | C类（真且拒答） | H类（真且无信息） | output changed | wrong→correct | correct→wrong | Δcorrect vs baseline [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 163 | 67 | 98 | 23 | 4 | 0 | 0 | 0 | +0 [+0, +0] |
| truthful_source_only | 124 | 72 | 53 | 63 | 4 | 123 | 16 | 4 | +5 [-7, +16] |
| gold_source_only | 164 | 162 | 2 | 16 | 12 | 184 | 83 | 1 | +95 [+80, +110] |
| oracle_dasce_L12 | 163 | 67 | 98 | 23 | 4 | 7 | 0 | 0 | +0 [+0, +0] |
| oracle_dasce_L20 | 164 | 69 | 97 | 23 | 3 | 14 | 1 | 0 | +2 [+0, +5] |
| oracle_dasce_L29 | 162 | 67 | 97 | 23 | 5 | 4 | 1 | 0 | +0 [-3, +3] |
| truthful_swap_dasce_L29 | 163 | 68 | 97 | 23 | 4 | 4 | 1 | 0 | +1 [+0, +3] |
| oracle_dasce_L31 | 158 | 65 | 95 | 25 | 7 | 16 | 1 | 0 | -2 [-6, +1] |
| oracle_qadas_L12 | 163 | 67 | 98 | 23 | 4 | 8 | 1 | 1 | +0 [-3, +3] |
| oracle_qadas_L20 | 162 | 67 | 96 | 23 | 6 | 15 | 1 | 0 | +0 [-3, +3] |
| oracle_qadas_L29 | 164 | 67 | 98 | 23 | 4 | 13 | 1 | 1 | +0 [-3, +3] |
| truthful_swap_qadas_L29 | 163 | 68 | 97 | 23 | 4 | 5 | 1 | 0 | +1 [+0, +3] |

内容判分与输出质量分开保留（不改写 A 为错或拒答）：

| 条件/规则 | 未自然结束 | 替换字符 | A但未结束 | A且替换字符 |
|---|---:|---:|---:|---:|
| baseline | 4 | 0 | 1 | 0 |
| truthful_source_only | 7 | 0 | 3 | 0 |
| gold_source_only | 0 | 0 | 0 | 0 |
| oracle_dasce_L12 | 4 | 0 | 1 | 0 |
| oracle_dasce_L20 | 4 | 0 | 1 | 0 |
| oracle_dasce_L29 | 4 | 0 | 1 | 0 |
| truthful_swap_dasce_L29 | 4 | 0 | 1 | 0 |
| oracle_dasce_L31 | 5 | 0 | 1 | 0 |
| oracle_qadas_L12 | 4 | 0 | 1 | 0 |
| oracle_qadas_L20 | 5 | 0 | 1 | 0 |
| oracle_qadas_L29 | 3 | 0 | 1 | 0 |
| truthful_swap_qadas_L29 | 4 | 0 | 1 | 0 |

两维 answered 只按信息轴 S；B 是所有假话，可能无信息量，因此不再假设 answered = A+B。JSON 的 abstain/hedge 是终判 C/H 类计数，不是全部拒答/含糊行为。行为子类另报 refusal_subtype/hedge_subtype；人工 F/R 可为 B，F/H 也为 B，均不能算实质作答。另报 non_refusal（S/H）与固定分母的两轴率。

| 条件/规则 | 无信息且未拒答（子类H） | 拒答/无答复（子类R） | 非拒答 S/H |
|---|---:|---:|---:|
| baseline | 6 | 23 | 169 |
| truthful_source_only | 5 | 63 | 129 |
| gold_source_only | 12 | 16 | 176 |
| oracle_dasce_L12 | 6 | 23 | 169 |
| oracle_dasce_L20 | 5 | 23 | 169 |
| oracle_dasce_L29 | 7 | 23 | 169 |
| truthful_swap_dasce_L29 | 6 | 23 | 169 |
| oracle_dasce_L31 | 9 | 25 | 167 |
| oracle_qadas_L12 | 6 | 23 | 169 |
| oracle_qadas_L20 | 7 | 23 | 169 |
| oracle_qadas_L29 | 5 | 23 | 169 |
| truthful_swap_qadas_L29 | 6 | 23 | 169 |

这里只是固定候选的调参诊断，各规则自然实质作答数另报；未执行同实际作答率配额比较，不得把此表当作已完成老师的同率主实验。oracle_* 与 gold_source_only 是看答案的诊断上界，不能当作可部署方法。


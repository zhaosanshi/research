# Qwen 全七组统一 256 预算：守卫、质量和证据审计

Qwen seven-arm uniform 256-token diagnostic; Llama full-256 was not run because its pilot failed
原48预算结果保留；新完整尾巴逐条重判。这里先报实际生成置信，统一原模型分须另有真实重算产物，不能沿用截断答案分。
证据审计612/614合规，两条完整列表仍引用改写而未决。它们明确是实质尝试，仅枚举A/B的4种组合，不把它们当拒答改交付资格。下列范围不是置信区间；裁判本身仍可能错。

| 视图 | 方法 | 自然答对/实答 | 各自生成置信25/50/75%答对 |
|---|---|---|---|
| legacy_guard | baseline | 51/116 | 28/45/不可行 |
| legacy_guard | source_only | 50/111 | 27/49/不可行 |
| legacy_guard | equal_logit_mix | 49/113 | 28/46/不可行 |
| legacy_guard | source_das_swap | 50/116 | 28/44/不可行 |
| legacy_guard | source_random_swap | 51/116 | 26/45/不可行 |
| legacy_guard | trained_source_swap_das_init | 86/186 | 24/50/71 |
| legacy_guard | pair_preference_source_swap | 104/177 | 37/63/92 |
| legacy_guard_complete | baseline | 51/116 | 28/45/不可行 |
| legacy_guard_complete | source_only | 50/111 | 27/49/不可行 |
| legacy_guard_complete | equal_logit_mix | 49/113 | 28/46/不可行 |
| legacy_guard_complete | source_das_swap | 50/116 | 28/44/不可行 |
| legacy_guard_complete | source_random_swap | 51/116 | 26/45/不可行 |
| legacy_guard_complete | trained_source_swap_das_init | 86/186 | 24/50/71 |
| legacy_guard_complete | pair_preference_source_swap | 104/177 | 37/63/92 |
| blind_evidence | baseline | 46/117 | 26/40/不可行 |
| blind_evidence | source_only | 47/111 | 28/46/不可行 |
| blind_evidence | equal_logit_mix | 44/114 | 28/41/不可行 |
| blind_evidence | source_das_swap | 45/117 | 27/40/不可行 |
| blind_evidence | source_random_swap | 46/117 | 25/40/不可行 |
| blind_evidence | trained_source_swap_das_init | 76/187 | 24/45/63 |
| blind_evidence | pair_preference_source_swap | 101–103/177 | 34/62/88–89 |
| blind_evidence_complete | baseline | 46/117 | 26/40/不可行 |
| blind_evidence_complete | source_only | 47/111 | 28/46/不可行 |
| blind_evidence_complete | equal_logit_mix | 44/114 | 28/41/不可行 |
| blind_evidence_complete | source_das_swap | 45/117 | 27/40/不可行 |
| blind_evidence_complete | source_random_swap | 46/117 | 25/40/不可行 |
| blind_evidence_complete | trained_source_swap_das_init | 76/187 | 24/45/63 |
| blind_evidence_complete | pair_preference_source_swap | 101–103/177 | 34/62/88–89 |

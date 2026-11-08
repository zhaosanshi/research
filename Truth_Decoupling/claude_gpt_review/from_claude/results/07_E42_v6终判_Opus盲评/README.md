# 07 E42 两批 v6 终判（14f204f 快照重新聚合）+ Opus 5.5 盲评 + 两口径表（2026-11-08 20:30–21:15）

服务器原件：`results_e42_20261108/llama3.1-8b/{oracle-tune-j6,select-tune-v2}/judge2/` 与 `judge2_shared/`。这里只放摘要与清单，不放 2304 / 1536 行的 `final.jsonl`（GPT 在服务器上可直接读）。

| 文件 | 内容 |
|---|---|
| `oracle-tune-j6/judge2/summary.{md,json}`、`select-tune-v2/judge2/summary.{md,json}` | 合票汇总：格数、各来源终判数、裁判一致率、待人工 0 |
| `*/judge2/final_manifest.json` | 终判清单的来源绑定（cells / 三裁判 / 共享人判 / 共享缓存 / 冲突文件的 sha） |
| `*/judge2/e42_analysis_two_axis_v6.md` | `e42_analysis.py --two-axis` 的两维表（神谕 11 臂；选答 13 条规则） |
| `*/judge2/human_queue.jsonl` | 送人（Opus）队列：必判 + 抽检 |
| `select-tune-v2/field_standard_by_rule.md` | **两口径并列**：我们的 A（v6 终判）对 AllenAI 两裁判 yes/yes（领域口径），逐规则、配对自举对 greedy；附 greedy / verdict 选中的全部 H 格原文 |
| `H类回答全清单_两批.txt` | 两批所有终判 H 的（题，回答，金标）去重清单——用来看"金标式纠偏句被判 H"的问题 |
| `judge2_shared/blind/blind2_manifest.json` | v2 盲评包：50 格（每批 25，seed 2），排除所有此前带票展示过的内容（96 条），冻结机器终判 |
| `judge2_shared/blind/blind2_verdicts.json` | 盲评判决账本（`--by opus-5.5-subagent`） |
| `judge2_shared/blind/blind2_opus_reasons.md` | Opus 5.5 子代理每格一句理由（含它查的事实来源） |
| `judge2_shared/blind/blind2_report.json` | 盲评一致率：对 AllenAI / Pro / GPT / 机器终判 |
| `judge2_shared/blind/must_judge_u.json` | 3 个必判输入（U1–U3）与对应格 |
| `judge2_shared/human_verdicts.jsonl` | 共享人判文件（内容指纹绑定；全部来自 Opus 子代理） |
| `judge2_shared/cross_run_check_final.json` | 跨批一致核查：1056 输入、332 共同、0 冲突、0 未决 |

一句话结果：神谕臂 ≈ 基线（消极，事先声明）；选答 verdict 规则 A +10 对 greedy，区间含 0（事先标准未达）；领域口径 +18 显著但差额是含糊答；盲评 Opus 对机器终判一致 0.86（κ .79）。详见 `from_claude/19`。

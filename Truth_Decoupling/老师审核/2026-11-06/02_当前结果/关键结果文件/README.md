# 关键结果文件（副本）

原件在 [`claude_gpt_review/from_claude/results/02_问答DAS_v1/`](../../../../claude_gpt_review/from_claude/results/02_问答DAS_v1/)，这里只复制最常看的报告，方便直接打开。逐题数组（`.npz`）、逐题生成与裁判记录只在原件目录。

| 文件 | 内容 | 原件 |
|---|---|---|
| `真假判断_<模型>/report_2x2_das_tfce.md` | 正式 2×2：E = 新 DAS 单层，C 对同层新 E 正交；两种 C 都列 | `第16轮_DAS正结果搜索/tfce/<模型>/E23/` |
| `真假判断_<模型>/compare_das_tfce_vs_{meandiff,das_tf}_full.md` | 新 DAS 对均值差、对旧 DAS 的配对比较 | 同上 |
| `真假判断_<模型>/knob_sweep_das_tfce_{full,C}.md` | E、C 全层扫描每个条件的调参 / 测试数字 | 同上 |
| `真假判断_llama3.1-8b/compare_oneway_vs_{bidir,kl}.md` | 拆分对照 | `第16轮_DAS正结果搜索/tfce1/llama3.1-8b/E23/` |
| `sweep_dissociation.json` | 不经规则的全扫描计数 | `第16轮_DAS正结果搜索/measurements/` |
| `自由问答_<模型>/v1_summary.txt` | 问答 DAS v1 全链汇总（代理、调参生成、冻结、测试、诊断） | `第16轮_全量/<模型>/summary.txt` |
| `自由问答_<模型>/新用户轮_summary.txt` | 新用户轮版全链汇总 | `第16轮_DAS正结果搜索/v1u/<模型>/summary.txt` |

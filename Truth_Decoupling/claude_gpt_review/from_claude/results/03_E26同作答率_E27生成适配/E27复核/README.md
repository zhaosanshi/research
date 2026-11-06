# E27 独立复算（2026-11-06）

对 GPT 第 18 轮交付的 E27 生成适配结果做的独立复算，由 Claude 派出的子代理用自写代码完成，不调用 GPT 的复算脚本。结论见 `../../../13_审核GPT第18轮_E26同作答率_E27生成适配.md` 第 3 节 A、B。

| 脚本 | 输出 | 内容 |
|---|---|---|
| `a1_integrity.py` | `a1_integrity.json` | 两个归档 194 个文件逐个核对 sha256 与大小 |
| `a2_recompute*.py`、`a2b`、`a2c`、`a2d` | `a2_recompute.json`、`a2c_v1u_compare.json` | 测试 / 调参各表、转移、配对区间、精确检验重算；原模型与原 DAS 输出和 v1u 文件逐题一致 |
| `a3_selection.py` | `a3_selection.json` | 冻结时间线与选择文件哈希 |
| `a4_overlap.py` | `a4_overlap.json` | 训练题与调参 / 测试题的编号与文本重叠（编号 0 重叠；25 对同模板题，不贡献增益） |
| `a5_quality.py`、`a5b_style.py` | `a5_quality.json`、`a5_quality_items.json`、`a5b_style.json` | 乱码（U+FFFD）、未结束、重复、长度；按质量规则重计答对 |
| `a6_judge_flags.py` | `a6_judge_flags.json` | 按 GPT 自己的人工标注翻转裁判标签后的净变化 |
| `a8_pilots.py` | `a8_pilots.json` | E26 三题试点报告与原始文件对照 |

脚本读取的是解压到本机临时目录的 GPT 归档（`from_gpt/main_experiment_20261106/artifacts/*.tar.gz`），复跑前把脚本开头的路径改成你的解压位置。用发布版代码重跑 E27 的结果在同级的 `../E27_release/`。

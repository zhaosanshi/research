# GPT 第 19 轮复核（2026-11-07）

对 GPT 第 19 轮（edcd02a）E27 自查的复核，结论见 [`../../15_审核GPT第19轮_E27自查_复现与分歧.md`](../../15_审核GPT第19轮_E27自查_复现与分歧.md)。

| 内容 | 文件 |
|---|---|
| **服务器复现比对**：Claude 用 GitHub 代码在服务器新目录重跑 GPT 本轮的 8 组 GPU 诊断，再与 GPT 的产物比对 | `r19_reproduction_comparison.json`：三份轨迹 / 剂量 JSON 逐字节相同；五组生成逐题比对原文与结束标记，0 处不同。由 `code/scripts/review_checks/r19_compare_outputs.py` 生成 |
| 复现用的启动脚本 | `code/scripts/review_checks/r19_reproduce.sh`、`r19_reproduce_full.sh`、`r19_reproduce_llama.sh`（只生成、不判分、不训练） |
| 张量核查（服务器 CPU） | `code/scripts/review_checks/r19_compare_adapters.py`（早期与发布版适配器逐张量比较）、`r19_random_init_check.py`（随机初值逐位核对）、`r19_angle_check.py`（训练前后转角） |
| **独立复算**（子代理用自写代码，不调用 GPT 的复算脚本） | `recompute/scripts/c0–c14_*.py` 与 `run_all.sh`，输出在 `recompute/outputs/*.out` |
| **代码红队的小核查** | `code_review/{judge_flip,edit_magnitude,bf16_check}.py`：判分缓存翻转、各臂实际写入量、bf16 写回反例 |

独立复算脚本读的是解压到本机临时目录的 GPT 归档和 Claude 的发布版重跑结果，路径写在 `recompute/scripts/common.py` 的 `SP` 里，复跑前要改成你自己的解压位置。归档本身在 `from_gpt/audit_20261106_round19/` 和 `from_gpt/main_experiment_20261106/artifacts/`。`safe_pt.py` 用白名单反序列化直接读 `.pt` 里的 float32 数据，不需要 torch。

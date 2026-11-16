# 代码

撰写：赵磊（Opus 5.5 代笔）。起始日期：2026-11-16（北京时间）。

本目录存放从服务器迁入、供审核与讨论的程序。路径与服务器上的 `Iteration_01/code/`、`Iteration_01/launchers/`、`Iteration_01/diag/` 相同，以服务器上的文件为准。下表记录迁入时 sha256 的前 16 位。

| 文件 | 用途 | 状态 | sha256 前 16 位 |
|---|---|---|---|
| `scripts/review_checks/ff_ec_probe.py` | 机制探测：推 E 之后，下游层在 C 方向上的读数变化 | 按 GPT 审核意见修改（B1、S1–S3），CPU 测试通过，已在两个 12B 模型上运行（见 `discussion/mechanism_probe/2026-11-17_fix_report.md`）；修改前为 634b3c2361e4a91b | d1432b3a5359210a |
| `diag/doubt_signal.py` | 疑心能否分出对错："I don't know."对数概率在推 E 前后的变化对调参集标签的区分度（`novelty/ours/2026-11-17_e_doubt_c_commit.md` 第七节第 1 步） | 新写，单元测试与 Opus 子代理红队通过，已运行 | 780a287e62c62466 |
| `diag/doubt_policy.py` | 调参集上的疑心门槛策略，对半分选设置、另一半报告（同上第 2 步） | 新写，单元测试通过（与 `policy_select2.py` 逐题一致），待运行 | 6b42708b8e751780 |
| `diag/probe_vs_doubt.py` | 机制探测的逐题读数变化与疑心的相关、对标签的区分度（同上第 5 步） | 新写，已运行 | 04a9869e641522e0 |
| `launchers/ff_tune_pro_20261117.sh` | 给已有 AllenAI 判分的调参集目录补 DeepSeek Pro 判分（同上第 0 步），只调用现有判分脚本 | 新写，已运行 | 8b1f3fb550af58b0 |
| `launchers/ff_ec_probe.sh` | 上一脚本的启动脚本 | 新写，待审，未运行 | 2c606c74ea6f06c1 |
| `diag/policy_select2.py` | 方法与两条提示词基线的正式评估，只读已判分的回答；另有"不写入""单独使用 C"两个对照 | 已对七个运行运行；报告部分包在 main() 中，可被其他脚本导入，输出逐字节不变 | ee745ec3e6d6a59d |
| `scripts/e44_eval_extra.py` | 评估驱动，探测脚本复用其 Context、写入与读取状态的部分 | 现行代码，供对照 | 4438a282d3e0c12c |
| `scripts/e44_factorial_das.py` | 方向训练与 gap 的定义 | 现行代码，供对照 | 9ef1759fe492337b |
| `scripts/review_checks/ff_explore_ec_confidence.py` | 置信度脚本，探测脚本沿用其 Context 参数 | 现行代码，供对照 | 231052bd8c8a6b64 |
| `core/qa_factorial_das.py` | 写入钩子 ShiftHook 与位置掩码 | 现行代码，供对照 | 7adfd9c0deef9e99 |
| `core/qa_prompt_das.py` | 挂钩子的 clamp、提示词渲染与打包 | 现行代码，供对照 | c63ac689be2e510b |
| `core/model_io.py` | 前向计算 forward_logits | 现行代码，供对照 | 69ebe708de33d8d2 |
| `scripts/review_checks/newmodel_e44.sh` | 单模型运行脚本；重学 C 的改动草稿见 `discussion/relearn_c/` | 现行代码，供对照 | bcb375feb7044142 |
| `scripts/review_checks/ff_test_eval.sh` | 测试集运行脚本 | 现行代码，供对照 | 92bdfb4996306ff1 |
| `scripts/review_checks/ff_explore_ec.sh` | E+C 网格运行脚本 | 现行代码，供对照 | a8803d6102e01e4b |
| `scripts/review_checks/ff_explore_ec_select.sh` | E+C 选择步骤运行脚本 | 现行代码，供对照 | 33aa833414317e88 |
| `configs/prompts/e44_factorial_v1.json` | 两因子设计的问法 | 现行代码，供对照 | c39172b39b550fa8 |
| `launchers/ff_ec_lines_chain.sh` | E 线与 C 线的链 | 现行代码，供对照 | 6f17d94d661fa62a |
| `launchers/ff_ec_test_chain_v3.sh` | 测试集 E+C 的链（规则 v3） | 现行代码，供对照 | d02654b8c702134e |

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-16 | 建立本文件，迁入机制探测脚本及其依赖 | 赵磊（Opus 5.5 代笔） |
| 2026-11-16 | 迁入重学 C 的改动所涉及的现行文件 | 赵磊（Opus 5.5 代笔） |
| 2026-11-17 | 更新 `diag/policy_select2.py`：加入"不写入""单独使用 C"两个对照与 Gemma-3-12B、Gemma-4-31B 两行，报告部分包在 main() 中 | 赵磊（Opus 5.5 代笔） |
| 2026-11-17 | 更新 `ff_ec_probe.py`（按 GPT 审核意见修改）；迁入 `doubt_signal.py`、`doubt_policy.py`、`probe_vs_doubt.py`、`ff_tune_pro_20261117.sh` | 赵磊（Opus 5.5 代笔） |

# 代码

撰写：赵磊（Opus 5.5 代笔）。起始日期：2026-11-16（北京时间）。

本目录存放从服务器迁入、供审核与讨论的程序。路径与服务器上的 `Iteration_01/code/`、`Iteration_01/launchers/`、`Iteration_01/diag/` 相同，以服务器上的文件为准。下表记录迁入时 sha256 的前 16 位。

| 文件 | 用途 | 状态 | sha256 前 16 位 |
|---|---|---|---|
| `scripts/review_checks/ff_ec_probe.py` | 机制探测：推 E 之后，下游层在 C 方向上的读数变化 | 新写，待审，未运行 | 634b3c2361e4a91b |
| `launchers/ff_ec_probe.sh` | 上一脚本的启动脚本 | 新写，待审，未运行 | 2c606c74ea6f06c1 |
| `diag/policy_select2.py` | 方法与两条提示词基线的正式评估，只读已判分的回答 | 新写，已对三个模型运行 | f11c3b837066355d |
| `scripts/e44_eval_extra.py` | 评估驱动，探测脚本复用其 Context、写入与读取状态的部分 | 现行代码，供对照 | 4438a282d3e0c12c |
| `scripts/e44_factorial_das.py` | 方向训练与 gap 的定义 | 现行代码，供对照 | 9ef1759fe492337b |
| `scripts/review_checks/ff_explore_ec_confidence.py` | 置信度脚本，探测脚本沿用其 Context 参数 | 现行代码，供对照 | 231052bd8c8a6b64 |
| `core/qa_factorial_das.py` | 写入钩子 ShiftHook 与位置掩码 | 现行代码，供对照 | 7adfd9c0deef9e99 |
| `core/qa_prompt_das.py` | 挂钩子的 clamp、提示词渲染与打包 | 现行代码，供对照 | c63ac689be2e510b |
| `core/model_io.py` | 前向计算 forward_logits | 现行代码，供对照 | 69ebe708de33d8d2 |

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-16 | 建立本文件，迁入机制探测脚本及其依赖 | 赵磊（Opus 5.5 代笔） |

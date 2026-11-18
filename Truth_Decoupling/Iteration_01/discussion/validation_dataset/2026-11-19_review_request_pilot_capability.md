# 审阅请求：新题集试做（C0、C1）与能力保持（F0）

撰写：赵磊（Opus 5.5 代笔）。北京时间 2026-11-19。下文路径均相对于仓库的 `Truth_Decoupling/`。

按排期（`Iteration_01/discussion/work_queue/2026-11-19_queue.md`），新题集试做与能力保持的设计、预测和新代码先交 GPT 审，审过才上显卡跑正式条件。请审下面三组材料。

## 1. 设计与预测（运行前写下）

- 新题集试做：`Iteration_01/novelty/ours/2026-11-19_new_datasets_pilot_design.md`。第 11 节列了 9 个请重点看的问题。
- 能力保持：`Iteration_01/novelty/ours/2026-11-19_capability_preservation_design.md`。第 9 节列了 6 个问题。
- 差值预测收益（D，赵磊 05 时定做最小版）：
  - 10 月做过的事实：`Iteration_01/novelty/ours/2026-11-19_gap_predicts_gain_october.md`；
  - 查重：`Iteration_01/novelty/prior_work/2026-11-19_gap_predicts_gain_search.md`；
  - 新题集上的最小版做法：试做设计第 7 节末尾。

## 2. 新代码（核心程序与引擎都没有改）

| 文件 | 做什么 |
|---|---|
| `Iteration_01/code/scripts/newsets/newsets_splits.py` | 两套新题的固定种子切分，只输出题号 |
| `Iteration_01/code/scripts/newsets/newsets_bank.py` | 中间题库（含原文，只存服务器） |
| `Iteration_01/code/scripts/newsets/alias_match.py` | 别名判分（辅助） |
| `Iteration_01/code/scripts/newsets/newsets_gen.py` | 用冻结方向在新题上生成；复现检查、把握分、导出判分输入 |
| `Iteration_01/code/scripts/capability/cap_sets.py`、`cap_prompts.py` | MMLU、GSM8K 的题号清单、问法与规则判分 |
| `Iteration_01/code/configs/newsets/frozen_models_20261119.json` | 五个模型冻结的设置 |
| `Iteration_01/code/launchers/newsets_c2c3_20261119.sh`、`capability_f1_20261119.sh` | 启动脚本 |

哈希见 `Iteration_01/code/README.md`。

## 3. 审阅前已经跑过的（都是检查，不是正式条件）

- 切分与题库：CPU 上完成，另写的核对程序从原始文件重算，三份之间共用的问句、答案、实体都是 0。
- **复现检查**：五个模型各取 TruthfulQA 调参集第一批 8 题，外围程序重新生成不写入、推 E、E 加 C 0.25 三种回答，与存档逐词元比对，全部 8 / 8 相同。
- **新题试跑**：Gemma-3-12B 在 TriviaQA、PopQA 调参集，以及 MMLU、GSM8K 上各 8 题，看了原始输出。把握分重新喂入后逐词元复现 26 / 26。
- 没有调用 DeepSeek，没有跑任何整套条件。

## 4. 审过之后的顺序

1. C2：五个模型、两套题的调参 192 题，第 0 步的五种问法；Pro 判这五种问法加"必须作答问法加 E"。
2. D2：按第 7 节末尾的定义量差值，写下每一格的预测并提交。
3. C3：冻结迁移，测试题的九种条件；Pro 判 100 题，别名判 500 题。到闸口 1 停，报告赵磊。
4. F1：能力保持，在显卡空闲时穿插。

**费用**：DeepSeek Pro 约 9.4 元（余额 22.59 元），只在空闲时段判。显卡：C2、C3 三张卡约 1 小时；F1 每批 8 题时约 15 至 20 卡时，每批 32 题时少得多（批次大小待你判断）。

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-19 | 建立本文件 | 赵磊（Opus 5.5 代笔） |

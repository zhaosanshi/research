# 重学 Gemma-4-12B 的 C：请 GPT 审核

撰写：赵磊（Opus 5.5 代笔）。起始日期：2026-11-16（北京时间）。

本目录是改动草稿，尚未放上服务器。被改动的现行文件见 `Iteration_01/code/`（sha256 见 `Iteration_01/code/README.md`）。审核通过前不改服务器上的文件，也不起跑。

## 1. 目的

- Gemma-4-12B 的 C 学得弱。C 由必须作答问法 conf 与谨慎问法 caut 的对比训练而来（训练集 96 题），这两种问法在 Gemma-4-12B 上的弃答差别很小：调参集 192 题上 conf 弃答 0 题、caut 10 题；训练集 96 题上 caut 4 题、caut 加如实句 6 题。只推 C 往敢答、力度 2 时，测试集答对反而少 29 题。
- 改用加强谨慎句 U1（"Only answer if you are certain that your answer is correct. If you have any doubt, say 'I don't know'."）接在谨慎基底之后重学 C。调参集 192 题上，caut 加 U1 的弃答为 31 题（规则判断）。
- 此前的加强谨慎运行（`results/ff_xu1`）只在普通问法 x 后加了 U1。C 的训练问法不含 x，两次运行的 C 训练记录完全相同，所以 C 实际没有改变。本次运行是第一次改变 C 的训练对比。

## 2. 改动

1. **新问法文件** `configs/prompts/e44_factorial_v1_cu1.json`（草稿：`2026-11-16_e44_factorial_v1_cu1.json`）：在四个谨慎问法 caut、caut_E1、caut_E2、caut_M 的基底之后加 U1，位于 E 句之前，使 caut_E2 = caut + E2，与 conf_E2 = conf + E2 的结构相同；其余与 `e44_factorial_v1.json` 相同。
2. **`newmodel_e44.sh` 新参数 `PROMPTS_E44`**（草稿：`2026-11-16_newmodel_e44.diff`）：给出时，把 `--prompts-e44` 传给 freeze、train、generate、mc 各步；不给时，每条命令与现在逐字相同；STAGE=diag 拒绝此参数。
3. **三个测试脚本**（`ff_test_eval.sh`、`ff_explore_ec.sh`、`ff_explore_ec_select.sh`）各加三行，同样传 `--prompts-e44`（草稿：`2026-11-16_test_runners.diff`）。
- 两个链的启动脚本不改：其 CLEAN 列表都不清除 `PROMPTS_E44`，参数会原样传给运行脚本。

## 3. 运行方式

- 新结果目录 `results/ff_cu1/gemma4/12b`，源目录逐字节拷自普通版运行。
- 问法属于运行身份，第 0 步与 E 必须重做。E 只训第 24 层，与普通版相同；C 训第 15 至 27 层的奇数层。单卡约 3 小时 50 分。
- 第 0 步完成后、C 线开始前检查：
  1. 其余 7 种问法的回答与普通版逐 token 相同；
  2. caut 在调参集的弃答为 31 题；
  3. 训练集上 caut 与 caut_E2 的弃答若仍很少，则在 C 线之前停止。

## 4. 已知的限制

- 普通问法下 Gemma-4-12B 几乎不留空（调参集 8 / 192，测试集 26 / 529）。按规则 v1.1，E+C 很可能仍不加 C，新的 C 主要体现在 C 线的表与 v1 网格中。
- `ff_explore_ec_confidence.py` 写死了原问法文件，相同作答数比较的步骤（sn_*_conf）会失败。这些步骤在 Gemma-4-12B 上原本就失败，且属于已不再追的强形式。
- 链的 `params.txt` 不记录 `PROMPTS_E44`，另存 `logs/prompts_e44.txt`。

## 5. 何时放上服务器

审核通过后，等 Gemma-3-12B 与 Gemma-3-27B 的 E 线、C 线结束，再以先写入新文件、后改名的方式替换 `newmodel_e44.sh`。测试脚本在排队的测试链全部结束后再改，因为测试链每个任务之前都会核对这些文件的校验值。

## 6. 请审核

1. 新问法文件是否只改了谨慎一侧，U1 的位置是否合适。
2. `PROMPTS_E44` 是否传到了所有需要它的调用；不给时，命令是否与现在完全相同。
3. 第 3 节的三项检查是否充分。
4. 其他问题。

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-16 | 建立本文件 | 赵磊（Opus 5.5 代笔） |

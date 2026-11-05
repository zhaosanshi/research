# 问答DAS v1：代码入口与分阶段运行说明

本版主入口为 **`from_claude/code/scripts/e25_qa_das.py`**。DAS（分布式对齐搜索 · 交换内部表示并训练反事实标签读出）是主方法；同题、同答案末位、同中心、同写法的均值差是对照。协议名：`qa_das_ptrue_ce_bidir_v1`。

**当前交付是代码和离线验证，不是新效果结果。以下真实模型命令供Claude审代码后执行，本文未运行。**先逐模型检查三题，不要直接跳到全量。

## 1. 实现的协议

- 原96道估计题不跨集合补题；按实质参考答案清理，当前为92对。调参192题，测试529题；试点只截各集合前三题。
- 先固定原问答＋答案的词元ID，再独立编码自评后缀。两种方法共用同一批在答案末位取得的状态；后缀和标签不计入答案区间。
- 双向金标交叉熵，秩1、底座冻结、种子42、10轮、学习率1e−3、每对更新、裁剪1。训练日志记录首对位置与实际写入位移。
- DAS逐层训练`0..L−2`；末块输出的早期答案位置不能影响后面的自评标签，记录为结构空对照。均值差保留全部层。试点可以明确指定一个可训练层。
- 主干沿模型配置精度，最后输出投影统一用FP32（32位浮点 · 避免先低精度投影、再转高精度的无效修法）；不改变共享嵌入权重，保留模型自己的归一化与输出后处理。
- 配对答案评分为整段条件对数概率求和，正负答案交错、同对同批，保存分差与词数；不冒充官方多选指标。
- 主选择固定连续写入；每种方法按代理分数保留3个不同层，再用真实自由回答选择。候选须正确率增量>0、作答率变化≤3个百分点且判定完整；同分先保护原对答案，再小强度、浅层。
- 金标来源坐标替换单独标为诊断，连续使用同一个来源标量，交换系数固定1；它不参与选择，不称可部署效果或数学上限。

自评后缀与标签已冻结在 `configs/experiments/E25_qa_das_v1.json`，标签前导空格不可删除。模型实际编码仍动态核验，不硬编码词元编号。

## 2. 文件和阶段

| 阶段 | 产物／作用 | 会做什么 |
|---|---|---|
| `estimate` | `questions.json`、`bank.npz/json` | 冻结题集、配置和词元签名；估均值差、两类中心、DAS来源状态缓存 |
| `train` | `das/*.pt`、`training.json` | 逐层训练DAS；已有同协议检查点校验后跳过，支持逐层续跑 |
| `check` | `check_selfcheck.json`、`check_generations.json` | 最多3道调参题：双向交换／放大、写入位移、同隐藏状态的两种投影、单条与成批隐藏状态差异、实际生成 |
| `judge-check` | `check_scored*` | 单独调用语义裁判，小样本只用于检查线路 |
| `tune-e` | `proxy_scores.json`、`shortlist.json` | 所有规定层／力度／写法的配对评分，固定Top-3不同层入围规则 |
| `generate-tune` | `tune.json` | 6个入围配置＋基线在调参题上真实生成 |
| `judge-tune` | `tune_scored.json`、`tune_scored_items.json` | 当前语义规则评分，保存逐题判定 |
| `freeze` | `selection.json` | 验证来源和判分一致性，按实际正确／错误／弃答选择；不依赖C实验 |
| `generate` | `test.json` | 基线＋各自冻结的DAS/均值差配置，测试集不回头选参数 |
| `judge-test` | `test_scored*` | 主表、按题区间和3×3状态转换 |
| `diagnose` | `diagnostic_selfcheck.json`、`tune_diagnostics.json` | 冻结工作点的仅提示写入／金标来源替换；复用同源调参基线与主臂文本 |
| `judge-diagnostics` | `diagnostics_scored*` | 独立诊断报告，明确标注不可参与主选择 |

每次只执行一个阶段，没有自动启动后续训练或裁判的总开关。同一模型的一个run目录只使用一个写入进程；不同模型用不同目录。配置、题目、词元或训练权重改变，不能沿用旧缓存接着算。

## 3. 审核后先执行三题

进入新代码中包含`core/`、`configs/`、`scripts/`的目录，在原实验容器环境设置现有模型和数据路径：

```bash
export TD_MODEL_ROOT=/workspace/Truth_Decoupling/models
export TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed
export TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_qa_das_v1

TD_QA_MODEL=llama3.1-8b
TD_QA_RUN="$TD_RESULTS_DIR/$TD_QA_MODEL/pilot3_v1"
python3 scripts/e25_qa_das.py estimate --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN" --layers 11 --pilot 3
python3 scripts/e25_qa_das.py train --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
python3 scripts/e25_qa_das.py check --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
```

第11层只是Llama线路核验起点，不是问答最优层。Qwen可用第18层、Gemma-4B可用第11层作同样核验；都不是最终选层。

先逐条看原始提示与输出、`answer_end`和标签预测位、双向训练目标、`kept_delta_norm`及实际生成写入次数。`check`固定用配置中的`check_alpha=1`，不拿三题挑力度。精度诊断明确比较原始输出投影，不把它当成已定位完整模型漂移原因。

生成无异常后再单独评分：

```bash
python3 scripts/e25_qa_das.py judge-check --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
```

裁判配置沿用 `E4_qa`；凭证继续由既有环境提供，不写入脚本或GitHub。

所有判分阶段都要求题目和条件判齐。出现未判定时，命令返回失败，摘要标为`completed=false`，逐题进度保存在`*_grading_input.json`；此时没有可引用的成绩主表。重跑同一个判分命令补齐未判项，已有有效判定继续复用，不换裁判、不删除未判题改变分母。

## 4. 小样本通过后才建全量目录

不用试点目录改参数续跑。新建全量目录，并显式要求全部层：

```bash
TD_QA_RUN="$TD_RESULTS_DIR/$TD_QA_MODEL/full_v1"
python3 scripts/e25_qa_das.py estimate --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN" --all-layers
python3 scripts/e25_qa_das.py train --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
python3 scripts/e25_qa_das.py tune-e --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
python3 scripts/e25_qa_das.py generate-tune --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
python3 scripts/e25_qa_das.py judge-tune --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
python3 scripts/e25_qa_das.py freeze --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
```

每个命令是一个独立阶段，先检查上一阶段产物再继续。没有合格方向时选择为`null`；不会用违规候选、随机初始化方向或另一估计器偷偷顶替DAS。

冻结后再执行主测试和另列诊断：

```bash
python3 scripts/e25_qa_das.py generate --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
python3 scripts/e25_qa_das.py judge-test --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
python3 scripts/e25_qa_das.py diagnose --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
python3 scripts/e25_qa_das.py judge-diagnostics --model "$TD_QA_MODEL" --run-dir "$TD_QA_RUN"
```

冻结后禁止改调参裁判结果再沿用选择。测试生成前重新校验全部冻结来源，并重算选择规则检查`selection.json`没有被手工替换。

## 5. 预算上限与判读

| 模型 | DAS可训练层上限／均值差层上限 | 代理评分条件上限（含基线） |
|---|---:|---:|
| Llama-3.1-8B | 31／32 | 379 |
| Qwen3-8B | 35／36 | 427 |
| Gemma-3-4B | 33／34 | 403 |

调参主生成最多1344条／模型，额外诊断最多768条；测试主表最多1587条／模型。零均值差方向会明确记不可用；DAS无梯度或训练异常会报错，不能作为有效检查点。试点和无合格配置时条数相应减少。

报告称“**逐层训练及代理筛选后，限定候选中的自由生成最优配置**”。正确率、错误率、弃答率、作答率、原对→错及原对→弃答必须一起看；只看净提升不能证明保护原本会的题。当前单裁判开发验证不替代正式论文双裁判与人工核验。

## 6. 旧入口与检查

`e25_qa_context.py`保留为旧的均值差上下文对照，已补 `freeze --target E`、模板/种子校验、冻结来源校验和E24原提示。新DAS主线使用本说明的`e25_qa_das.py`，不要把旧脚本误认为已含DAS训练。

在项目目录复跑离线检查：

```bash
python3 -m unittest discover -s claude_gpt_review/from_gpt -p 'test_*.py' -q
```

新测试使用冻结随机小网络与本地模拟裁判，覆盖完整训练到测试生成流程和来源篡改拒绝；没有外部模型调用，不构成科学效果证据。

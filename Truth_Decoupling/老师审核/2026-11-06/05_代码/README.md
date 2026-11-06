# 05 代码

**完整可运行代码**：[`claude_gpt_review/from_claude/code/`](../../../claude_gpt_review/from_claude/code/)（`core/` 核心模块、`scripts/` 入口与链、`configs/` 实验配置）。本目录 [`快照/`](快照/) 只是当前主结果用到的关键文件副本，方便直接阅读；以完整树为准。

## 1. 核心模块

| 文件 | 作用 |
|---|---|
| `core/directions.py` | 三种方向估计器：均值差、逻辑回归探针、秩 1 DAS。DAS 训练目标可选 `loss="kl"`（round 4：多步匹配模型自己的续写）、`"jsd"`、`"ce"`（11-05 新目标：源陈述金标签、标签各写法首词元概率之和的交叉熵、提示末位一步，可双向）；方向加载器 |
| `core/intervene.py` | 干预算子：沿方向放大（E，围绕中心点 `h + α·WWᵀ(h − c)`）与加法（C，`h + α·gap·w`） |
| `core/readout.py` | 标签词元读出（True / False、Yes / No 各写法合并） |
| `core/metrics.py` | J、AUROC、偏置；精确整数选择键（避免浮点并列误选） |
| `core/qa_das.py` | 问答上下文 DAS（GPT 第 16 轮交付）：答案末位交换写入、自检监督、`selfcheck_format="user_turn"`（11-05 新增） |
| `core/model_io.py` | 模型加载与激活采集 |

## 2. 结果 → 脚本 → 命令

| 结果（`02` 中的位置） | 链 / 脚本 | 关键参数 |
|---|---|---|
| 新 DAS 训练（第 2 节） | `scripts/review_checks/tfce_chain.sh` 第 1 步 → `scripts/e1_directions.py` | `--exp E1_directions_tqa --direction-prompt tf --das-anchor prompt --das-tags E --das-pairs 96 --das-all-layers --das-loss ce --das-bidir` |
| E 行全层扫描（第 2 节） | 同链第 2 步 → `scripts/e23_knob_sweep.py` | `--exp E2b_tqa --all-layers --tune-pairs 192 --skip-cp --no-cs --e-estimator das --das-variant tf-ce-bidir --tag das_tfce_full` |
| 与旧 DAS、均值差配对比较（第 2 节） | 同链第 3 步 → `scripts/e23_compare_estimators.py` | 要求两份扫描基线数组逐位相同，否则拒绝出表 |
| 拆分对照（第 2 节） | `scripts/review_checks/tfce1_chain.sh` | 单向金标签交叉熵 |
| 三层窗口（第 2 节） | `scripts/review_checks/tfce_spans.sh` | 与 round 4 相同的候选窗口 |
| C 重扫与 2×2（第 3 节） | `scripts/review_checks/tfce_C.sh` → `e23_knob_sweep.py`（C 对同层新 DAS-E 正交）→ `scripts/report_2x2.py` | 选择规则写在 `report_2x2.py` 开头 |
| 全扫描计数（第 3 节） | `scripts/review_checks/sweep_dissociation.py` | 不用任何规则，测试集全部条件 |
| 自由问答 v1（第 4 节） | `scripts/e25_qa_das.py` + `configs/experiments/E25_qa_das_v1.json`；启动 `review_checks/full_v1*.sh`；汇总 `review_checks/summarize_qa_das_v1.py` | 冻结协议 `qa_das_ptrue_ce_bidir_v1`；DeepSeek 语义裁判 |
| 自由问答新用户轮（第 4 节） | 同上 + `E25_qa_das_v1_userturn.json`；启动 `review_checks/full_variant.sh` | 只改自检格式 |
| 读出格式（第 4 节） | `review_checks/readout_formats.py` | 无干预 |
| 选择题、一正一负偏好（第 5 节） | `review_checks/heldout_mc.py`、`heldout_pairs.py` | 与 v1 代理同一打分路径 |
| 认错题（第 5 节） | `review_checks/own_answer_detect.py`（问答坐标）、`own_answer_tfdetect.py`（TF 坐标；11-06 起另存逐题载荷） | 层与符号在调参集上定 |
| 核验后弃答（第 5 节） | `review_checks/verify_own_answers.py` | 11-06 修复方向缓存键（见 `06`） |

### 老师的主实验（E26）与生成适配（E27），对应 `08`

| 结果 | 代码 | 谁写的 / 审核状态 |
|---|---|---|
| E26 相同作答率主实验 | `scripts/e26_matched_coverage.py`（prepare / collect-tune / judge-tune / freeze / collect-test / judge-test / report 七个独立阶段）、`core/joint_steering.py`（C 的估计与 E、C 同时写入）、`core/selective_eval.py`（配额选择、自举与检验，不依赖模型）；启动 `review_checks/e26_full_chain.sh` | 实验代码 GPT 第 18 轮交付，Claude 开跑前审过（`13`）；启动脚本由 Claude 写，待 GPT 审 |
| C 剂量探测 | `review_checks/e26_c_dose_probe.py` | Claude 写，待 GPT 审 |
| 开跑前声明的主分析 | `review_checks/e26_primary_analysis.py`（在任何测试结果出来之前提交，`ca67e33`） | Claude 写，待 GPT 审 |
| 结果核查（不导入 E26 代码的独立重算） | `review_checks/e26_independent_recount.py` | Claude 写，待 GPT 审 |
| E27 生成适配 | `scripts/e27_qa_generation.py`、`core/qa_generation_adapter.py`；发布版重跑 `review_checks/e27_release_rerun.sh`；质量规则汇总 `e27_quality_summary.py`；带适配器的 E26 分析 `e26_adapter_analysis.py` | 适配器代码 GPT 第 18 轮交付；Claude 的脚本待 GPT 审 |
| GPT 第 19 轮的 E27 自查 | `review_checks/e27_{frozen_initial_controls,trace_corruption,current_dose,generation_dose,pure_das_layer_controls}.py`；证据在 `from_gpt/audit_20261106_round19/` | GPT 写；Claude 尚未复核 |

## 3. 测试与检查

| 检查 | 范围 | 结果 |
|---|---|---|
| GPT 离线回归 `claude_gpt_review/from_gpt/test_*.py` | 10 个文件 116 项：问答管线、训练、选择、精度、裁判、旧 E 路径 | 全部通过（11-05 新代码改后复跑） |
| `review_checks/chk_das_ce.py` | 新交叉熵目标 11 项：零交换时损失等于手算、梯度非零、标签交换改变损失、缺标签 / 重叠标签被拒、默认 KL 路径不变等 | 11 / 11 |
| `review_checks/chk_verify_cache.py` | 核验脚本方向缓存：新旧 DAS 同层各得其方向 | 6 / 6 |
| GPT 第 17 轮审计脚本 `from_gpt/audit_20261106_round17/` | 从逐题数组独立复算真假判断主扫描（E、C）与配对比较、问答两版六条链与附加评估、训练检查点；未覆盖的项目见 `06` 第 3 节 | 见 `06` |

## 4. 运行环境

- 服务器：4 卡；Docker 容器内运行；环境变量 `TD_RESULTS_DIR`、`TD_DATA_DIR`、`TD_MODEL_ROOT` 指向结果、数据、模型目录；Hugging Face 离线模式。
- 模型：Llama-3.1-8B-Instruct、Qwen3-8B、Gemma-3-4B-it（本地权重，bfloat16，不在仓库；问答 DAS 的末端投影另用 fp32）。
- 数据：TruthfulQA 的处理版本（陈述改写与 96 / 192 / 529 切分，种子 42），在服务器上，不在仓库。
- 裁判：DeepSeek，判分缓存随结果目录发布；API 密钥只在服务器上，不在仓库。
- 随机性：切分、自举、训练种子固定；同一输入的前向可复现（11-06 重跑认错题与原结果逐位一致）。

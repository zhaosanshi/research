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

### 博士的主实验（E26）与生成适配（E27），对应 `08`

| 结果 | 代码 | 谁写的 / 审核状态 |
|---|---|---|
| E26 相同作答率主实验 | `scripts/e26_matched_coverage.py`（prepare / collect-tune / judge-tune / freeze / collect-test / judge-test / report 七个独立阶段）、`core/joint_steering.py`（C 的估计与 E、C 同时写入）、`core/selective_eval.py`（配额选择、自举与检验，不依赖模型）；启动 `review_checks/e26_full_chain.sh` | 实验代码 GPT 第 18 轮交付，Claude 开跑前审过（`13`）；启动脚本由 Claude 写，待 GPT 审 |
| C 剂量探测 | `review_checks/e26_c_dose_probe.py` | Claude 写，待 GPT 审 |
| 测试前声明的分析计划及分析脚本 | 计划在 `66868ab`（18:27，测试收集 18:42 起）；`review_checks/e26_primary_analysis.py` 于 `ca67e33`（19:17）提交，此时真假判断测试文件已生成、问答判分尚未完成；规则与计划一致，未按结果改过 | Claude 写；GPT 第 20 轮核对了提交内容与服务器时间线 |
| 结果核查（不导入 E26 代码的独立重算） | `review_checks/e26_independent_recount.py` | Claude 写；GPT 第 20 轮复算同一批一致 |
| E27 生成适配 | `scripts/e27_qa_generation.py`、`core/qa_generation_adapter.py`；发布版重跑 `review_checks/e27_release_rerun.sh`；质量规则汇总 `e27_quality_summary.py`；带适配器的 E26 分析 `e26_adapter_analysis.py` | 适配器代码 GPT 第 18 轮交付；Claude 的脚本待 GPT 审 |
| GPT 第 19 轮的 E27 自查 | `review_checks/e27_{frozen_initial_controls,trace_corruption,current_dose,generation_dose,pure_das_layer_controls}.py`；证据在 `from_gpt/audit_20261106_round19/` | GPT 写；Claude 已用 GitHub 版在服务器重跑复现（`15`） |
| 第 19 轮的复现与比对 | `review_checks/r19_reproduce.sh`、`r19_reproduce_full.sh`、`r19_reproduce_llama.sh`（重跑 GPT 的 8 组 GPU 诊断，只生成、不判分）；`r19_compare_adapters.py`（早期与发布版适配器逐张量比较）；`r19_random_init_check.py`（随机初值逐位核对） | Claude 写；GPT 第 20 轮审后改成了"只能用于未来运行"的版本（原版在提交 `1ffaedf`） |

### E 逐层表与 C 严格性表，对应 `10`

| 结果 | 代码 | 谁写的 / 审核状态 |
|---|---|---|
| E 逐层 / C 逐层 / 最优层 C 全部影响的三张表 | `review_checks/tf_layer_tables.py`（读 2×2 的 `knob_sweep_das_tfce_{full,C}.npz`，测试半，按题对自举 2000 次） | Claude 写（11-07），待 GPT 审 |
| 每个强 C 条件的等价边界 | `review_checks/tf_c_equivalence.py`；逐层 E 与严格 C 的汇总 `tf_e_layers_c_strict.py` | 同上 |

### 按赵磊口径重做的 C+E（E41），对应 `09`

| 结果 | 代码 | 谁写的 / 审核状态 |
|---|---|---|
| E41：2×2 验证过的 C 固定 × E 全部层 | `scripts/e41_fixed_c_all_e.py`（与 `e23_knob_sweep.py` 同一方向估计路径，新增保存门控分；共享条件须逐位复现 2×2）；启动 `review_checks/e41_chain.sh MODEL GPU [pilot]`；预声明分析 `review_checks/e41_analysis.py` | Claude 写，分析先于结果提交（`003627e`）；待 GPT 审 |

### GPT 第 20 / 21 轮新增的代码（GPT 写；Claude 尚未复核）

E26 修正版配置（`E26_tf_ability.json`、`E26_qa_ability.json`）、E28 原生门控交付、E29 / E30 来源状态交换与训练、E31 预设 C 层诊断、E32 来源方向的同作答率评估、E33 / E35 门控提示诊断、E34 配对偏好损失、E36 尺度匹配随机对照、E37 / E38 固定 E 的 C 层序、E39 / E40 邻层 C 功能检查与扩展；对应核心模块 `core/qa_source_*.py`、`core/qa_norm_matched_random.py`、`core/experiment_provenance.py` 等。协议与证据见 GPT [第 20 轮](../../../claude_gpt_review/from_gpt/review_20261107_20.md)、[第 21 轮](../../../claude_gpt_review/from_gpt/review_20261107_21.md)。E27 生成适配（`e27_qa_generation.py`、`qa_generation_adapter.py`）已停用，代码保留作记录。

## 3. 测试与检查

| 检查 | 范围 | 结果 |
|---|---|---|
| GPT 离线回归 `claude_gpt_review/from_gpt/test_*.py` | 第 17 轮 116 项 → 第 18 轮 193 项 → 第 19 轮 219 项（Claude 在服务器容器复跑通过）→ 第 21 轮 426 项（GPT 自报，本机与服务器各通过） | 通过 |
| E41 共享条件复现 | 原模型、E 单独、C 单独与 2×2 扫描文件逐位比较 | Llama 48 条试点 67 / 67 相同；全量运行中逐条检查 |
| `review_checks/chk_das_ce.py` | 新交叉熵目标 11 项：零交换时损失等于手算、梯度非零、标签交换改变损失、缺标签 / 重叠标签被拒、默认 KL 路径不变等 | 11 / 11 |
| `review_checks/chk_verify_cache.py` | 核验脚本方向缓存：新旧 DAS 同层各得其方向 | 6 / 6 |
| GPT 第 17 轮审计脚本 `from_gpt/audit_20261106_round17/` | 从逐题数组独立复算真假判断主扫描（E、C）与配对比较、问答两版六条链与附加评估、训练检查点；未覆盖的项目见 `06` 第 3 节 | 见 `06` |

## 4. 运行环境

- 服务器：4 卡；Docker 容器内运行；环境变量 `TD_RESULTS_DIR`、`TD_DATA_DIR`、`TD_MODEL_ROOT` 指向结果、数据、模型目录；Hugging Face 离线模式。
- 模型：Llama-3.1-8B-Instruct、Qwen3-8B、Gemma-3-4B-it（本地权重，bfloat16，不在仓库；问答 DAS 的末端投影另用 fp32）。
- 数据：TruthfulQA 的处理版本（陈述改写与 96 / 192 / 529 切分，种子 42），在服务器上，不在仓库。
- 裁判：DeepSeek，判分缓存随结果目录发布；API 密钥只在服务器上，不在仓库。
- 随机性：切分、自举、训练种子固定；同一输入的前向可复现（11-06 重跑认错题与原结果逐位一致）。

# MANIFEST — 3.11_转写_E14

生成时间：2026-11-04 07:58 PST。来源标记：`local:` = 本机项目 `/Users/airsanshi/MyPapers/Truth_Decoupling`；`pro5000:` = 服务器 `/data/sanshizhao/Truth_Decoupling`（布局与本机相同，`code/` 对应本机根目录）。
文件 204 个，合计 54.7 MB。单文件 > 20 MB 不复制（见末尾缺口）。

**这条线是什么**：转写式写入（KAPPA 式）：在探针层 l_r 读知识探针分 s_i（探针只在 96 对上拟合，标准化截断 ±3），在写层 l_w 把最后 token 残差在均值差 E 方向上的投影 **clamp** 到 z_i = g·half_gap·s_i（g=2）。
问答级口径 = **陈述读出的 MC1**：TruthfulQA 每题每个候选答案改写成陈述 'The answer to the question "q" is: a'，问 P(True)，best 答案分数高于所有错误答案才算对（`e14_mc_eval.py` 的 `question_metrics`）；**不是官方按答案对数概率算的 MC1**。
主脚本 `scripts/e14_mc_eval.py`（`--write-layer` 给出才有 transfer 条件），结果 `results/results_tqa/E14/<model>/mc_eval*.{md,json,npz}`。Llama 的转写在单独一跑 `mc_eval_transfer.*`，其他模型在 `mc_eval.*` 里。

## 06 第 3 节的数字 → 出处

| 06 里的数字 | 出处文件 | 位置 |
|---|---|---|
| Llama MC1 0.649 → 0.779 | `results/results_tqa/E14/llama3.1-8b/mc_eval_transfer.md` | 行 `baseline` 0.649 / 行 `transfer_global_g2` 0.779（MC1 fixed 113 / damaged 19；argmax 0.759 → 0.895；MC2 0.707 → 0.799）；E 层 [12,13,14]、探针 L14、写层 L16、half_gap 0.760（`mc_eval_transfer.json`） |
| 五模型 +8.6…+13.7（transfer_global_g2 − baseline，MC1 百分点） | 各模型 `mc_eval.md` 行 `baseline` 与 `transfer_global_g2` | qwen3-4b 0.650 → 0.736（**+8.6**）；mistral-7b 0.702 → 0.795（+9.3）；qwen3-8b 0.700 → 0.796（+9.6）；llama3.1-8b 0.649 → 0.779（+13.0）；qwen3-1.7b 0.470 → 0.607（**+13.7**）。第六个模型 qwen3-0.6b 0.272 → 0.269（−0.3）不在'五模型'内 |
| 对照：统一放大 E_a3 | 各 `mc_eval.md` 行 `E_a3` | Llama 0.749（+10.0）、Qwen3-8B 0.732、Mistral 0.687（负）、Qwen3-4B 0.610（负）… 转写比放大稳 |
| 对照：门控转写 / 谨慎提示 | 各 `mc_eval.md` 行 `transfer_disagree_g2` / `prompt_only_cautious` | Llama 门控转写 0.738；谨慎提示 0.584（−6.5） |
| '不是自由生成改善' | `results/results_tqa/E4b/<model>/summary_truthfulqa.md` | E_a1/a2/a3/p5 的 penalised 增益 CI 全含 0 或为负（Llama −0.004…−0.026）；E4b 没有 transfer 条件（自由生成从未做过转写）|
| 陈述级转写（同机制、同模型） | `results/results_tqa/E10/llama3.1-8b/gating.md` 行 `transfer_global_g2`；`results/results_tqa/E6/llama3.1-8b/transfer.md` | 纠正 215 / 误伤 61（6.4%）、AUROC +0.130；E6 为读层×写层×g 的网格 + shuffled 对照 |
| 大模型（06 行 3.14 的一部分） | `mc_eval.md` of qwen3-14b / qwen3-32b / qwen3-30b-a3b / gemma3-4b / 12b / 27b / mistral-small-24b | 14B +10.2、Mistral-Small +7.1、Gemma-12B +4.3、Qwen3-32B/30B-A3B 0、Gemma-4B 全局转写崩到 0.277（门控转写 +0.6）、Gemma-27B +0.2 |
| 721 题 / 5179 候选 | `mc_eval*.md` 标题；`data/processed/truthfulqa_mc/manifest.json` | 817 题 − 96 估计题 = 721；5882 − 估计题候选 = 5179 |
| 逐题分数 | `results/results_tqa/E14/<model>/mc_eval*.npz` | 键：ids（候选 id 如 tqa:765:B0）、qids（题 id）、labels、roles、probe_s、disagree、每条件 P(True)/判定 |

## 当时的实际命令行（摘自服务器队列/链脚本，已核对无密钥）

```bash
# Llama：先跑放大/门控条件（run_llama_extra.sh）
python3 scripts/e14_mc_eval.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --e-layers 12 13 14 --probe-layer 14
# Llama：转写条件单独一跑（q_gpu1.sh）→ mc_eval_transfer.*
python3 scripts/e14_mc_eval.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --e-layers 12 13 14 --probe-layer 14 --write-layer 16 --conditions transfer --tag transfer
# 其他模型：big_chain.sh → run_e14_model.sh <gpu> <model> <E1> <E2> <E3> <probe> <write>，层来自该模型 results_tqa/E13 的规则选层
bash scripts/run_e14_model.sh 1 qwen3-0.6b 14 15 16 15 18
```
各模型实际用的层写在每个 `mc_eval.md` 标题里（例：qwen3-8b E layers [19,20,21], probe L21, write L23；mistral-7b [12,13,14] / L18 / L16；qwen3-4b [20,21,22] / L21 / L24；qwen3-1.7b [15,16,17] / L16 / L19；qwen3-0.6b [14,15,16] / L15 / L18）。

## 说明

- **选层规则**（`scripts/run_model_chain.sh` 注释，`results/results_tqa/E13/<model>/clean_gap.md` 为依据）：E 写层 = 干净均值差最准层 L* 及 ±1；探针层 = 干净探针最准层；**转写写层 = L*+3**（Llama：13 → 16）。g=2 是事先定的，不在 dev 上调。
- **估计 / 调参 / 测试**：估计 = 96 对（探针 + 方向 + half_gap）；测试 = 剔除估计题后的 721 题 5179 候选；**没有独立调参集**，也没有做 A/B 半（E14 脚本不分半）。
- 方向文件 `results/results_tqa/E1/<model>/directions_meandiff.npz` 是转写实际 clamp 的 E 方向与中心；只复制了'五模型 + 0.6B'六个。
- `results_tqa/E14/qwen3-32b_broken_eager2gpu/` 是已知坏跑（层数读错：E 层 37–39 / 写 41），正式结果在 `qwen3-32b/`。
- `results/results_bbq/E14/` 只放了 md/json（BBQ 的 MC 口径对照，不是 06 行 3.11 的数字）。

## 文件清单

| 包内路径 | 来源 | 大小 | 它是什么 | 支撑 06 里的哪个数字 |
|---|---|---|---|---|
| `README_experiments.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/README.md | 4.2 KB | experiments/ 的 core/周边契约与数学口径（amplify/add/clamp 定义、J/B、容器运行方式） | 口径说明 |
| `scripts/common.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/common.py | 1.7 KB | 驱动脚本共用：路径（TD_DATA_DIR/TD_RESULTS_DIR）、load_split 读切分、tf_prompt 拼陈述提示 | 依赖钩子 |
| `scripts/e2b_h1_rescue.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e2b_h1_rescue.py | 10.4 KB | 提供 run()：按批取 P(True) 与 True/False/Abstain 判定（被 e10/e14/e16 import） | 依赖钩子：'同读出' |
| `scripts/prepare_data.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/prepare_data.py | 4.2 KB | 下载/适配/平衡抽样/切分数据 → data/processed；seed 42，按题对切 calib 32 / dev 64 / 其余 test | 题 ID 切分的生成代码 |
| `core/__init__.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/__init__.py | 1.1 KB | core 包说明 | 依赖（实际 import） |
| `core/config.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/config.py | 4.1 KB | JSON 配置加载（实验/模型/提示/数据集），${ENV} 展开、深合并 | 依赖（实际 import） |
| `core/directions.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/directions.py | 33.4 KB | 方向估计：均值差、逻辑回归探针 fit_logreg（numpy IRLS，l2=1、标准化）、DAS、Haar 随机方向 | 依赖（实际 import） |
| `core/intervene.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/intervene.py | 7.9 KB | 推理时 hook 干预 SteerSpec：add / amplify / clamp，支持逐样本 α；next_token_logits / generate | 依赖（实际 import） |
| `core/io_utils.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/io_utils.py | 5.0 KB | 原子写 JSON、读 JSONL、set_seed | 依赖（实际 import） |
| `core/metrics.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/metrics.py | 12.0 KB | auroc、confusion_tf（J / B / 作答率）、三分转移等 | 依赖（实际 import） |
| `core/model_io.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/model_io.py | 8.2 KB | 加载模型/分词器、拼 chat 模板、capture_last_token 抓最后 token 残差（探针特征） | 依赖（实际 import） |
| `core/readout.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/readout.py | 5.4 KB | 陈述级读出：True/False/Unknown 首 token 对数概率 → P(True)、判定、弃答规则 | 依赖（实际 import） |
| `data_adapters/base.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/data_adapters/base.py | 5.8 KB | 统一记录 schema（id/pair_id/label/meta）与下载、平衡抽样、切分函数 | 陈述集的构造方式 |
| `data_adapters/__init__.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/data_adapters/__init__.py | 0.9 KB | 适配器注册表 | 陈述集的构造方式 |
| `data_adapters/truthfulqa.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/data_adapters/truthfulqa.py | 3.7 KB | TruthfulQA 原始 CSV → 问答记录（gold / incorrect 列表） | 陈述集的构造方式 |
| `data_adapters/truthfulqa_tf.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/data_adapters/truthfulqa_tf.py | 1.8 KB | 每题 1 真（best answer）+ 1 假（首个错误答案）陈述：'The answer to the question "q" is: a'；pair_id = 题 id | 陈述集的构造方式 |
| `configs/experiments/E2b_tqa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/experiments/E2b_tqa.json | 0.5 KB | 评测集 = truthfulqa_tf test 全部 1442 条；batch 8；seed 42 | n=1442 |
| `configs/experiments/E1_directions_tqa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/experiments/E1_directions_tqa.json | 0.6 KB | 估计集 = truthfulqa_tf calib+dev（96 对 / 192 条）；均值差 E 方向与知识探针只在这上面估 | '96 对' / '192 estimation statements' |
| `configs/prompts/default_v1.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/prompts/default_v1.json | 2.7 KB | 陈述判定提示（允许 Unknown）、候选词变体、谨慎/果断 system prompt（prompt_only_cautious 条件）、QA 提示 | 读出口径、提示 |
| `configs/models/llama3.1-8b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/llama3.1-8b.json | 0.5 KB | Llama-3.1-8B-Instruct：32 层、bf16、chat 规则；模型路径为 ${TD_MODEL_ROOT} 占位 | 模型配置 |
| `configs/datasets/truthfulqa_tf.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/datasets/truthfulqa_tf.json | 0.8 KB | TruthfulQA→陈述对：模板、按题对平衡抽样、切分 calib 32 对 / dev 64 对 / 其余 test，seed 42 | 题 ID 切分规则 |
| `data/processed/truthfulqa_tf/manifest.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa_tf/manifest.json | 0.7 KB | 切分计数 calib 64 / dev 128 / test 1442 条；unit=pair；seed 42；来源 URL | 题 ID |
| `data/processed/truthfulqa_tf/calib.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa_tf/calib.jsonl | 32.6 KB | 估计集 A：32 对 / 64 条（id 形如 tqa:765:T / tqa:765:F，pair_id = tqa:765） | 估计题 ID |
| `data/processed/truthfulqa_tf/dev.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa_tf/dev.jsonl | 67.2 KB | 估计集 B：64 对 / 128 条（calib+dev = 96 对 = 方向与探针的全部训练数据） | 估计题 ID |
| `data/processed/truthfulqa_tf/test.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa_tf/test.jsonl | 753.0 KB | 测试集 721 对 / 1442 条（与估计集零重叠，本次已核） | 测试题 ID |
| `scripts/e14_mc_eval.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e14_mc_eval.py | 11.6 KB | **主脚本**：候选陈述 P(True) → MC1 / argmax / MC2；条件 baseline / E_a* / gate_disagree* / C_abst / transfer_global_g / transfer_disagree_g / 谨慎提示 | 3.11 全部数字 |
| `scripts/run_e14_model.sh` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/run_e14_model.sh | 1.2 KB | 单模型 E14 驱动（容器内），参数 = 层选择 | 命令行 |
| `scripts/big_chain.sh` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/big_chain.sh | 2.8 KB | 大模型全链：等下载 → run_model_chain → 读 E13 选层 → run_e14_model（TQA + BBQ） | 命令行 / 选层 |
| `scripts/run_llama_extra.sh` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/run_llama_extra.sh | 2.2 KB | Llama 额外跑：E14 放大/门控条件、提示改写、估计种子（含 E14 命令） | 命令行 |
| `scripts/run_model_chain.sh` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/run_model_chain.sh | 3.8 KB | 六模型链；**选层规则写在注释里**（转写写层 = L*+3） | 选层规则 |
| `scripts/e13_clean_gap.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e13_clean_gap.py | 11.6 KB | 干净缺口逐层表（链据此选层） | 选层记录的生成代码 |
| `scripts/e10_gating.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e10_gating.py | 14.2 KB | 提供 state() 被 e14 import；其 transfer_* 条件 = 同机制的陈述级版本 | 依赖钩子 |
| `scripts/e1_directions.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e1_directions.py | 23.4 KB | 估均值差方向（转写 clamp 的方向来源） | 方向来源 |
| `scripts/e6_transfer.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e6_transfer.py | 11.0 KB | 陈述级转写（读层×写层×g 网格 + shuffled / own_g 对照）——转写机制最初在这里验证 | 机制来源 |
| `data_adapters/truthfulqa_mc.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/data_adapters/truthfulqa_mc.py | 1.9 KB | 每题每个候选答案改写成一条陈述（best / 其他正确 = True，错误 = False；meta.role） | 陈述读出口径的 MC1/MC2 |
| `configs/datasets/truthfulqa_mc.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/datasets/truthfulqa_mc.json | 0.8 KB | truthfulqa_mc 配置：全部 817 题、5882 候选，不切分；E14 评测时剔除估计题 | 721 题 / 5179 候选 |
| `data/processed/truthfulqa_mc/manifest.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa_mc/manifest.json | 0.7 KB | 5882 条候选陈述（True 2587 / False 3295） | 题 ID |
| `data/processed/truthfulqa_mc/test.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa_mc/test.jsonl | 3.2 MB | 全部候选陈述；pair_id = 题 id；评测时剔除 pair_id ∈ truthfulqa_tf calib+dev 的 96 题 → 721 题 / 5179 候选 | MC 评测题 ID（剔除后 721 题） |
| `configs/models/gemma3-12b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/gemma3-12b.json | 0.5 KB | 14 个模型配置 | 模型配置 |
| `configs/models/gemma3-27b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/gemma3-27b.json | 0.5 KB | 14 个模型配置 | 模型配置 |
| `configs/models/gemma3-4b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/gemma3-4b.json | 0.6 KB | 14 个模型配置 | 模型配置 |
| `configs/models/mistral-7b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/mistral-7b.json | 0.5 KB | 14 个模型配置 | 模型配置 |
| `configs/models/mistral-small-24b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/mistral-small-24b.json | 0.5 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-0.6b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-0.6b.json | 0.4 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-1.7b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-1.7b.json | 0.4 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-14b-dm.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-14b-dm.json | 0.5 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-14b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-14b.json | 0.5 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-30b-a3b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-30b-a3b.json | 0.7 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-32b-1gpu.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-32b-1gpu.json | 0.5 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-32b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-32b.json | 0.6 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-4b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-4b.json | 0.4 KB | 14 个模型配置 | 模型配置 |
| `configs/models/qwen3-8b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-8b.json | 0.5 KB | 14 个模型配置 | 模型配置 |
| `results/results_tqa/E14/gemma3-12b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-12b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/gemma3-12b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-12b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/gemma3-12b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-12b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/gemma3-27b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-27b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/gemma3-27b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-27b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/gemma3-27b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-27b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/gemma3-4b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-4b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/gemma3-4b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-4b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/gemma3-4b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/gemma3-4b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/llama3.1-8b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/llama3.1-8b/mc_eval.json | 3.2 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/llama3.1-8b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/llama3.1-8b/mc_eval.md | 0.9 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/llama3.1-8b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/llama3.1-8b/mc_eval.npz | 2.1 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/llama3.1-8b/mc_eval_transfer.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/llama3.1-8b/mc_eval_transfer.json | 1.4 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/llama3.1-8b/mc_eval_transfer.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/llama3.1-8b/mc_eval_transfer.md | 0.6 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/llama3.1-8b/mc_eval_transfer.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/llama3.1-8b/mc_eval_transfer.npz | 1.2 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/mistral-7b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/mistral-7b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/mistral-7b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/mistral-7b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/mistral-7b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/mistral-7b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/mistral-small-24b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/mistral-small-24b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/mistral-small-24b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/mistral-small-24b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/mistral-small-24b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/mistral-small-24b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-0.6b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-0.6b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-0.6b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-0.6b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-0.6b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-0.6b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-1.7b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-1.7b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-1.7b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-1.7b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-1.7b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-1.7b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-14b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-14b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-14b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-14b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-14b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-14b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-30b-a3b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-30b-a3b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-30b-a3b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-30b-a3b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-30b-a3b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-30b-a3b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-32b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-32b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-32b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-32b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-32b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-32b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-32b_broken_eager2gpu/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-32b_broken_eager2gpu/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-32b_broken_eager2gpu/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-32b_broken_eager2gpu/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-32b_broken_eager2gpu/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-32b_broken_eager2gpu/mc_eval.npz | 2.4 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-4b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-4b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-4b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-4b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-4b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-4b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-8b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-8b/mc_eval.json | 4.0 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-8b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-8b/mc_eval.md | 1.1 KB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_tqa/E14/qwen3-8b/mc_eval.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E14/qwen3-8b/mc_eval.npz | 2.5 MB | 每模型 mc_eval.{md,json,npz}（Llama 另有 mc_eval_transfer.*）：表 / 机器可读（含层、half_gap）/ 逐题 | 3.11 全部数字 + 逐题 |
| `results/results_bbq/E14/gemma3-12b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/gemma3-12b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/gemma3-27b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/gemma3-27b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/gemma3-4b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/gemma3-4b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/llama3.1-8b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/llama3.1-8b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/mistral-7b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/mistral-7b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/mistral-small-24b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/mistral-small-24b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/qwen3-0.6b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-0.6b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/qwen3-1.7b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-1.7b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/qwen3-14b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-14b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/qwen3-30b-a3b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-30b-a3b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/qwen3-32b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-32b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/qwen3-4b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-4b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/qwen3-8b/mc_eval.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-8b/mc_eval.md | 1.1 KB | BBQ MC 口径对照表 | 辅助 |
| `results/results_bbq/E14/gemma3-12b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/gemma3-12b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/gemma3-27b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/gemma3-27b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/gemma3-4b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/gemma3-4b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/llama3.1-8b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/llama3.1-8b/mc_eval.json | 4.1 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/mistral-7b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/mistral-7b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/mistral-small-24b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/mistral-small-24b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/qwen3-0.6b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-0.6b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/qwen3-1.7b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-1.7b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/qwen3-14b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-14b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/qwen3-30b-a3b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-30b-a3b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/qwen3-32b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-32b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/qwen3-4b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-4b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_bbq/E14/qwen3-8b/mc_eval.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E14/qwen3-8b/mc_eval.json | 4.0 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E13/gemma3-12b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/gemma3-12b/clean_gap.md | 3.6 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/gemma3-27b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/gemma3-27b/clean_gap.md | 4.5 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/gemma3-4b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/gemma3-4b/clean_gap.md | 2.7 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/llama3.1-8b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/llama3.1-8b/clean_gap.md | 1.9 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/mistral-7b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/mistral-7b/clean_gap.md | 2.5 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/mistral-small-24b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/mistral-small-24b/clean_gap.md | 3.1 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-0.6b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-0.6b/clean_gap.md | 2.3 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-1.7b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-1.7b/clean_gap.md | 2.3 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-14b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-14b/clean_gap.md | 3.0 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-30b-a3b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-30b-a3b/clean_gap.md | 3.6 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-32b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-32b/clean_gap.md | 4.6 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-32b_broken_devicemap/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-32b_broken_devicemap/clean_gap.md | 4.6 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-32b_broken_eager2gpu/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-32b_broken_eager2gpu/clean_gap.md | 4.6 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-4b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-4b/clean_gap.md | 2.8 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/qwen3-8b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-8b/clean_gap.md | 2.8 KB | **选层记录**：每模型逐层干净探针 / 均值差 AUROC vs 输出；best 层 → E 层 ±1、探针层、写层 +3 | 选层记录 |
| `results/results_tqa/E13/gemma3-12b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/gemma3-12b/clean_gap.json | 18.9 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/gemma3-27b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/gemma3-27b/clean_gap.json | 24.5 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/gemma3-4b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/gemma3-4b/clean_gap.json | 13.4 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/llama3.1-8b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/llama3.1-8b/clean_gap.json | 9.0 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/mistral-7b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/mistral-7b/clean_gap.json | 12.6 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/mistral-small-24b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/mistral-small-24b/clean_gap.json | 15.8 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-0.6b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-0.6b/clean_gap.json | 11.0 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-1.7b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-1.7b/clean_gap.json | 11.0 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-14b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-14b/clean_gap.json | 15.8 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-30b-a3b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-30b-a3b/clean_gap.json | 18.9 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-32b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-32b/clean_gap.json | 25.3 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-32b_broken_devicemap/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-32b_broken_devicemap/clean_gap.json | 26.6 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-32b_broken_eager2gpu/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-32b_broken_eager2gpu/clean_gap.json | 25.4 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-4b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-4b/clean_gap.json | 14.2 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E13/qwen3-8b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/qwen3-8b/clean_gap.json | 14.2 KB | 同上机器可读（best_meandiff_layer / best_probe_layer） | 选层记录 |
| `results/results_tqa/E1/llama3.1-8b/summary.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/llama3.1-8b/summary.json | 20.1 KB | 方向估计摘要（逐层 gap / sep；LA-Score 层） | 方向来源 |
| `results/results_tqa/E1/llama3.1-8b/directions_meandiff.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/llama3.1-8b/directions_meandiff.npz | 3.2 MB | **转写实际 clamp 的均值差方向**（E_L*、E_center_L*） | 写入向量 |
| `results/results_tqa/E1/qwen3-8b/summary.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/qwen3-8b/summary.json | 19.7 KB | 方向估计摘要（逐层 gap / sep；LA-Score 层） | 方向来源 |
| `results/results_tqa/E1/qwen3-8b/directions_meandiff.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/qwen3-8b/directions_meandiff.npz | 3.6 MB | **转写实际 clamp 的均值差方向**（E_L*、E_center_L*） | 写入向量 |
| `results/results_tqa/E1/mistral-7b/summary.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/mistral-7b/summary.json | 17.7 KB | 方向估计摘要（逐层 gap / sep；LA-Score 层） | 方向来源 |
| `results/results_tqa/E1/mistral-7b/directions_meandiff.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/mistral-7b/directions_meandiff.npz | 3.2 MB | **转写实际 clamp 的均值差方向**（E_L*、E_center_L*） | 写入向量 |
| `results/results_tqa/E1/qwen3-4b/summary.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/qwen3-4b/summary.json | 19.7 KB | 方向估计摘要（逐层 gap / sep；LA-Score 层） | 方向来源 |
| `results/results_tqa/E1/qwen3-4b/directions_meandiff.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/qwen3-4b/directions_meandiff.npz | 2.2 MB | **转写实际 clamp 的均值差方向**（E_L*、E_center_L*） | 写入向量 |
| `results/results_tqa/E1/qwen3-1.7b/summary.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/qwen3-1.7b/summary.json | 15.4 KB | 方向估计摘要（逐层 gap / sep；LA-Score 层） | 方向来源 |
| `results/results_tqa/E1/qwen3-1.7b/directions_meandiff.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/qwen3-1.7b/directions_meandiff.npz | 1.4 MB | **转写实际 clamp 的均值差方向**（E_L*、E_center_L*） | 写入向量 |
| `results/results_tqa/E1/qwen3-0.6b/summary.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/qwen3-0.6b/summary.json | 15.6 KB | 方向估计摘要（逐层 gap / sep；LA-Score 层） | 方向来源 |
| `results/results_tqa/E1/qwen3-0.6b/directions_meandiff.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/qwen3-0.6b/directions_meandiff.npz | 716.0 KB | **转写实际 clamp 的均值差方向**（E_L*、E_center_L*） | 写入向量 |
| `results/results_tqa/E6/llama3.1-8b/transfer.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E6/llama3.1-8b/transfer.json | 36.0 KB | Llama 陈述级转写网格（读层 × 写层 × g，shuffled 对照） | 机制来源 |
| `results/results_tqa/E6/llama3.1-8b/transfer.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E6/llama3.1-8b/transfer.md | 6.4 KB | Llama 陈述级转写网格（读层 × 写层 × g，shuffled 对照） | 机制来源 |
| `results/results_tqa/E6/llama3.1-8b/transfer_L13.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E6/llama3.1-8b/transfer_L13.json | 22.9 KB | Llama 陈述级转写网格（读层 × 写层 × g，shuffled 对照） | 机制来源 |
| `results/results_tqa/E6/llama3.1-8b/transfer_L13.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E6/llama3.1-8b/transfer_L13.md | 4.1 KB | Llama 陈述级转写网格（读层 × 写层 × g，shuffled 对照） | 机制来源 |
| `results/results_tqa/E10/llama3.1-8b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/gating.md | 2.6 KB | 陈述级：行 transfer_global_g2 / transfer_disagree_g2 / transfer_probeconf_t1_g2（215/61、201/42、88/3） | 陈述级转写 |
| `results/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.md | 3.4 KB | 自由生成汇总：E_a* 增益 CI 含 0（没有 transfer 条件） | '不是自由生成改善' |
| `results/results_tqa/E4b/qwen3-8b/summary_truthfulqa.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/qwen3-8b/summary_truthfulqa.md | 3.4 KB | 自由生成汇总：E_a* 增益 CI 含 0（没有 transfer 条件） | '不是自由生成改善' |
| `results/results_tqa/E4b/mistral-7b/summary_truthfulqa.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/mistral-7b/summary_truthfulqa.md | 3.3 KB | 自由生成汇总：E_a* 增益 CI 含 0（没有 transfer 条件） | '不是自由生成改善' |
| `server_logs/e14_llama_transfer.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_llama_transfer.log | 1.5 KB | **Llama 转写运行 stdout**：721 题（96 排除）、探针 L14 AUROC 0.901、三条件 MC1 | 0.649 → 0.779 运行记录 |
| `server_logs/extra_tqa_e14_llama3.1-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/extra_tqa_e14_llama3.1-8b.log | 2.1 KB | Llama 放大 / 门控条件运行 stdout | 运行记录 |
| `server_logs/e14_gemma3-12b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_gemma3-12b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_gemma3-27b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_gemma3-27b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_gemma3-4b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_gemma3-4b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_mistral-7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_mistral-7b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_mistral-small-24b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_mistral-small-24b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_qwen3-0.6b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_qwen3-0.6b.log | 2.4 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_qwen3-1.7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_qwen3-1.7b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_qwen3-14b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_qwen3-14b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_qwen3-30b-a3b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_qwen3-30b-a3b.log | 3.0 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_qwen3-32b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_qwen3-32b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_qwen3-4b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_qwen3-4b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14_qwen3-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14_qwen3-8b.log | 2.6 KB | 其他模型 E14 运行 stdout（头行含 E 层 / 探针层 / 写层） | 运行记录 / 选层 |
| `server_logs/e14bbq_gemma3-12b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_gemma3-12b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_gemma3-27b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_gemma3-27b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_gemma3-4b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_gemma3-4b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_llama3.1-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_llama3.1-8b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_mistral-7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_mistral-7b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_mistral-small-24b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_mistral-small-24b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_qwen3-0.6b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_qwen3-0.6b.log | 2.4 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_qwen3-1.7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_qwen3-1.7b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_qwen3-14b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_qwen3-14b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_qwen3-30b-a3b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_qwen3-30b-a3b.log | 3.0 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_qwen3-32b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_qwen3-32b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_qwen3-4b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_qwen3-4b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/e14bbq_qwen3-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e14bbq_qwen3-8b.log | 2.6 KB | BBQ E14 运行 stdout | 辅助 |
| `server_logs/q_gpu1.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_gpu1.sh | 0.8 KB | **队列脚本**：Llama 转写命令 + 六模型 BBQ E14 的层参数 | 命令行 |
| `server_logs/big_chain.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/big_chain.sh | 1.1 KB | 服务器上的 big_chain.sh 副本 | 命令行 |
| `server_logs/chain_llama_extra.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/chain_llama_extra.log | 0.5 KB | run_llama_extra.sh 的外层日志（时间戳 / rc） | 运行记录 |
| `server_logs/common.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/common.sh | 0.5 KB | dex() 容器环境变量 | 运行环境 |

## 缺口（GPT 要而没有 / 未复制）

- 官方 MC1/MC2（按答案 token 对数概率）口径的结果：没有做过，本包全部是陈述读出口径。
- 独立调参集题 ID / A-B 半：E14 不分半（见说明）。
- 其余 8 个模型的方向文件 `results_tqa/E1/<model>/directions_meandiff.npz`（qwen3-14b 5.0 MB、qwen3-32b 7.9 MB、qwen3-30b-a3b 2.4 MB、gemma3-4b 2.1 MB、gemma3-12b 4.5 MB、gemma3-27b 8.1 MB、mistral-small-24b 5.0 MB、qwen3-32b_broken_* 7.9 MB×2）未复制。
- 其他模型的陈述级转写 E6（`results_tqa/E6/<model>/`）未复制，只带了 Llama。
- BBQ E14 的逐题 npz（`results_bbq/E14/<model>/mc_eval.npz`，各 ≈ 1.5 MB）未复制。
- 当时的报告段落（`docs/给〈老师〉_不积极点清单与攻克_20261027.md` 行 154、219 的 '0.779 (+13.0)' 表）文件名含老师名字，未复制；数字可在本包 `mc_eval*.md` 全部找到。

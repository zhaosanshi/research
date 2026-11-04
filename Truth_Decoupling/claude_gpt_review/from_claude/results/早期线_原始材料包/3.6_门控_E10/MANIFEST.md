# MANIFEST — 3.6_门控_E10

生成时间：2026-11-04 07:58 PST。来源标记：`local:` = 本机项目 `/Users/airsanshi/MyPapers/Truth_Decoupling`；`pro5000:` = 服务器 `/data/sanshizhao/Truth_Decoupling`（布局与本机相同，`code/` 对应本机根目录）。
文件 140 个，合计 17.9 MB。单文件 > 20 MB 不复制（见末尾缺口）。

**这条线是什么**：在 TruthfulQA 真/假陈述上（Llama-3.1-8B，test 1442 条），把证据方向 E 在第 12+13+14 层按 `amplify`（放大模型自己沿 E 的信号，α 逐题）写入；
门信号全部不碰标签：u = 输出不确定度 1−|2P(True)−1|；s = 第 14 层知识探针分（只在 96 对 calib+dev 上拟合，标准化、截断 ±3）；d = 探针与模型输出倾向相反。
主脚本 `scripts/e10_gating.py`，结果 `results/results_tqa/E10/llama3.1-8b/gating.{md,json,npz}`。

## 06 第 3 节的数字 → 出处

| 06 里的数字 | 出处文件 | 位置 |
|---|---|---|
| 纠正 211 / 误伤 39 / 误伤率 4.1% / 全题正确率 0.787 | `results/results_tqa/E10/llama3.1-8b/gating.md` | 行 `disagree_a5`（fixed 211、damaged 39、known damage 4.1、acc all 0.787、J 0.585）；机器可读 `gating.json` → `conditions.disagree_a5.{corrected,damaged,known_damage_rate,acc_all}` |
| 分母 959（原本判对的陈述）/ 基线 0.665 | 同上 | 标题 `baseline correct 959 / wrong 467 / abstain 16`；行 `baseline` acc all 0.665 |
| '有把握才推'门：误伤 0.3%、只纠正 83 | 同上 | 行 `probeconf_t1_a3`（83 / 3 / 0.3% / acc 0.721）；更严的 `probeconf_t2_a3` 2 / 0 |
| 不确定度门无效 | 同上 | 行 `soft_unc_a3/a5/a8`、`hard_unc_t0.5_a3/t0.8_a3`：纠正:误伤 ≈ 2:1、acc ≤ 0.712，都不如无门 `global_a3`（0.758） |
| 无门参考 215 / 90（9.4%）、0.758 | 同上 | 行 `global_a3` |
| 选门的乐观度（事后检查） | `results/results_tqa/E10/llama3.1-8b/gating_splithalf.md` | A 半按 penalised 选出 `disagree_a5`，B 半 acc 0.774 vs 基线 0.679 vs 无门 0.751；按题聚类 bootstrap：disagree_a5 − global_a3 Δacc +0.029 [+0.013, +0.044]，Δ误伤率 −0.053 [−0.069, −0.037] |
| GPT 第 13 轮：不满足'不伤已会题 ≤ 1 分' | 同 `gating.md` | known damage 4.1% > 1%；满足 1% 线的只有 `probeconf_t1_a3`（0.3%，纠正 83 = 18% 的错题） |
| 逐题分数 | `results/results_tqa/E10/llama3.1-8b/gating.npz` | 键：ids(1442)、labels、probe_s、unc、disagree、`<条件>__p`（P(True)）、`<条件>__pred` |
| 问答级（自由生成）门控延伸 | `results/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.md` | `gate_*` 条件 penalised 增益 CI 全含 0（+0.004…+0.025）；逐题生成与判分在 `truthfulqa_<条件>.json` |

## 当时的实际命令行（摘自服务器队列/链脚本，已核对无密钥）

```bash
# 六模型链（服务器 run_model_chain.sh）：B0 → E1 → E13 → 按规则选层 → E11 → E10 → E6；Llama 选出 E 层 12 13 14 / 探针 14 / 转写写层 16
python3 scripts/e10_gating.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --e-layers 12 13 14 --probe-layer 14 --write-layer 16
python3 scripts/e10_splithalf.py --npz results_tqa/E10/llama3.1-8b/gating.npz            # 本机，无 GPU
python3 scripts/e12_subset_prediction.py --npz results_tqa/E10/llama3.1-8b/gating.npz --test data/processed/truthfulqa_tf/test.jsonl
# 问答级门控（E4b，q_gpu3.sh / q_judge.sh）
python3 scripts/e4b_qa_gated.py --phase generate --regime tqa --model llama3.1-8b --force --batch-size 16
python3 scripts/e4b_qa_gated.py --phase judge --regime tqa --model llama3.1-8b --workers 12   # DEEPSEEK_API_KEY_FILE 指向密钥文件
```
容器环境变量：`TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_tqa`、`TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed`（见 `server_logs/common.sh`）。

## 说明

- **选层记录**：`results/results_tqa/E13/llama3.1-8b/clean_gap.md`——干净探针最准层 L14（AUROC 0.917）、干净均值差最准层 L13（0.843）；链规则（`scripts/run_model_chain.sh` 注释）：E 写层 = L*−1, L*, L*+1 = 12,13,14；探针层 = 最佳探针层 14；转写写层 = L*+3 = 16。α=3 的依据：`results/results_tqa/E7/llama3.1-8b/multilayer.md`（12+13+14 α=3 AUROC +0.084，α=5 塌回 +0.018）。
- **方向**：`results/results_tqa/E1/llama3.1-8b/directions_meandiff.npz` 是 E10 实际写入的均值差方向（键 E_L12/13/14/16、E_center_L*、C_L20）；`summary.json` 里 LA-Score 选的层（E_read 14 / E_write 18 / C_write 20）只用于 C 条件（`C_global_a1` 在 L20）。
- **估计 / 调参 / 测试**：估计 = calib+dev 96 对；测试 = test 1442 条。**没有独立调参集**——门的阈值（T、A）和 α 是在 test 全集上并列报告的，选择乐观度由 `gating_splithalf` 事后补（按 pair_id sha1 奇偶分 A/B 半，可由 `gating.npz` 的 ids 复算）。
- **问答级 n=717**：E4b 用 `truthfulqa` test 717 题；本次核对 96 道估计题全部落在 `truthfulqa` dev（100 题）里，test 与估计集零重叠。
- `results_tqa/E10/qwen3-32b_broken_*` 两个目录是已知坏跑（device_map / eager 两卡问题），保留原名以示区分，正式结果在 `qwen3-32b/`。
- `core/gate.py` 不是 `e10_gating.py` 的 import；它是老师方案 §3.5（τ_lo/τ_hi 三区间、连续强度）在 core 里的实现，E10 实际用的是脚本内的二值门。一并附上供对照。

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
| `scripts/e10_gating.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e10_gating.py | 14.2 KB | **主脚本**：18 个门控条件（无门 / 不确定度软门硬门 / 不一致门 / 只推一致（对照）/ 有把握门 / 转写 clamp / 组合 / C / 谨慎提示），输出 gating.{json,md,npz} | 全部 3.6 数字 |
| `scripts/e10_splithalf.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e10_splithalf.py | 8.9 KB | 事后：按题 hash 分 A/B 半，A 半选门 B 半报；按题聚类 bootstrap CI（读 gating.npz，无 GPU） | 选择乐观度检查 |
| `scripts/e12_subset_prediction.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e12_subset_prediction.py | 5.4 KB | 事后：按 TruthfulQA 类别看'缺口是否预测增益'（读 gating.npz） | subset_prediction.* |
| `scripts/e4b_qa_gated.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e4b_qa_gated.py | 20.7 KB | 自由生成问答版门控（两阶段：GPU 生成 / DeepSeek 判分），含 ITI 臂 | 问答级门控（E4b） |
| `scripts/e11_dose_rates.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e11_dose_rates.py | 16.1 KB | 剂量曲线脚本；E4b 从它的 dose_rates.json 读层；含 gate_disagree 复现条件 | 依赖 / 门控复现 |
| `scripts/e13_clean_gap.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e13_clean_gap.py | 11.6 KB | 干净缺口逐层表（探针 / 均值差 vs 输出 AUROC），链据此选层 | 选层记录的生成代码 |
| `scripts/e1_directions.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e1_directions.py | 23.4 KB | 估 E / C 方向（均值差）+ LA-Score 选层，产出 results_tqa/E1 | 方向来源 |
| `scripts/run_model_chain.sh` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/run_model_chain.sh | 3.8 KB | 六模型链；**选层规则写在注释里**（E 层 = L*±1，探针层 = 最佳探针层，转写层 = L*+3） | 选层规则 |
| `core/gate.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/gate.py | 5.1 KB | 老师方案 §3.5 门控（τ_lo/τ_hi、连续强度）的 core 实现——E10 脚本未 import，供对照 | 参照 |
| `core/iti.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/iti.py | 6.3 KB | ITI 钩子（e4b 的 iti 臂 import） | e4b 依赖 |
| `configs/datasets/truthfulqa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/datasets/truthfulqa.json | 0.5 KB | TruthfulQA 问答集配置：dev 100 / test 717（按题切，seed 42） | 自由生成题 ID |
| `data/processed/truthfulqa/manifest.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa/manifest.json | 0.8 KB | 问答集切分计数与原始文件 sha1 | 自由生成题 ID |
| `data/processed/truthfulqa/dev.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa/dev.jsonl | 81.0 KB | 问答 dev 100 题（本次核对：96 道估计题全部落在这 100 题里） | — |
| `data/processed/truthfulqa/test.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa/test.jsonl | 583.9 KB | 问答 test 717 题（与 96 道估计题零重叠，本次已核；E4b 的 n=717） | 自由生成题 ID |
| `configs/experiments/E4_qa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/experiments/E4_qa.json | 1.1 KB | E4b 判分器配置（DeepSeek deepseek-chat，thinking 关；只写环境变量名，不含密钥） | 判分口径 |
| `core/judge.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/judge.py | 7.3 KB | OpenAI 兼容 LLM judge + 磁盘缓存；密钥只从环境变量/文件读 | e4b 判分依赖 |
| `configs/models/gemma3-12b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/gemma3-12b.json | 0.5 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/gemma3-27b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/gemma3-27b.json | 0.5 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/gemma3-4b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/gemma3-4b.json | 0.6 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/mistral-7b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/mistral-7b.json | 0.5 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/mistral-small-24b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/mistral-small-24b.json | 0.5 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-0.6b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-0.6b.json | 0.4 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-1.7b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-1.7b.json | 0.4 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-14b-dm.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-14b-dm.json | 0.5 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-14b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-14b.json | 0.5 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-30b-a3b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-30b-a3b.json | 0.7 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-32b-1gpu.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-32b-1gpu.json | 0.5 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-32b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-32b.json | 0.6 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-4b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-4b.json | 0.4 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `configs/models/qwen3-8b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-8b.json | 0.5 KB | 其他模型配置（E10 在 14 个模型上跑过） | 模型配置 |
| `results/results_tqa/E10/llama3.1-8b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/gating.md | 2.6 KB | **关键表**：18 条件 × 纠正/误伤/误伤率/转弃答/正确率/J/B/作答率/penalised | 3.6 全部数字 |
| `results/results_tqa/E10/llama3.1-8b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/gating.json | 21.5 KB | 同表机器可读 + 每条件 ΔAUROC 95% CI、九格转移矩阵、基线计数 | 3.6 全部数字 |
| `results/results_tqa/E10/llama3.1-8b/gating.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/gating.npz | 1.0 MB | **逐题**：ids、labels、probe_s、unc、disagree、每条件 P(True) 与判定 | 3.6 全部数字 |
| `results/results_tqa/E10/llama3.1-8b/gating_splithalf.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/gating_splithalf.md | 4.1 KB | 分半选门 + 按题聚类 bootstrap CI + 全集一致性核对 | 选择乐观度 |
| `results/results_tqa/E10/llama3.1-8b/gating_splithalf.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/gating_splithalf.json | 25.9 KB | 同上机器可读（含每条件在 ALL/halfA/halfB 的指标） | 选择乐观度 |
| `results/results_tqa/E10/llama3.1-8b/subset_prediction.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/subset_prediction.json | 5.4 KB | 类别级：缺口 vs 实际增益（global_a3 / disagree_a5 / transfer_global_g2 三版） | 辅助（'差值预测收益'） |
| `results/results_tqa/E10/llama3.1-8b/subset_prediction.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/subset_prediction.md | 1.9 KB | 类别级：缺口 vs 实际增益（global_a3 / disagree_a5 / transfer_global_g2 三版） | 辅助（'差值预测收益'） |
| `results/results_tqa/E10/llama3.1-8b/subset_prediction_disagree_a5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/subset_prediction_disagree_a5.json | 5.4 KB | 类别级：缺口 vs 实际增益（global_a3 / disagree_a5 / transfer_global_g2 三版） | 辅助（'差值预测收益'） |
| `results/results_tqa/E10/llama3.1-8b/subset_prediction_disagree_a5.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/subset_prediction_disagree_a5.md | 2.0 KB | 类别级：缺口 vs 实际增益（global_a3 / disagree_a5 / transfer_global_g2 三版） | 辅助（'差值预测收益'） |
| `results/results_tqa/E10/llama3.1-8b/subset_prediction_transfer_global_g2.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/subset_prediction_transfer_global_g2.json | 5.4 KB | 类别级：缺口 vs 实际增益（global_a3 / disagree_a5 / transfer_global_g2 三版） | 辅助（'差值预测收益'） |
| `results/results_tqa/E10/llama3.1-8b/subset_prediction_transfer_global_g2.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/llama3.1-8b/subset_prediction_transfer_global_g2.md | 2.0 KB | 类别级：缺口 vs 实际增益（global_a3 / disagree_a5 / transfer_global_g2 三版） | 辅助（'差值预测收益'） |
| `results/results_tqa/E10/gemma3-12b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/gemma3-12b/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/gemma3-27b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/gemma3-27b/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/gemma3-4b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/gemma3-4b/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/mistral-7b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/mistral-7b/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/mistral-small-24b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/mistral-small-24b/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-0.6b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-0.6b/gating.md | 2.5 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-1.7b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-1.7b/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-14b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-14b/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-30b-a3b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-30b-a3b/gating.md | 2.5 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-32b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-32b/gating.md | 2.5 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-32b_broken_devicemap/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-32b_broken_devicemap/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-32b_broken_eager2gpu/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-32b_broken_eager2gpu/gating.md | 2.7 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-4b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-4b/gating.md | 2.5 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/qwen3-8b/gating.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-8b/gating.md | 2.6 KB | 同脚本在其他模型上的结果表（模型名见路径；06 行 3.7 的来源之一） | 辅助 |
| `results/results_tqa/E10/gemma3-12b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/gemma3-12b/gating.json | 21.4 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/gemma3-27b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/gemma3-27b/gating.json | 21.3 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/gemma3-4b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/gemma3-4b/gating.json | 21.6 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/mistral-7b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/mistral-7b/gating.json | 21.8 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/mistral-small-24b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/mistral-small-24b/gating.json | 21.6 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-0.6b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-0.6b/gating.json | 20.9 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-1.7b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-1.7b/gating.json | 21.8 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-14b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-14b/gating.json | 21.6 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-30b-a3b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-30b-a3b/gating.json | 21.6 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-32b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-32b/gating.json | 21.6 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-32b_broken_devicemap/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-32b_broken_devicemap/gating.json | 19.9 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-32b_broken_eager2gpu/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-32b_broken_eager2gpu/gating.json | 22.0 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-4b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-4b/gating.json | 21.7 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E10/qwen3-8b/gating.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E10/qwen3-8b/gating.json | 21.8 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E1/llama3.1-8b/summary.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/llama3.1-8b/summary.json | 20.1 KB | E/C 方向估计摘要：n_E/n_C、逐层 gap/sep/cos、LA-Score 读写层（E_read 14 / E_write 18 / C_write 20） | 方向与 C 层来源 |
| `results/results_tqa/E1/llama3.1-8b/layer_scores.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/llama3.1-8b/layer_scores.json | 25.2 KB | 逐层 LA-Score 四量（dact/dattn/sep/mass） | 辅助 |
| `results/results_tqa/E1/llama3.1-8b/directions_meandiff.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E1/llama3.1-8b/directions_meandiff.npz | 3.2 MB | **实际写入的方向**：E_L*、E_center_L*、C_L*（均值差） | 写入向量 |
| `results/results_tqa/E13/llama3.1-8b/clean_gap.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/llama3.1-8b/clean_gap.md | 1.9 KB | **选层记录**：逐层干净探针/均值差 AUROC vs 输出 0.757；L14 探针 0.917、L13 均值差 0.843；预注册预测 | 层 12/13/14、探针 14、写 16 的依据 |
| `results/results_tqa/E13/llama3.1-8b/clean_gap.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E13/llama3.1-8b/clean_gap.json | 9.0 KB | 同上机器可读（best_meandiff_layer / best_probe_layer / per_layer） | 选层记录 |
| `results/results_tqa/E7/llama3.1-8b/multilayer.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E7/llama3.1-8b/multilayer.md | 2.5 KB | 12+13+14 联合放大的首次发现：α=3 AUROC +0.084、α=5 塌回；单层 L13/L14 对照 | α=3 的依据 |
| `results/results_tqa/E7/llama3.1-8b/multilayer.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E7/llama3.1-8b/multilayer.json | 16.0 KB | 同上机器可读 | 辅助 |
| `results/results_tqa/E7/llama3.1-8b/multilayer_random.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E7/llama3.1-8b/multilayer_random.md | 1.0 KB | 同层随机方向对照（AUROC 变化 ≤ +0.001） | 辅助 |
| `results/results_tqa/E11/llama3.1-8b/dose_rates.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E11/llama3.1-8b/dose_rates.md | 5.9 KB | 剂量曲线（E_a−1…a5、C、随机、谨慎提示、gate_disagree_a5 复现）+ A 半选强度 B 半报 | 门控复现 / 强度选择 |
| `results/results_tqa/E11/llama3.1-8b/dose_rates.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E11/llama3.1-8b/dose_rates.json | 45.7 KB | 同上机器可读（e_layers / probe_layer / c_layer / selection） | 辅助 |
| `results/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.json | 11.6 KB | 问答级汇总：每条件 correct/wrong/abstain/penalised + 增益 bootstrap CI；探针判自己答案的诊断 | 问答级门控（CI 含 0） |
| `results/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.md | 3.4 KB | 问答级汇总：每条件 correct/wrong/abstain/penalised + 增益 bootstrap CI；探针判自己答案的诊断 | 问答级门控（CI 含 0） |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_C_abst_a1.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_C_abst_a1.json | 769.2 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_E_a1.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_E_a1.json | 768.3 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_E_a2+C_abst_a1.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_E_a2+C_abst_a1.json | 768.9 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_E_a2.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_E_a2.json | 767.6 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_E_a3.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_E_a3.json | 768.4 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_E_p5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_E_p5.json | 772.7 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_baseline.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_baseline.json | 768.4 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_cautious_prompt.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_cautious_prompt.json | 762.6 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_disagree_a3.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_disagree_a3.json | 901.3 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_disagree_a5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_disagree_a5.json | 28.7 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_disagree_abst_a3.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_disagree_abst_a3.json | 901.6 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_disagree_p5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_disagree_p5.json | 902.6 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_probeneg_a3.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_probeneg_a3.json | 901.2 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_probeneg_a5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_gate_probeneg_a5.json | 28.7 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_iti_k48_a15.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_iti_k48_a15.json | 772.3 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_iti_k48_a5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_iti_k48_a5.json | 768.3 KB | **逐题生成 + 判分**（evaluations[]：question / gold / output / judge / outcome；gate_* 还有 fired / probe_s / p_true_self / own_verdict / base_output） | 问答级门控逐题 |
| `server_logs/tqa_e10_llama3.1-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_llama3.1-8b.log | 4.3 KB | **E10 Llama 的运行 stdout**：头行（n=1442、探针 L14 AUROC 0.917、不一致 434 条 30.1%、半间距 L16 0.76）+ 每条件一行 | 运行记录 |
| `server_logs/tqa_e10_gemma3-12b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_gemma3-12b.log | 4.4 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_gemma3-27b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_gemma3-27b.log | 4.4 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_gemma3-4b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_gemma3-4b.log | 4.3 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_mistral-7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_mistral-7b.log | 4.3 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_mistral-small-24b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_mistral-small-24b.log | 4.4 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_qwen3-0.6b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_qwen3-0.6b.log | 4.2 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_qwen3-1.7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_qwen3-1.7b.log | 4.3 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_qwen3-14b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_qwen3-14b.log | 4.3 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_qwen3-30b-a3b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_qwen3-30b-a3b.log | 4.8 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_qwen3-32b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_qwen3-32b.log | 4.3 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_qwen3-4b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_qwen3-4b.log | 4.3 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/tqa_e10_qwen3-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/tqa_e10_qwen3-8b.log | 4.3 KB | 其他模型的 E10 运行 stdout | 辅助 |
| `server_logs/e4b_tqa_llama3.1-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e4b_tqa_llama3.1-8b.log | 3.3 KB | E4b 生成阶段 stdout（门触发数等） | 运行记录 |
| `server_logs/q_gpu3.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_gpu3.sh | 1.5 KB | 服务器队列脚本：E4b 生成命令（Llama TQA/BBQ、Qwen BBQ） | 命令行 |
| `server_logs/q_judge.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_judge.sh | 1.7 KB | 服务器队列脚本：E4b 判分命令（密钥走 *_FILE 环境变量，脚本内无密钥） | 命令行 |
| `server_logs/q_judge2.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_judge2.sh | 1.9 KB | 判分第二轮（重判）命令 | 命令行 |
| `server_logs/common.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/common.sh | 0.5 KB | 队列脚本公用 dex()：容器环境变量（TD_RESULTS_DIR / TD_DATA_DIR） | 运行环境 |

## 缺口（GPT 要而没有 / 未复制）

- 独立'调参集'的题 ID：不存在（见说明）。
- 其他 13 个模型的逐题文件 `results_tqa/E10/<model>/gating.npz`（各 ≈ 1 MB）未复制，只放了 md/json；BBQ / HaluEval / 知识集上的 E10（`results_bbq/E10`、`results_halu/E10`、`results_v2/E10`）未复制——都不是 06 行 3.6 的数字。
- E4b 判分缓存 `results/judge_cache/` 未复制（判分结果已写在各 `truthfulqa_<条件>.json` 的 `evaluations[].judge`）。
- 当时给老师的报告（`docs/给〈老师〉_证据旋钮与EmptyShelves_20261026.md` §2.5、`docs/给〈老师〉_回应下一步实验_20261026.md` §9、`docs/实验_llama_2x3_20261026.md` §9 行 237–257）文件名或内文含老师名字，未复制；其中 3.6 的全部数字都能在本包 `gating.md` / `gating_splithalf.md` 找到，不需另行摘录。

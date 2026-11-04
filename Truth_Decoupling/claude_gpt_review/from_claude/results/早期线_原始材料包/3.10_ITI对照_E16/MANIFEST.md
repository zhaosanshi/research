# MANIFEST — 3.10_ITI对照_E16

生成时间：2026-11-04 07:58 PST。来源标记：`local:` = 本机项目 `/Users/airsanshi/MyPapers/Truth_Decoupling`；`pro5000:` = 服务器 `/data/sanshizhao/Truth_Decoupling`（布局与本机相同，`code/` 对应本机根目录）。
文件 120 个，合计 16.2 MB。单文件 > 20 MB 不复制（见末尾缺口）。

**这条线是什么**：ITI（Inference-Time Intervention，Li et al. NeurIPS 2023）在**我们协议**下的自实现对照：同一 96 对估计陈述（calib+dev）选 K=48 个注意力头（按 4 折按题分组 CV 探针准确率），
干预 = 每次前向最后位置给选中头加 α·σ_h·单位均值差方向；同一 1442 条测试陈述、同一 True/False/Unknown 读出、同一指标；另跑陈述读出口径的 MC1/MC2（721 题）和自由生成（E4b 的 iti 臂）。
**不是原论文协议**（原文：Q/A 少样本提示、官方 MC 口径、GPT-judge 自由生成）——06 行 3.10 已注明，本包不改变这个限定。
主脚本 `scripts/e16_iti_baseline.py` + `core/iti.py`，结果 `results/results_tqa/E16/<model>/iti.{md,json,npz}`、`iti_heads.npz`。

## 06 第 3 节的数字 → 出处

| 06 里的说法 | 出处文件 | 位置 |
|---|---|---|
| 96 对、同读出、六模型 | `configs/experiments/E1_directions_tqa.json`；`results/results_tqa/E16/{llama3.1-8b,qwen3-8b,mistral-7b,qwen3-4b,qwen3-1.7b,qwen3-0.6b}/iti.md` | 各 iti.md 标题：`192 estimation statements; K=48 heads`；表列 = 与 E10/E11 同口径 |
| 陈述级：负或零 | 各 `iti.md` 'All items' 表与 'Half B only' 表 | Llama：penalised +0.343（α5）≈ 基线 +0.341，α10/15/20 → +0.259/+0.218/+0.010，已知题误伤 13.8%→56%；Mistral α5 +0.077 vs 基线 +0.378；Qwen3-8B α5 +0.325 vs +0.348；Qwen3-4B +0.340 vs +0.377；Qwen3-1.7B −0.003 vs +0.196；Qwen3-0.6B −0.013 vs −0.016 |
| MC：负或零 | 各 `iti.md` 'truthfulqa_mc question level' 表 | Llama MC1 0.649 → 0.626/0.603/0.595/0.628；Mistral 0.702 → 0.630…；Qwen3-8B 0.700 → 0.703/0.691/0.692/0.699（零）；0.6B 0.272 → 0.308…0.401（唯一上涨，但该模型基线 J=0 几乎全答 True） |
| 自由生成：负 | `results/results_tqa/E4b/<model>/summary_truthfulqa.md` 行 `iti_k48_a5` / `iti_k48_a15` | Llama penalised 增益 −0.067 [−0.105, −0.031] / −0.141 [−0.195, −0.081]；Qwen3-8B −0.226 / −0.286 vs 基线 −0.188；Mistral +0.029 / −0.086 vs 基线 +0.060 |
| '加性 = 偏置旋钮' | 各 `iti.md` 的 B 列（P(说 True｜作答)） | Llama 0.391 → 0.616 → 0.744 → 0.830 → 0.999 随 α 单调升，J 反而降 |
| 选头记录 | `results/results_tqa/E16/<model>/iti.json` → `heads`（[层, 头, 验证准确率] × 48）；`iti_heads.npz`（heads / val_acc / directions / sigmas） | iti.md 标题列出被选头所在层与验证准确率范围（Llama 0.812–0.870） |
| 强度选择 | `iti.json` → `selection` | A 半按 penalised 选 α（Llama 选 a5），B 半报：Llama +0.398 vs 基线 +0.369 |
| 数据量敏感性 | `results/results_tqa/E16/llama3.1-8b/iti_halfA.md` | 把 test A 半也喂给 ITI（926 条估计），B 半仍只 +0.404 vs +0.369、MC1 0.664 → 0.675（α5） |
| 逐题分数 | `results/results_tqa/E16/<model>/iti.npz` | 键：ids、labels、pair_ids、half_a、`<条件>__p`、`<条件>__pred` |

## 当时的实际命令行（摘自服务器队列/链脚本，已核对无密钥）

```bash
# 服务器 q_gpu2.sh（六模型 TQA；--mc 同时跑 MC1/MC2）
python3 scripts/e16_iti_baseline.py --exp E2b_tqa --e1-exp E1_directions_tqa --model <model> --k 48 --alphas 5 10 15 20 --mc
# Llama 数据量敏感性：把 test A 半并入估计（结果只看 B 半）
python3 scripts/e16_iti_baseline.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --k 48 --alphas 5 10 15 20 --mc --train-on-half-a --tag halfA
# BBQ 对照（两模型）
python3 scripts/e16_iti_baseline.py --exp E2b_bbq --e1-exp E1_directions_bbq --model <model> --k 48 --alphas 5 10 15 20 --mc --mc-dataset bbq_mc --exclude-dataset bbq_tf
# 自由生成 ITI 臂（E4b；读 results_tqa/E16/<model>/iti_heads.npz）
python3 scripts/e4b_qa_gated.py --phase generate --regime tqa --model <model> --batch-size 16 --conditions iti_k48_a5 iti_k48_a15
python3 scripts/e4b_qa_gated.py --phase judge --regime tqa --model <model> --workers 12
```

## 说明

- **估计 / 调参 / 测试**：估计（选头 + 方向 + σ）= calib+dev 96 对 192 条；α ∈ {5,10,15,20} 在 test 的 A 半按 penalised 选、B 半报（`iti.json.selection`）；没有独立 dev。MC 评测剔除 96 道估计题 → 721 题；halfA 变体再剔除 A 半题 → 354 题。
- **选头规则**（`core/iti.py` + `select_heads`）：对每层每头的最后 token o_proj 输入拟合逻辑回归探针，4 折按题分组 CV 准确率排序取前 48；方向 = 单位均值差（真 − 假），σ_h = 投影标准差；干预在每次前向（含每个解码步）的最后位置。
- 不需要 E1 的均值差方向（ITI 自带方向），所以本包不含 `directions_meandiff.npz`。
- Qwen3-0.6B 的基线 J=0、B=1.0（几乎全答 True），它的 MC1 '上涨' 是在这个退化基线上发生的，不能当正例。
- BBQ 对照只放了结果 + 日志，BBQ 的数据/配置文件未复制（不是 06 行 3.10 的数字）。

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
| `scripts/e16_iti_baseline.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e16_iti_baseline.py | 11.9 KB | **主脚本**：选头 → 陈述级四档 α → A/B 半 → 可选 MC1/MC2；写 iti.{json,md,npz} 与 iti_heads.npz | 3.10 全部数字 |
| `core/iti.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/iti.py | 6.3 KB | **ITI 实现**：按头抓 o_proj 输入、逐头探针选头、均值差方向与 σ、iti_steer 钩子（每次前向最后位置） | 干预机制 |
| `scripts/e11_dose_rates.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e11_dose_rates.py | 16.1 KB | 提供 rates()（correct/wrong/abstain/fixed/damaged/penalised/AUROC/J/B）被 e16 import | 依赖钩子：指标口径 |
| `scripts/e14_mc_eval.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e14_mc_eval.py | 11.6 KB | 提供 question_metrics()（MC1 / argmax / MC2）被 e16 import | 依赖钩子：MC 口径 |
| `scripts/e10_gating.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e10_gating.py | 14.2 KB | 提供 state()（known/wrong/abstain 三态）被 e14/e16 import | 依赖钩子 |
| `scripts/e4b_qa_gated.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/scripts/e4b_qa_gated.py | 20.7 KB | 自由生成（含 iti_k48_a5 / a15 臂，读 iti_heads.npz）+ DeepSeek 判分 | 自由生成为负 |
| `data_adapters/truthfulqa_mc.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/data_adapters/truthfulqa_mc.py | 1.9 KB | 每题每个候选答案改写成一条陈述（best / 其他正确 = True，错误 = False；meta.role） | 陈述读出口径的 MC1/MC2 |
| `configs/datasets/truthfulqa_mc.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/datasets/truthfulqa_mc.json | 0.8 KB | truthfulqa_mc 配置：全部 817 题、5882 候选，不切分；E14 评测时剔除估计题 | 721 题 / 5179 候选 |
| `data/processed/truthfulqa_mc/manifest.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa_mc/manifest.json | 0.7 KB | 5882 条候选陈述（True 2587 / False 3295） | 题 ID |
| `data/processed/truthfulqa_mc/test.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa_mc/test.jsonl | 3.2 MB | 全部候选陈述；pair_id = 题 id；评测时剔除 pair_id ∈ truthfulqa_tf calib+dev 的 96 题 → 721 题 / 5179 候选 | MC 评测题 ID（剔除后 721 题） |
| `configs/datasets/truthfulqa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/datasets/truthfulqa.json | 0.5 KB | TruthfulQA 问答集配置：dev 100 / test 717（按题切，seed 42） | 自由生成题 ID |
| `data/processed/truthfulqa/manifest.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa/manifest.json | 0.8 KB | 问答集切分计数与原始文件 sha1 | 自由生成题 ID |
| `data/processed/truthfulqa/dev.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa/dev.jsonl | 81.0 KB | 问答 dev 100 题（本次核对：96 道估计题全部落在这 100 题里） | — |
| `data/processed/truthfulqa/test.jsonl` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/data/processed/truthfulqa/test.jsonl | 583.9 KB | 问答 test 717 题（与 96 道估计题零重叠，本次已核；E4b 的 n=717） | 自由生成题 ID |
| `configs/experiments/E4_qa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/experiments/E4_qa.json | 1.1 KB | E4b 判分器配置（DeepSeek deepseek-chat，thinking 关；只写环境变量名，不含密钥） | 判分口径 |
| `core/judge.py` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/core/judge.py | 7.3 KB | OpenAI 兼容 LLM judge + 磁盘缓存；密钥只从环境变量/文件读 | e4b 判分依赖 |
| `configs/models/qwen3-8b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-8b.json | 0.5 KB | 模型配置（层数、chat 规则） | 六模型 |
| `configs/models/mistral-7b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/mistral-7b.json | 0.5 KB | 模型配置（层数、chat 规则） | 六模型 |
| `configs/models/qwen3-4b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-4b.json | 0.4 KB | 模型配置（层数、chat 规则） | 六模型 |
| `configs/models/qwen3-1.7b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-1.7b.json | 0.4 KB | 模型配置（层数、chat 规则） | 六模型 |
| `configs/models/qwen3-0.6b.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/experiments/configs/models/qwen3-0.6b.json | 0.4 KB | 模型配置（层数、chat 规则） | 六模型 |
| `results/results_tqa/E16/llama3.1-8b/iti.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti.md | 2.2 KB | **每模型结果表**：All items / Half B（A 半选 α）/ truthfulqa_mc 题级 | 陈述级 / MC 负或零 |
| `results/results_tqa/E16/mistral-7b/iti.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/mistral-7b/iti.md | 2.2 KB | **每模型结果表**：All items / Half B（A 半选 α）/ truthfulqa_mc 题级 | 陈述级 / MC 负或零 |
| `results/results_tqa/E16/qwen3-0.6b/iti.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-0.6b/iti.md | 2.1 KB | **每模型结果表**：All items / Half B（A 半选 α）/ truthfulqa_mc 题级 | 陈述级 / MC 负或零 |
| `results/results_tqa/E16/qwen3-1.7b/iti.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-1.7b/iti.md | 2.1 KB | **每模型结果表**：All items / Half B（A 半选 α）/ truthfulqa_mc 题级 | 陈述级 / MC 负或零 |
| `results/results_tqa/E16/qwen3-4b/iti.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-4b/iti.md | 2.1 KB | **每模型结果表**：All items / Half B（A 半选 α）/ truthfulqa_mc 题级 | 陈述级 / MC 负或零 |
| `results/results_tqa/E16/qwen3-8b/iti.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-8b/iti.md | 2.1 KB | **每模型结果表**：All items / Half B（A 半选 α）/ truthfulqa_mc 题级 | 陈述级 / MC 负或零 |
| `results/results_tqa/E16/llama3.1-8b/iti.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti.json | 14.1 KB | 同上机器可读 + **heads（选头记录）** + selection | 选头记录 / 强度选择 |
| `results/results_tqa/E16/mistral-7b/iti.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/mistral-7b/iti.json | 14.0 KB | 同上机器可读 + **heads（选头记录）** + selection | 选头记录 / 强度选择 |
| `results/results_tqa/E16/qwen3-0.6b/iti.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-0.6b/iti.json | 13.3 KB | 同上机器可读 + **heads（选头记录）** + selection | 选头记录 / 强度选择 |
| `results/results_tqa/E16/qwen3-1.7b/iti.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-1.7b/iti.json | 13.5 KB | 同上机器可读 + **heads（选头记录）** + selection | 选头记录 / 强度选择 |
| `results/results_tqa/E16/qwen3-4b/iti.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-4b/iti.json | 13.7 KB | 同上机器可读 + **heads（选头记录）** + selection | 选头记录 / 强度选择 |
| `results/results_tqa/E16/qwen3-8b/iti.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-8b/iti.json | 13.6 KB | 同上机器可读 + **heads（选头记录）** + selection | 选头记录 / 强度选择 |
| `results/results_tqa/E16/llama3.1-8b/iti.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti.npz | 358.3 KB | **逐题**：ids、labels、pair_ids、half_a、每条件 P(True)/判定 | 逐题分数 |
| `results/results_tqa/E16/mistral-7b/iti.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/mistral-7b/iti.npz | 358.3 KB | **逐题**：ids、labels、pair_ids、half_a、每条件 P(True)/判定 | 逐题分数 |
| `results/results_tqa/E16/qwen3-0.6b/iti.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-0.6b/iti.npz | 358.3 KB | **逐题**：ids、labels、pair_ids、half_a、每条件 P(True)/判定 | 逐题分数 |
| `results/results_tqa/E16/qwen3-1.7b/iti.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-1.7b/iti.npz | 358.3 KB | **逐题**：ids、labels、pair_ids、half_a、每条件 P(True)/判定 | 逐题分数 |
| `results/results_tqa/E16/qwen3-4b/iti.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-4b/iti.npz | 346.8 KB | **逐题**：ids、labels、pair_ids、half_a、每条件 P(True)/判定 | 逐题分数 |
| `results/results_tqa/E16/qwen3-8b/iti.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-8b/iti.npz | 358.3 KB | **逐题**：ids、labels、pair_ids、half_a、每条件 P(True)/判定 | 逐题分数 |
| `results/results_tqa/E16/llama3.1-8b/iti_heads.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_heads.npz | 27.4 KB | 选中头 (层,头)、验证准确率、方向、σ（E4b 的 iti 臂直接读它） | 选头记录 |
| `results/results_tqa/E16/mistral-7b/iti_heads.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/mistral-7b/iti_heads.npz | 27.4 KB | 选中头 (层,头)、验证准确率、方向、σ（E4b 的 iti 臂直接读它） | 选头记录 |
| `results/results_tqa/E16/qwen3-0.6b/iti_heads.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-0.6b/iti_heads.npz | 27.4 KB | 选中头 (层,头)、验证准确率、方向、σ（E4b 的 iti 臂直接读它） | 选头记录 |
| `results/results_tqa/E16/qwen3-1.7b/iti_heads.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-1.7b/iti_heads.npz | 27.4 KB | 选中头 (层,头)、验证准确率、方向、σ（E4b 的 iti 臂直接读它） | 选头记录 |
| `results/results_tqa/E16/qwen3-4b/iti_heads.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-4b/iti_heads.npz | 27.4 KB | 选中头 (层,头)、验证准确率、方向、σ（E4b 的 iti 臂直接读它） | 选头记录 |
| `results/results_tqa/E16/qwen3-8b/iti_heads.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/qwen3-8b/iti_heads.npz | 27.4 KB | 选中头 (层,头)、验证准确率、方向、σ（E4b 的 iti 臂直接读它） | 选头记录 |
| `results/results_tqa/E16/llama3.1-8b/iti_halfA.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_halfA.json | 13.7 KB | Llama 数据量敏感性（估计 + test A 半 = 926 条；只看 B 半） | 数据量敏感性 |
| `results/results_tqa/E16/llama3.1-8b/iti_halfA.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_halfA.md | 2.2 KB | Llama 数据量敏感性（估计 + test A 半 = 926 条；只看 B 半） | 数据量敏感性 |
| `results/results_tqa/E16/llama3.1-8b/iti_halfA.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_halfA.npz | 323.7 KB | Llama 数据量敏感性（估计 + test A 半 = 926 条；只看 B 半） | 数据量敏感性 |
| `results/results_tqa/E16/llama3.1-8b/iti_heads_halfA.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_heads_halfA.npz | 27.4 KB | halfA 变体的选头 | 辅助 |
| `results/results_tqa/E16/llama3.1-8b/iti_smoke.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_smoke.json | 9.2 KB | 冒烟跑（小样本），仅存档 | — |
| `results/results_tqa/E16/llama3.1-8b/iti_smoke.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_smoke.md | 1.6 KB | 冒烟跑（小样本），仅存档 | — |
| `results/results_tqa/E16/llama3.1-8b/iti_smoke.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_smoke.npz | 12.5 KB | 冒烟跑（小样本），仅存档 | — |
| `results/results_tqa/E16/llama3.1-8b/iti_heads_smoke.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E16/llama3.1-8b/iti_heads_smoke.npz | 27.4 KB | 冒烟跑选头，仅存档 | — |
| `results/results_bbq/E16/llama3.1-8b/iti.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E16/llama3.1-8b/iti.json | 13.7 KB | BBQ 对照（Llama、Qwen3-8B） | 辅助 |
| `results/results_bbq/E16/llama3.1-8b/iti.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E16/llama3.1-8b/iti.md | 2.1 KB | BBQ 对照（Llama、Qwen3-8B） | 辅助 |
| `results/results_bbq/E16/llama3.1-8b/iti.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E16/llama3.1-8b/iti.npz | 593.0 KB | BBQ 对照（Llama、Qwen3-8B） | 辅助 |
| `results/results_bbq/E16/qwen3-8b/iti.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E16/qwen3-8b/iti.json | 14.0 KB | BBQ 对照（Llama、Qwen3-8B） | 辅助 |
| `results/results_bbq/E16/qwen3-8b/iti.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E16/qwen3-8b/iti.md | 2.1 KB | BBQ 对照（Llama、Qwen3-8B） | 辅助 |
| `results/results_bbq/E16/qwen3-8b/iti.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E16/qwen3-8b/iti.npz | 593.0 KB | BBQ 对照（Llama、Qwen3-8B） | 辅助 |
| `results/results_bbq/E16/llama3.1-8b/iti_heads.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E16/llama3.1-8b/iti_heads.npz | 27.4 KB | BBQ 对照选头 | 辅助 |
| `results/results_bbq/E16/qwen3-8b/iti_heads.npz` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_bbq/E16/qwen3-8b/iti_heads.npz | 27.4 KB | BBQ 对照选头 | 辅助 |
| `results/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.json | 11.6 KB | 自由生成汇总（iti_k48_a5 / a15 行 + 增益 CI） | 自由生成为负 |
| `results/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/summary_truthfulqa.md | 3.4 KB | 自由生成汇总（iti_k48_a5 / a15 行 + 增益 CI） | 自由生成为负 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_baseline.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_baseline.json | 768.4 KB | 逐题生成文本 + DeepSeek 判分（baseline） | 自由生成逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_iti_k48_a5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_iti_k48_a5.json | 768.3 KB | 逐题生成文本 + DeepSeek 判分（iti_k48_a5） | 自由生成逐题 |
| `results/results_tqa/E4b/llama3.1-8b/truthfulqa_iti_k48_a15.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/llama3.1-8b/truthfulqa_iti_k48_a15.json | 772.3 KB | 逐题生成文本 + DeepSeek 判分（iti_k48_a15） | 自由生成逐题 |
| `results/results_tqa/E4b/qwen3-8b/summary_truthfulqa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/qwen3-8b/summary_truthfulqa.json | 11.6 KB | 自由生成汇总（iti_k48_a5 / a15 行 + 增益 CI） | 自由生成为负 |
| `results/results_tqa/E4b/qwen3-8b/summary_truthfulqa.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/qwen3-8b/summary_truthfulqa.md | 3.4 KB | 自由生成汇总（iti_k48_a5 / a15 行 + 增益 CI） | 自由生成为负 |
| `results/results_tqa/E4b/qwen3-8b/truthfulqa_baseline.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/qwen3-8b/truthfulqa_baseline.json | 766.6 KB | 逐题生成文本 + DeepSeek 判分（baseline） | 自由生成逐题 |
| `results/results_tqa/E4b/qwen3-8b/truthfulqa_iti_k48_a5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/qwen3-8b/truthfulqa_iti_k48_a5.json | 767.1 KB | 逐题生成文本 + DeepSeek 判分（iti_k48_a5） | 自由生成逐题 |
| `results/results_tqa/E4b/qwen3-8b/truthfulqa_iti_k48_a15.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/qwen3-8b/truthfulqa_iti_k48_a15.json | 770.8 KB | 逐题生成文本 + DeepSeek 判分（iti_k48_a15） | 自由生成逐题 |
| `results/results_tqa/E4b/mistral-7b/summary_truthfulqa.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/mistral-7b/summary_truthfulqa.json | 11.6 KB | 自由生成汇总（iti_k48_a5 / a15 行 + 增益 CI） | 自由生成为负 |
| `results/results_tqa/E4b/mistral-7b/summary_truthfulqa.md` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/mistral-7b/summary_truthfulqa.md | 3.3 KB | 自由生成汇总（iti_k48_a5 / a15 行 + 增益 CI） | 自由生成为负 |
| `results/results_tqa/E4b/mistral-7b/truthfulqa_baseline.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/mistral-7b/truthfulqa_baseline.json | 836.8 KB | 逐题生成文本 + DeepSeek 判分（baseline） | 自由生成逐题 |
| `results/results_tqa/E4b/mistral-7b/truthfulqa_iti_k48_a5.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/mistral-7b/truthfulqa_iti_k48_a5.json | 829.3 KB | 逐题生成文本 + DeepSeek 判分（iti_k48_a5） | 自由生成逐题 |
| `results/results_tqa/E4b/mistral-7b/truthfulqa_iti_k48_a15.json` | local:/Users/airsanshi/MyPapers/Truth_Decoupling/results_tqa/E4b/mistral-7b/truthfulqa_iti_k48_a15.json | 890.0 KB | 逐题生成文本 + DeepSeek 判分（iti_k48_a15） | 自由生成逐题 |
| `server_logs/e16_tqa_llama3.1-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_tqa_llama3.1-8b.log | 2.0 KB | E16 运行 stdout：头行含 192 条估计、头数×层数、选中头验证准确率范围与所在层 | 运行记录 / 选头记录 |
| `server_logs/e16_tqa_llama_halfA.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_tqa_llama_halfA.log | 2.0 KB | E16 运行 stdout：头行含 192 条估计、头数×层数、选中头验证准确率范围与所在层 | 运行记录 / 选头记录 |
| `server_logs/e16_tqa_mistral-7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_tqa_mistral-7b.log | 2.0 KB | E16 运行 stdout：头行含 192 条估计、头数×层数、选中头验证准确率范围与所在层 | 运行记录 / 选头记录 |
| `server_logs/e16_tqa_qwen3-0.6b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_tqa_qwen3-0.6b.log | 1.9 KB | E16 运行 stdout：头行含 192 条估计、头数×层数、选中头验证准确率范围与所在层 | 运行记录 / 选头记录 |
| `server_logs/e16_tqa_qwen3-1.7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_tqa_qwen3-1.7b.log | 2.0 KB | E16 运行 stdout：头行含 192 条估计、头数×层数、选中头验证准确率范围与所在层 | 运行记录 / 选头记录 |
| `server_logs/e16_tqa_qwen3-4b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_tqa_qwen3-4b.log | 2.0 KB | E16 运行 stdout：头行含 192 条估计、头数×层数、选中头验证准确率范围与所在层 | 运行记录 / 选头记录 |
| `server_logs/e16_tqa_qwen3-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_tqa_qwen3-8b.log | 2.0 KB | E16 运行 stdout：头行含 192 条估计、头数×层数、选中头验证准确率范围与所在层 | 运行记录 / 选头记录 |
| `server_logs/e16_bbq_llama3.1-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_bbq_llama3.1-8b.log | 2.0 KB | BBQ 对照运行 stdout | 辅助 |
| `server_logs/e16_bbq_qwen3-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e16_bbq_qwen3-8b.log | 2.0 KB | BBQ 对照运行 stdout | 辅助 |
| `server_logs/e4b_tqa_llama3.1-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e4b_tqa_llama3.1-8b.log | 3.3 KB | E4b 生成阶段 stdout（含 ITI heads loaded 行） | 运行记录 |
| `server_logs/e4b_tqa_mistral-7b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e4b_tqa_mistral-7b.log | 1.9 KB | E4b 生成阶段 stdout（含 ITI heads loaded 行） | 运行记录 |
| `server_logs/e4b_tqa_qwen3-8b.log` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/e4b_tqa_qwen3-8b.log | 3.2 KB | E4b 生成阶段 stdout（含 ITI heads loaded 行） | 运行记录 |
| `server_logs/q_gpu2.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_gpu2.sh | 1.0 KB | **E16 队列脚本**：六模型 + halfA + BBQ 的完整命令行 | 命令行 |
| `server_logs/q_gpu0.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_gpu0.sh | 0.8 KB | E4b Qwen 生成 + ITI 臂命令 | 命令行 |
| `server_logs/q_gpu3.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_gpu3.sh | 1.5 KB | E4b Llama 生成 + ITI 臂命令 | 命令行 |
| `server_logs/q_gpu3b.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_gpu3b.sh | 0.9 KB | E4b Mistral 生成与判分命令 | 命令行 |
| `server_logs/q_judge.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_judge.sh | 1.7 KB | E4b 判分命令（密钥走 *_FILE，脚本内无密钥） | 命令行 |
| `server_logs/q_judge2.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/q_judge2.sh | 1.9 KB | 判分第二轮命令 | 命令行 |
| `server_logs/common.sh` | pro5000:/data/sanshizhao/Truth_Decoupling/logs/common.sh | 0.5 KB | dex() 容器环境变量 | 运行环境 |

## 缺口（GPT 要而没有 / 未复制）

- 原论文协议的任何材料（Q/A 少样本提示、官方按答案对数概率的 MC1/MC2、GPT-judge 自由生成）：没有，本包全部是我们协议下的自实现。
- 独立调参集题 ID：不存在（见说明）。
- BBQ 对照的数据文件（`data/processed/bbq_tf`、`bbq_mc`）与配置（`E2b_bbq.json`、`E1_directions_bbq.json`）未复制；BBQ 的 E4b 自由生成（`results_bbq/E4b`）未复制。
- E4b 判分缓存 `results/judge_cache/` 未复制（判分已在各 json 的 evaluations[].judge）。
- 当时的报告段落（`docs/给〈老师〉_不积极点清单与攻克_20261027.md` ITI 小节、行 154 附近）文件名含老师名字，未复制；数字可在本包 `iti.md` / `summary_truthfulqa.md` 全部找到。

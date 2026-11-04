# experiments/ — core 与周边的契约

> 换模型、换数据集、换提示词、换实验参数 = **只改 `configs/` 和 `data_adapters/`**，不动 `core/`。
> core 出 bug = 最小改动修复 + 回归测试那个修复，不重写。

## 目录

```
core/              稳定核心（不随模型/数据集变化）
  config.py        JSON 配置加载，${ENV:-default} 展开，深合并
  io_utils.py      原子写 JSON/JSONL、seed、实验命名、断点续跑 ResumableResults
  model_io.py      加载模型/分词器、拼 chat 模板（含 Qwen3 thinking stub）、按层抓最后 token 激活（resid + attn）
  directions.py    方向估计：均值差 / 逻辑回归探针（numpy IRLS）/ rank-r DAS（hook 实现，数学同 v12 pyvene 版）/ Haar 随机方向
  intervene.py     推理时 hook 干预：add / amplify / interchange / clamp，多方向多层，逐样本 α
  layer_score.py   LA-Score 免训练选层（dact, dattn, sep, mass），读层 R*、写层 W*、Δ
  readout.py       陈述级读出（True/False/Unknown 候选词首 token 对数概率）、弃答规则正则、严格解析
  metrics.py       J、B、作答率、2×2 解耦矩阵、纠正/损伤/转弃答三分、Wilson/bootstrap/McNemar、Spearman、AUROC、ECE、risk-coverage
  judge.py         OpenAI 兼容 LLM judge（DeepSeek 等）+ 磁盘缓存；SimpleQA 式三分评分；弃答复核
  gate.py          知识探针 + 逐输入门控（τ_lo/τ_hi, α_E(x), α_C(x)）+ dev 上带约束的阈值搜索
configs/
  models/*.json        路径（${TD_MODEL_ROOT}）、层数、chat 规则
  datasets/*.json      来源 URL、抽样、切分（seed 固定 42）
  prompts/*.json       陈述判定提示、承诺对比 system prompt、QA 提示、候选词变体
  experiments/*.json   E1/E2/…：引用 model/prompts/datasets 名 + 实验超参
data_adapters/     各公开数据集 → 统一记录 schema（见 base.py 顶部）
scripts/           薄驱动：prepare_data / smoke_test / e1_directions / e2_dissociation（E3–E9 续加）
tests/             numpy 级单测（本机可跑）：python3 tests/test_core_numpy.py
```

## 数学口径（与 v12 一致）

- 激活 = 第 ℓ 层 decoder block 输出（残差流 hidden_states[ℓ+1]）最后一个 prompt token。
- rank-1 DAS：`h' = h + ((h_src − h)·w) w`，多步 teacher-forced 前向 KL `KL(p_src ‖ p_int)`，T=8，10 epoch，Adam 1e-3，梯度裁剪 1.0。
- 随机方向对照：Haar QR，范数匹配。
- 陈述级：J = TPR + TNR − 1（判别力），B = P(说 True | 作答)（偏置），R_ans = 作答率。
- `add` 模式的 α 单位 = 该方向在训练集上的平均投影差（gap），α=1 ≈ 一个均值差。
- `amplify` 模式：`h' = h + α · w wᵀ (h − center)`，放大模型自己沿 w 的信号，不加常量偏移。

## 运行（服务器容器 `truth_dec`，工作目录 `/workspace/Truth_Decoupling/code/experiments`）

进容器统一用宿主机用户身份，避免 root 写出 ubuntu 删不掉的结果文件：

```bash
DEX="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e CUDA_VISIBLE_DEVICES=0 \
  -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed \
  -w /workspace/Truth_Decoupling/code/experiments truth_dec"
$DEX python3 scripts/smoke_test.py --model qwen3-8b --dataset counterfact --layers 10,18,24 --das
```

长任务一律在服务器上 `nohup bash logs/run_xxx.sh > logs/xxx.log 2>&1 &`，别挂在 Mac 的 ssh 会话上。

```bash
# 数据（Mac 可直连的先在 Mac 跑，再 rsync；mmlu 需 datasets 库，在容器里跑）
python3 scripts/prepare_data.py --datasets got counterfact fever truthfulqa simpleqa halueval_qa gsm8k
# 冒烟
python3 scripts/smoke_test.py --model qwen3-8b --dataset counterfact --layers 10,18,24 --das
# E1 / E2
python3 scripts/e1_directions.py --exp E1_directions --model qwen3-8b --das-topk 3
python3 scripts/e2_dissociation.py --exp E2_dissociation --model qwen3-8b
```

环境变量：`TD_MODEL_ROOT`（默认 `/workspace/Truth_Decoupling/models`）、`TD_DATA_DIR`、`TD_RESULTS_DIR`、`DEEPSEEK_API_KEY`。

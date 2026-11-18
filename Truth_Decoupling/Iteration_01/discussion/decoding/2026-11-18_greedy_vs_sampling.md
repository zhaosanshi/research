# 生成方式：贪心还是取样

撰写：赵磊（Fable 5.2 代笔）。起始日期：2026-11-18（北京时间）。

文献核查由 Opus 5.5 子代理于 2026-11-17 完成，Fable 5.2 抽查。下文路径均相对于仓库的 `Truth_Decoupling/`。

## 1. 起因

- 赵磊 2026-11-17 19 时："我们是不是所有模型都用错了，这个你要看下其他顶会用gemma3，qwen，llama是怎么用的，是贪心，还是temperature 1.0、top_p 0.95、top_k 64。"
- 赵磊 2026-11-18："你综合下给出方案，这个影响整个论文。"

## 2. 本项目的做法

- 所有模型的回答都用贪心生成：`Iteration_01/code/core/model_io.py` 的 `greedy_generate` 调用 `model.generate(do_sample=False)`。模型自带生成配置中的其余各项（终止词元、屏蔽词元等）照常生效。
- 裁判 DeepSeek Pro 使用 temperature 0、非思考模式。

## 3. 官方的默认配置与建议

| 模型 | 随权重附带的生成配置 | 模型卡的说法 |
|---|---|---|
| Llama-3.1-8B-Instruct | 取样，temperature 0.6，top_p 0.9 | 没有关于取样的建议 |
| Qwen3.5-9B | 未附带生成配置文件 | 建议取样；不思考模式的一般任务为 temperature 0.7、top_p 0.8、top_k 20 |
| Qwen3.8-27B | 取样，temperature 1.0，top_k 20，top_p 0.95 | 建议取样；不思考模式为 temperature 0.7、top_p 0.80、top_k 20 |
| Gemma-3-12B、Gemma-3-27B | 取样，top_k 64，top_p 0.95 | 示例代码用贪心：`model.generate(**inputs, max_new_tokens=100, do_sample=False)` |
| Gemma-4-12B、Gemma-4-31B | 取样，temperature 1.0，top_k 64，top_p 0.95 | "Use the following standardized sampling configuration across all use cases"，数值同左 |

- 以上取自服务器模型目录中的 `generation_config.json` 与 `README.md`（官方模型卡）。
- Qwen3 公开模型卡（Qwen3-8B，Best Practices 一节）写有 "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions."，这句话只写在思考模式之下；不思考模式只给了取样参数的建议。本项目的 Qwen 模型都以不思考模式运行。

## 4. 顶会论文的做法

**核查方式**：下载原文（arXiv 或正式版）后抽取文字，按 greedy、temperature、sampling、decod 等关键词检索，逐字摘录写明生成设置的句子。共 25 篇，其中 18 篇写明了自由生成回答的设置，7 篇未写明。Fable 5.2 用脚本在抽取出的原文文字中核对了其中 14 条摘录，全部找到。多数论文读的是 arXiv 最新版，不一定是会议定稿。

### 4.1 被评测的回答用贪心生成（11 篇）

| 论文 | 会议与年份 | 模型 | 原文 |
|---|---|---|---|
| TruthfulQA（Lin et al.） | ACL 2022 | GPT-3 等 | "Answers are generated using greedy decoding (i.e. temperature set to zero)."；附录另以 temperature 1 重做，"higher temperatures are not substantially changing performance trends." |
| Inference-Time Intervention（Li et al.） | NeurIPS 2023 | LLaMA-7B 等 | "the model generates an answer to each question with greedy autoregressive decoding." |
| DoLa（Chuang et al.） | ICLR 2024 | LLaMA | "we use greedy decode for TruthfulQA, StrategyQA, and GSM8K. For the Vicuna QA Benchmark, we use random sampling with temperature 0.7" |
| TruthX | ACL 2024 | Llama-2-7B-Chat 等 | "The model generates responses to questions directly using greedy decoding." |
| Refusal is mediated by a single direction（Arditi et al.） | NeurIPS 2024 | Llama-3、Gemma、Qwen 等 | "we always use greedy decoding and a maximum generation length of 512 tokens, as suggested in Mazeika et al. (2024)." |
| Don't Hallucinate, Abstain | ACL 2024 | Mistral-7B、LLaMA2-70B 等 | "zeroshot prompting and greedy decoding to obtain the ground truth of whether the LLM 'knows' the answer"；另有 "We set the default sampling temperature to 0.1, and employ 0.7 where multiple runs are required." |
| LLMs Know More Than They Show（Orgad et al.） | ICLR 2025 | Mistral-7B、Llama-3-8B 等 | "We use representations extracted from the answers produced via greedy decoding."；错误分类另以 temperature 1 每题取样 30 次 |
| Improving Instruction-Following via Activation Steering（Stolfo et al.） | ICLR 2025 | Phi-3、Gemma 2、Mistral 7B | "Model outputs are decoded greedily" |
| SpARE | NAACL 2025 | Llama3-8B、Gemma2-9B 等 | "We use the greedy decoding method to evaluate the LLMs in an open-ended generation setting." |
| TruthFlow | ICML 2025 | Llama-3-8B-Instruct、Gemma-2-9B-it 等 | "In all the open-ended generation tasks, we utilize the greedy decoding strategy to generate new tokens." |
| The Geometry of Refusal（Wollschläger et al.） | ICML 2025 | Gemma 2、Qwen2.5、Llama-3 | "Model completions for evaluation are generated using greedy decoding with a maximum generation length of 512 tokens." |

### 4.2 被评测的回答用取样生成（7 篇）

| 论文 | 会议与年份 | 任务 | 原文 |
|---|---|---|---|
| Semantic Uncertainty（Kuhn et al.） | ICLR 2023 | 不确定性估计 | 用于估计的回答集合以多项式取样生成；用于判正确率的回答以束搜索生成 |
| Farquhar et al. | Nature 2024 | 幻觉检测 | "we sample at temperature 1 using nucleus sampling (P = 0.9) and top-K sampling (K = 50). We also sample a single generation at low temperature (0.1) as an estimate of the 'best generation'" |
| INSIDE | ICLR 2024 | 幻觉检测 | "we set temperature to 0.5, top-p to 0.99 and top-k to 5 … The number of generations is set to K = 10." |
| Kernel Language Entropy | NeurIPS 2024 | 不确定性估计 | "We sample 10 answers per input … at temperature T = 1. To assess model accuracy, we draw an additional low-temperature sample (T = 0.1)" |
| AxBench | ICML 2025 | 写入方向的评测（Gemma-2） | "We use the default decoding strategy (i.e., setting the decoding temperature to 1.0) … We believe that this setup reflects the typical user interaction with language models. However, this is not a common practice in representation-based model steering." |
| SAEs Are Good for Steering（Arad et al.） | EMNLP 2025 | 稀疏自编码器写入（Gemma-2） | "We generate 20 tokens using a temperature of 0.7" |
| AbstentionBench | NeurIPS 2025（数据集与基准） | 弃答（Llama 3.1 / 3.3、Qwen 等） | "Responses are sampled using temperature 0.8, Top-p sampling with p = 0.95, and a fixed random seed."；裁判用贪心 |

### 4.3 原文未写明（7 篇）

Steering Llama 2 via Contrastive Activation Addition（ACL 2024）、Representation Engineering、Do I Know This Entity?（ICLR 2025）、Programming Refusal with Conditional Activation Steering（ICLR 2025）、R-Tuning 的主评测（NAACL 2024）、Spectral Editing of Activations（NeurIPS 2024）、Refusal Feature Adversarial Training（ICLR 2025）。

### 4.4 归纳

1. **真实性评测（TruthfulQA 的生成任务）与写入方向这两类工作，主评测以贪心为通例**，其中包括使用 Llama-3、Qwen、Gemma-2 的 2024 至 2025 年论文。
2. **取样的 7 篇中有 4 篇是不确定性估计或幻觉检测**，方法本身要求同一题生成多份回答；这 4 篇中用来判正确率的那一份，3 篇用束搜索或低温度（0.1）。
3. **另 3 篇直接用取样**：AxBench 的理由是取样更接近用户的实际使用，并承认这在写入方向的研究中不是通例；Arad et al. 用 temperature 0.7，未说明理由；AbstentionBench（与本项目同为弃答题目）用 temperature 0.8。
4. TruthfulQA 原论文在附录中以 temperature 1 重做，报告趋势不变。

## 5. 结论

1. **本项目没有用错**。贪心是这两类工作的通行做法，也是 TruthfulQA 原协议的做法。
2. **各模型官方的默认配置都是取样**，Gemma-4 的模型卡要求所有用途都取样。论文中需要写明本项目用贪心，并说明理由。
3. 少数近期工作（AxBench、AbstentionBench）用取样，其中一篇与本项目题目相同，审稿人可能据此提问。

## 6. 建议的做法（待赵磊决定）

- **正文沿用贪心**，不重跑已有结果。写明："所有回答以贪心方式生成，与 TruthfulQA 及多数写入方向的工作一致"，并引第 4.1 节的工作。理由有两条：结果可逐字复现；同一道题在写入前后的差别只来自写入，不混入取样的随机性。
- **附录加一项取样下的稳健性检查**，仿照 TruthfulQA 附录的做法：方案定下之后，在两到三个模型上，按各自官方的取样配置每题生成若干次，报告平均得分，看"方案优于单独使用 E"是否仍成立。需要 GPU 与判分费用，范围与次数届时再定。
- **不建议**把全部实验改为取样重跑：与通行做法不符，且每个比较都要多次生成与判分。

## 7. 来源

- 论文原文的 arXiv 编号：2109.07958、2306.03341、2309.03883、2402.17811、2406.11717、2402.00367、2410.02707、2410.12877、2410.15999、2502.04556、2502.17420、2302.09664、2402.03744、2405.20003、2501.17148、2505.20063、2506.09038；Farquhar et al. 见 <https://www.nature.com/articles/s41586-024-07421-0>。
- 未写明的 7 篇：2312.06681、2310.01405、2411.14257、2409.05907、2311.09677、2405.09719、2409.20089。
- Qwen3 模型卡：<https://huggingface.co/Qwen/Qwen3-8B>。其余模型卡与生成配置取自服务器上的模型目录。

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-18 | 建立本文件 | 赵磊（Fable 5.2 代笔） |

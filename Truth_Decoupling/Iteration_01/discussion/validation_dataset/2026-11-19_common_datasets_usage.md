# 常用问答题集在相近顶会论文中的实际用法

撰写：赵磊（Fable 5.2 代笔）。起始日期：2026-11-19（北京时间）。

赵磊 2026-11-18 23 时："FalseQA 根本不常用，没意义。还是要找常用数据集，除了popQA和simpleQA，还有常用的吗，一定要用常用的。"

本文第 1 至 5 节只列事实，由 Opus 5.5 子代理读 26 篇论文的实验部分整理；第 6 节是我们的建议。核对情况见第 7 节。

**名词**
- **闭卷自由回答**：只给问题，不给材料，也不给选项，模型自己写出答案。这是本文方法的评测形式。
- **选择题形式**：给出候选答案，看模型选哪个，或看哪个候选的概率更高。
- **给材料**：把含有答案的段落一并给模型。

## 1. 计数：26 篇论文中各题集被用了几次

26 篇分三组：让模型更如实的方向引导与解码 8 篇，弃答与知识边界 9 篇，不确定性估计与幻觉检测 9 篇（清单见第 3 节）。

| 题集 | 闭卷自由回答 | 任何形式 | 引用数 |
|---|---|---|---|
| **TriviaQA** | **15** | 18 | 4,720 |
| TruthfulQA（已用） | 10 | 12 | 4,358 |
| **Natural Questions（NQ-open）** | **9** | 14 | 5,561（NQ-open 的出处另有 1,270） |
| MMLU | 0 | 10 | 10,609 |
| CoQA | 0 | 5（都给材料） | 1,685 |
| SQuAD | 2（去掉材料后） | 4 | 10,286 |
| SciQ | 3 | 3 | 1,059 |
| HotpotQA | 2 | 3 | 6,143 |
| BioASQ | 2 | 2 | 896 |
| PopQA | 1 | 1 | 1,690 |
| SimpleQA | 0 | 0 | 404 |
| EntityQuestions、WebQuestions、2WikiMultihopQA | 0 | 0 | 252、2,333、1,962 |
| FalseQA（已否决） | 未单独计 | 2（R-Tuning、AbstentionBench，只看拒答） | 64 |

引用数为 Semantic Scholar 2026-11-18 的数字。SimpleQA 2024 年 11 月才发布，26 篇中多数早于它。26 篇里用 PopQA 的只有 [IDK] 词元一篇。

## 2. 几条和选题集直接有关的事实

1. **做方向引导的四篇（ITI、TruthX、LoFiT、Truth Forest）都用 TriviaQA 和 NQ 检验推广，但都改成了选择题形式**，不是自由回答。这一组里在 NQ、TriviaQA 上做自由回答的只有 TruthFlow（ICML 2025）与 LITO（Findings of ACL 2024）。
2. **TriviaQA 的测试集答案不公开**，各篇都用验证集，常见的是去重后的 9,960 题。
3. **"答对 +1、不答 0、答错 −1"在顶会论文中有先例**："Don't Hallucinate, Abstain"（ACL 2024）的指标是答对的比例减答错的比例；TruthRL（ICML 2026）的奖励按对、不确定、错三档给，并在 NQ、HotpotQA 上做闭卷评测。
4. **把对、错、弃答分三类计的**还有 Alignment for Honesty（NeurIPS 2024，TriviaQA 与 NQ 的无歧义子集）、CRaFT（AAAI 2025，TriviaQA 与 NQ）、[IDK] 词元论文（NeurIPS 2024，TriviaQA 与 PopQA）。
5. Nature 上的语义熵论文把 SQuAD、BioASQ 的材料去掉、闭卷使用，原文："we do not provide the context passages for any of the datasets"；每个题集只抽 400 题测试。
6. AbstentionBench（NeurIPS 2025）的 20 个题集中没有 TriviaQA、NQ。

## 3. 逐篇用法

### 3.1 方向引导、激活编辑、事实性解码

| 论文 | 题集与形式 | 弃答是否单列 | 模型 |
|---|---|---|---|
| ITI（NeurIPS 2023） | TruthfulQA 生成与选择题；NQ、TriviaQA、MMLU 为选择题形式 | 否 | LLaMA-7B 一类 |
| DoLa（ICLR 2024） | TruthfulQA 选择题与生成；FACTOR 选择题；StrategyQA、GSM8K | TruthfulQA 生成部分报拒答比例 | LLaMA-7B 至 65B |
| TruthX（ACL 2024） | TruthfulQA；NQ、TriviaQA、FACTOR 全部为选择题形式 | 否 | Llama-2-7B-Chat 为主 |
| LoFiT（NeurIPS 2024） | TruthfulQA 等；TriviaQA、NQ 改成选择题 | 否 | Gemma-7B、Llama-2-7B / 13B |
| Truth Forest（AAAI 2024） | TruthfulQA；NQ 选择题 | 否 | 7B 一类 |
| RepE（预印本） | TruthfulQA 选择题及其他选择题集 | 否 | LLaMA-2 7B / 13B / 70B |
| TruthFlow（ICML 2025） | TruthfulQA 生成与选择题；HaluEval-QA、NQ（3,610 题）、TriviaQA 闭卷自由回答 | 否 | Llama-2 / 3、Mistral-7B、Gemma-2-9B |
| LITO（Findings of ACL 2024） | NQ、TriviaQA、SciQ、TruthfulQA 闭卷自由回答 | 是：表示不确定的回答计入"如实" | 未记录 |

### 3.2 弃答与知识边界

| 论文 | 题集与形式 | 弃答是否单列 | 模型 |
|---|---|---|---|
| R-Tuning（NAACL 2024） | ParaRel 闭卷自由回答；MMLU 选择题；HotpotQA 给材料；SelfAware、FalseQA 等只报拒答率 | 是 | 3B、7B、13B |
| Alignment for Honesty（NeurIPS 2024） | TriviaQA（训练 8,000 题，评测 9,960 题）、NQ 的无歧义子集（5,325 题）等，闭卷自由回答 | 是：谨慎、过度保守、诚实三个指标 | LLaMA2-Chat-13B 为主 |
| [IDK] 词元（NeurIPS 2024） | TriviaQA、PopQA，改写成句子补全，闭卷 | 是 | Mistral-7B |
| Don't Hallucinate, Abstain（ACL 2024） | MMLU 等选择题，AmbigQA | 是：答对比例减答错比例 | Mistral-7B、LLaMA2-70B 等 |
| Do I Know This Entity?（ICLR 2025） | 自建的维基数据实体问题，闭卷 | 是 | Gemma 2 2B / 9B |
| AbstentionBench（NeurIPS 2025） | 20 个题集，含 SQuAD 2.0、FreshQA、FalseQA | 是：该弃答时是否弃答 | 未记录 |
| TruthRL（ICML 2026） | CRAG、NQ、HotpotQA、MuSiQue；分不带检索（闭卷）与带检索 | 是：对、不确定、错三档奖励 | Llama3.1-8B、Qwen2.5-7B 至 32B |
| CRaFT（AAAI 2025） | TriviaQA 验证集 11,313 题；NQ 3,610 题作域外测试；闭卷 | 是 | LLaMA2-7B、LLaMA3-8B |
| Unfamiliar Finetuning Examples（NAACL 2025） | TriviaQA 闭卷短答，可以答"不知道" | 是：对 +2、错 −3 | Llama 2 7B |

### 3.3 不确定性估计与幻觉检测

| 论文 | 题集与形式 | 模型 |
|---|---|---|
| Semantic Uncertainty（ICLR 2023） | TriviaQA 闭卷（8,000 题）；CoQA 给材料 | OPT 2.7B 至 30B |
| 语义熵（Nature 2024） | TriviaQA、SQuAD 1.1、BioASQ、NQ-open、SVAMP，都不给材料 | LLaMA 2 Chat 7B / 13B / 70B、Falcon、Mistral 7B |
| INSIDE（ICLR 2024） | NQ（3,610）、TriviaQA（9,960）闭卷；CoQA、SQuAD 给材料 | 未记录 |
| HaloScope（NeurIPS 2024） | TruthfulQA、TriviaQA 闭卷；CoQA、TydiQA 给材料 | 未记录 |
| LLMs Know More Than They Show（ICLR 2025） | TriviaQA 闭卷；HotpotQA 有无材料两种；NQ 只用给材料的版本 | Mistral-7B、Llama3-8B |
| Semantic Entropy Probes（预印本） | TriviaQA、SQuAD、BioASQ、NQ 闭卷 | Llama-2-7B、Mistral-7B 等 |
| SAR（ACL 2024） | TriviaQA、SciQ 自由回答；CoQA 给材料 | 未记录 |
| LM-Polygraph（TACL 2025） | TriviaQA 闭卷；CoQA 给材料；MMLU 选择题 | Mistral 7B、StableLM 2 12B |
| TSV（ICML 2025） | TruthfulQA、TriviaQA、NQ-open、SciQ，只给问题 | Llama-3.1-8B、Qwen-2.5-7B / 14B |

## 4. 候选题集的基本情况

| 题集 | 有公开答案的题数 | 答案形式 | 已知问题 |
|---|---|---|---|
| TriviaQA（ACL 2017） | 验证集 11,313（另一种配置 17,944，去重后 9,960） | 短答案，附别名表 | 测试集约 30% 的题在训练集中有近似重复（EACL 2021） |
| NQ-open（ACL 2019） | 验证集 3,610 | 短答案，每题可有多个 | 一半以上的题有歧义（EMNLP 2020）；约 16.5% 的答案随时间或地点变化（EMNLP 2021） |
| PopQA（ACL 2023） | 14,267 | 短实体，附别名表 | 由模板生成，16 种关系 |
| SimpleQA（OpenAI 预印本 2024） | 4,326；清理版 1,000 | 唯一的短答案，无别名表 | 作者自估约 3% 的题有错 |
| HotpotQA（EMNLP 2018） | 验证集 7,405 | 短答案，无别名表 | 需要串两条事实；相当一部分题一步就能答出（ACL 2019） |
| SciQ（EMNLP 2017 的研讨会） | 2,000 | 原为四选一，去掉选项后作短答 | 许可为非商用 |

## 5. 8B 至 30B 开源模型闭卷能答对多少

各来源的提示与计分口径不同，只能看量级。

| 题集 | 数字（出处） |
|---|---|
| TriviaQA | Gemma 3 的 27B / 12B / 4B：85.5 / 78.2 / 65.8（官方报告，预训练版，给 5 个示例）；Llama-2-7B / Mistral-7B：64.8 / 52.3（Semantic Entropy Probes 的表 3；同一篇另一处写 26%，口径不明） |
| NQ | Gemma 3 的 27B / 12B / 4B：36.1 / 31.4 / 20.0；Llama-2-7B / Mistral-7B：38.3 / 28.3 |
| PopQA（冷门实体的 1,399 题） | Llama2 7B / 13B：14.7 / 14.7；Alpaca 13B：24.4（Self-RAG） |
| SimpleQA | Gemma 3 的 27B / 12B / 4B：10.0 / 6.3 / 4.0（官方报告，指令版） |

四套题从"多数会"到"几乎都不会"：TriviaQA、NQ、PopQA、SimpleQA。

## 6. 建议与决定

**赵磊 2026-11-19 00 时的决定**：先做 TriviaQA 和 PopQA，"先不用全做，做一部分试一试就知道该不该继续了"。试做的范围与停点见 `Iteration_01/discussion/work_queue/2026-11-19_queue.md` 第 2.3 节。以下是当时给出的建议。

1. **TriviaQA 必做**。它是相近论文里最常用的闭卷自由回答题集（26 篇中 15 篇），答案带别名表，判分最省事。
2. **第二套用 PopQA**：博士 2026-11-15 点名，引用数高；模型多数不会，正好检验"不会就不答"。
3. **NQ-open 排第三**：常用程度仅次于 TriviaQA，与 TriviaQA 一起是 ITI、TruthX、LoFiT 的标准搭配；缺点是一半以上的题有歧义，判分噪声大。
4. **SimpleQA 排第四**：博士点名；模型几乎都不会，放出的回答大多是错的，主要看方法会不会乱放。
5. 每套题整套使用或按固定种子抽样，全部结果照报；先在各套的调参题上做第 0 步（把如实句加进必须作答问法，看净分是否上升），结果不用来删题集。
6. 本次查的方向引导一组 8 篇里，在这些题集上做自由回答并把弃答单列的只有 LITO，没有一篇按净分计。这是评测设置上的一个事实，是否作为差别来写交博士判断。

## 7. 核对情况与未核实之处

- 引用数由 Semantic Scholar 的批量接口一次取得，我读了原始返回。
- 我另外核对了三处：TruthRL 的 arXiv 页面写明 ICML 2026、奖励为对 / 错 / 弃答三档；CRaFT 把拒答作为单独的结果；AbstentionBench 官方仓库列出的 20 个题集中没有 TriviaQA、NQ、SelfAware，有 FalseQA。
- 第 3 节各篇的题集与形式，多数由网页抓取工具读 arXiv 全文得到，中间有一层摘要，没有逐字对照 PDF；Nature 语义熵论文的原句取自全文。
- 未核实：Semantic Entropy Probes 是否被主会收录；BioASQ 的许可；SciQ 上的闭卷正确率；EntityQuestions 的规模。
- 2025 至 2026 年的同类论文没有查全，第 3.1 节只补到 TruthFlow 一篇。

## 8. 来源

- 引用数：<https://api.semanticscholar.org/api-docs/graph>
- 论文（arXiv 编号）：ITI 2306.03341；DoLa 2309.03883；TruthX 2402.17811；LoFiT 2406.01563；Truth Forest 2312.17484；RepE 2310.01405；TruthFlow 2502.04556；LITO 2405.00301；R-Tuning 2311.09677；Alignment for Honesty 2312.07000；[IDK] 词元 2412.06676；Don't Hallucinate, Abstain 2402.00367；Do I Know This Entity? 2411.14257；AbstentionBench 2506.09038；TruthRL 2509.25760；CRaFT 2410.06913；Unfamiliar Finetuning Examples 2403.05612；Semantic Uncertainty 2302.09664；INSIDE 2402.03744；HaloScope 2409.17504；LLMs Know More Than They Show 2410.02707；Semantic Entropy Probes 2406.15927；SAR 2307.01379；LM-Polygraph 2406.15627；TSV 2503.01917；SimpleQA 2411.04368；Gemma 3 报告 2503.19786；Self-RAG 2310.11511
- 语义熵（Nature）：<https://www.nature.com/articles/s41586-024-07421-0>
- 题集质量：训练与测试重复 <https://aclanthology.org/2021.eacl-main.86>；AmbigQA <https://arxiv.org/abs/2004.10645>；HotpotQA 的一步可答 <https://arxiv.org/abs/1906.02900>
- AbstentionBench 的题集清单：<https://github.com/facebookresearch/AbstentionBench>

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-19 | 建立本文件 | 赵磊（Fable 5.2 代笔） |
| 2026-11-19 | 第 6 节记入赵磊的决定：先试做 TriviaQA 与 PopQA | 赵磊（Fable 5.2 代笔） |

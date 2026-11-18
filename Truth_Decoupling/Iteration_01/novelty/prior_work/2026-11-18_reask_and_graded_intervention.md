# 弃答后再问、分级干预与回答转移：七项相近工作的核实

撰写：赵磊（Fable 5.2 代笔）。起始日期：2026-11-18（北京时间）。

七项工作由 GPT 的审阅意见（`Iteration_01/discussion/novelty_review/2026-11-18_gpt_review.md`）列出。Opus 5.5 子代理逐项读原文核实，Fable 5.2 在下载的原文文字中抽查了 13 条摘录，均能找到。本文件只记事实，是否构成创新由博士判断。下文路径均相对于仓库的 `Truth_Decoupling/`。

对照用的四项（本文的做法）：

- (a) 区分两种来历的弃答（模型原本就不答；原本作答、干预后才不答）并分别处理；
- (b) 逐级加大某种干预，以第几档才作答作为把握的读数；
- (c) 检查被收回的回答再次出现时的正误；
- (d) 与"换成必须作答的问法再问"这类提示基线比较。

## 1. 已发表的论文

### 1.1 Refusal Index

- **出处**：Pan 等，*Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks*，ICLR 2026（ICLR 官网海报页与论文首页确认），arXiv 2510.01782。
- **做了什么**：两遍评估。第一遍允许弃答；第二遍去掉弃答选项，只对第一遍拒答的题进行，"We run the second pass only on questions refused in the first pass"（第 3.1 节）。第二遍用来给被拒答的题补上正误，再由两遍的拒答率与错误率拟合一个指数，衡量"越不会的题越拒答"的程度。附录 B 改变谨慎措辞的强度与示例中弃答的比例。
- **任务与模型**：SimpleQA 等 5 个数据集，16 个模型（含 Gemma-3-12B、Qwen3、Llama 3.1 70B）。只用提示，不写入隐状态。
- **四项对照**：(a) 原文未见；(b) 原文未见，四档谨慎提示只用来得到不同的整体拒答率；(c) 原文未见；(d) 有，第二遍即必须作答的提示。

### 1.2 LITO

- **出处**：Bayat 等，*Enhanced Language Model Truthfulness with Learnable Intervention and Uncertainty Expression*，Findings of ACL 2024。
- **做了什么**：沿真实性方向以 k 个强度各生成一个回答，"collect responses A = {a1, …, ak} at each intensity level, and output the most truthful answer, if available, or express uncertainty"（第 2.2 节）；用一层 LSTM 读整条强度序列来挑选；全部被判为不实则输出 "I have no comment"。
- **任务与模型**：NQ、SciQ、TriviaQA、TruthfulQA；Vicuna-7B、Llama2-Chat、GPT2。
- **四项对照**：(a) 原文未见；(b) 原文未见"第几档才作答"的读数，相近的是 LSTM 读强度序列；(c) 原文未见；(d) 原文未见。

### 1.3 SteerConf

- **出处**：Zhou 等，*SteerConf: Steering LLMs for Confidence Elicitation*，NeurIPS 2025 主会。
- **做了什么**：用五档提示引导模型口头报告把握，"{very cautious, cautious, vanilla, confident, very confident}"（第 3.1 节），再把五档的均值、离散度与答案一致性合成校准后的把握。任务里没有弃答选项。
- **任务与模型**：StrategyQA、GSM8K 等；GPT-3.5、LLaMA3-70B、GPT-4。
- **四项对照**：(a) 至 (d) 原文均未见。

### 1.4 Kumaran 等

- **出处**：Kumaran 等，*Causal evidence that language models use confidence to drive behaviour*，Nature Machine Intelligence 8，2026。
- **做了什么**：四选一题另加一个弃答选项。向残差流写入把握方向，"boosting or suppressing confidence correspondingly decreased or increased abstention"；另用提示设定弃答的把握阈值。写入在 Gemma 3 27B 上做。
- **四项对照**：(a) 原文未见；(b) 原文未见逐题"第几档才作答"的读数，只有群体层面的弃答率随强度变化；(c) 原文未见逐题检查；(d) 部分有，第一阶段为不含弃答选项的四选一，但没有与写入比较作答的收益。

### 1.5 Confidence v.s. Critique

- **出处**：Yang 等，*Confidence v.s. Critique: A Decomposition of Self-Correction Capability for LLMs*，ACL 2025 长文。
- **做了什么**：按初答与自我纠正后的正误把回答分为四类：对变对、对变错、错变对、错变错，并定义相应的条件概率。任务里没有弃答，不写入隐状态。
- **四项对照**：(a) 至 (d) 原文均未见。它统计的是初答与修改后回答之间的正误转移。

## 2. 预印本（未经同行评审）

### 2.1 I-CALM

- **出处**：Zong 等，*I-CALM: Incentivizing Confidence-Aware Abstention for LLM Selective Answering*，arXiv 2604.03904。
- **做了什么**：模型先选择作答或弃答，弃答的题"are then forced to provide a best guess"；最佳猜测写在同一次回答里，不另起一遍。提示中用文字说明答对、答错、弃答的得分。只用提示。
- **任务与模型**：PopQA、TriviaQA、SimpleQA-Verified；Llama-3-8B-Instruct、Qwen3-4B 等。
- **四项对照**：(a) 原文未见；(b) 原文未见；(c) 原文未见逐题检查，相近的是用第二部分猜测的正误计算"本会答错的题被弃答拦住的比例"与"本会答对的题照常答出的比例"；(d) 有，基线为不加说明直接作答与明确允许回答不知道。

### 2.2 YOPO

- **出处**：Luo 等，*You Only Pass Once: Answering and Abstaining Together in a Single Forward Pass of a Frozen Language Model*，arXiv 2608.14465。
- **做了什么**：一次前向中，写入一个改善答题的探针，同时在读取层按一个方向打分，低于门槛则弃答；并处理写入对读取的干扰。弃答针对"上下文是否足以回答"，不是模型自身知识的把握。
- **四项对照**：(a) 部分相关：最终弃答是"生成里自己说无法回答"与"门槛判弃答"的或组合，并拆分两者各自的贡献，另写道 "a strong writer stops the model from abstaining on its own"；没有对两类弃答分别处理。(b) 原文未见；(c) 原文未见；(d) 有提示基线，但问的是上下文够不够，不是必须作答。

## 3. 汇总

| 工作 | (a) 区分弃答的来历 | (b) 以第几档才作答为读数 | (c) 被收回的回答重现 | (d) 与必须作答类提示比较 |
|---|---|---|---|---|
| Refusal Index（ICLR 2026） | 未见 | 未见 | 未见 | 有 |
| LITO（ACL Findings 2024） | 未见 | 未见（LSTM 读强度序列） | 未见 | 未见 |
| SteerConf（NeurIPS 2025） | 未见 | 未见 | 未见 | 未见 |
| Kumaran 等（NMI 2026） | 未见 | 未见（群体层面） | 未见 | 部分 |
| Confidence v.s. Critique（ACL 2025） | 未见 | 未见 | 未见（统计的是初答与修改后的转移） | 未见 |
| I-CALM（预印本） | 未见 | 未见 | 未见 | 有 |
| YOPO（预印本） | 部分相关 | 未见 | 未见 | 提示基线不同 |

## 4. 审阅意见中的概括需要修正之处

1. Refusal Index 的第二遍用于给被拒答的题补上正误并拟合指数，原文没有"估计模型收回了什么"的表述。
2. I-CALM 的"强制最佳猜测"在同一次回答之内，不是第二遍提问。
3. YOPO 的弃答对象是上下文是否足以回答。
4. Confidence v.s. Critique 分四类，审阅意见列了三类。

## 5. 建议

- 论文的相关工作一节需要逐一说明与这七项的差别。"弃答后再问"已有先例（Refusal Index、I-CALM），"多档干预后挑选"已有先例（LITO），"写入把握改变弃答"已有先例（Kumaran 等）。
- 按本次核实，七项中没有一项做了 (a)、(b)、(c)。这只说明这七项的情况，不等于没有别的工作做过；进一步的检索范围由博士确定。

## 6. 来源

- <https://iclr.cc/virtual/2026/poster/10011093>，<https://arxiv.org/abs/2510.01782>
- <https://aclanthology.org/2024.findings-acl.737.pdf>
- <https://proceedings.neurips.cc/paper_files/paper/2025/hash/1bfd87d2d92f0556819467dc08034f76-Abstract-Conference.html>
- <https://www.nature.com/articles/s42256-026-01293-x>
- <https://aclanthology.org/2025.acl-long.203.pdf>
- <https://arxiv.org/abs/2604.03904>
- <https://arxiv.org/abs/2608.14465>

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-18 | 建立本文件 | 赵磊（Fable 5.2 代笔） |

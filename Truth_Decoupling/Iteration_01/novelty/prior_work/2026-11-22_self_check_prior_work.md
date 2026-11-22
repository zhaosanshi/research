# "答完自判"一步的已有做法：文献核对

撰写：赵磊（Opus 5.5 代笔）。北京时间 2026-11-22 16:14。本文件回答：已发表的论文里，有没有"模型先答题、再对自己的回答判真假、据此不答或改答"的做法；有没有用内部表示来判；有没有在"判"这一步上做因果干预（往模型内部写方向、改激活）。只记事实，是否构成创新由博士判断。

对照用的步骤（下文称"本步骤"）：模型先正常回答一道问答题；再问它"题目是……，给出的回答是……，这个回答是真的还是假的？"，读它判"真 / 假"的倾向；判为假的回答改成不答（或换方式重答）。自判这一遍里，沿一个用 DAS（Distributed Alignment Search，分布式对齐搜索 · 用梯度学出一个方向，沿它替换激活能改变模型对真假的判断）在真假判断任务上学出的方向，往残差流（residual stream，残差流 · 每一层之间传递的那条主干隐状态）里写入。记分：答对 +1、不答 0、答错 −1。题集：TruthfulQA 这类容易答错的问答题。

术语（第一次出现时的解释）：

- P(True)（自判为真的概率 · 把"题目 + 回答 + 这个回答对吗？(A) True (B) False"喂给模型，读它选 True 的概率）。
- AUROC（ROC 曲线下面积 · 在所有门槛下平均看，把对的和错的分开的能力；0.5 是瞎猜，1 是完全分开）。
- 探针（probe，探针 · 在模型某层隐状态上训练的小分类器，用来读出"对 / 错"之类的信息）。
- 引导向量 / 激活相加（steering vector / activation addition，引导向量 · 推理时往隐状态上加一个固定方向，改变模型行为）。
- 选择性生成 / 选择性预测（selective generation / prediction，选择性作答 · 只输出把握高的回答，其余不答）。

## 1. 总表

"读到什么程度"：全文 = 读了正文相关章节与实验设置；方法与结果 = 读了摘要、方法、主表；摘要 = 只读了摘要（或摘要页）。"核查一步做干预"指在判断回答对错的那一遍计算里改动模型内部激活。

### 1.1 顶会、顶刊正式发表

| 论文 | 场所 | 读到 | 答后用什么信号判对错 | 内部表示 | 核查一步做干预 | 判断之后 | 模型 | 题集（含 TruthfulQA？） | 主要数字 |
|---|---|---|---|---|---|---|---|---|---|
| Park 等，Steer LLM Latents for Hallucination Detection（TSV） | ICML 2025 | 全文 | 对"题目 + 已生成回答"再过一遍，读末层最后一个词元的嵌入，按两类原型打分 | 是 | **是**：在这一遍里往某个中间层残差流加一个学出的向量（λv，所有位置共用），目的是让真 / 假两类分得更开 | 只报检测 | LLaMA-3.1-8B/70B、Qwen-2.5-7B/14B | TruthfulQA、TriviaQA、SciQ、NQ Open（含） | AUROC，LLaMA-3.1-8B 上 TruthfulQA：TSV 84.2，Self-evaluation 67.8，SAPLMA 78.2（表 1） |
| Hedström 等，To Steer or Not to Steer?（MERA） | ICML 2025（PMLR 267） | 方法与结果 | 线性探针读答案词元位置的激活，预测出错概率 | 是 | 否（干预在作答这一遍，按探针预测的出错程度决定写入力度） | 改答（预测出错才写入；"abstention"指不做干预，不是模型不答） | LLaMA-3.x-1B、Gemma-2-2B、Qwen-2.5-3B | SMS Spam、Yes/No、Sentiment、MMLU 子集（不含） | SPI（相对准确率变化），Yes/No 上 +0.42 至 +0.53（表 1） |
| Orgad 等，LLMs Know More Than They Show | ICLR 2025 | 全文 | 探针读作答时"确切答案词元"处的隐状态；对照 P(True)、对数概率 | 是 | 否 | 只报检测；另用探针从 30 个重采样里挑答案（作诊断，不作纠错方法） | Mistral-7B(-Instruct)、Llama3-8B(-Instruct) | TriviaQA、HotpotQA、NQ、Winobias 等 10 个（不含） | AUROC，Mistral-7B-Instruct TriviaQA：探针（确切答案词元）0.85，p(True) 0.66，p(True)-exact 0.74（表 1） |
| Asai 等，Self-RAG | ICLR 2024 | 方法与结果 | 训练模型自己生成反思标记（是否检索、相关、有支撑、有用） | 否（读输出的特殊词元） | 否 | 用反思标记的概率做分段束搜索，挑输出 | Llama2-7B/13B 微调 | PopQA、TriviaQA、PubHealth、ARC、Bio、ASQA（不含） | 13B：PopQA 55.8、TriviaQA 69.3（表 2） |
| Chen 等，INSIDE | ICLR 2024 | 方法与结果 | 多次采样回答的句向量协方差特征值（EigenScore） | 是 | 部分：在采样时截断倒数第二层的极端激活（feature clipping），用来帮检测；不是写方向 | 只报检测 | LLaMA-7B/13B、OPT-6.7B | CoQA、SQuAD、NQ、TriviaQA；附录 A 有 TruthfulQA | 截断最多提高 AUROC 1.8 点；TruthfulQA 检测准确率 EigenScore 76.7–81.3（随提示数），ITI 报告值 83.3（表 4） |
| Burns 等，Discovering Latent Knowledge（CCS） | ICLR 2023 | 摘要 | 无监督找满足"命题与否定命题真值相反"的方向，读是 / 否 | 是 | 否 | 只答是 / 否题 | 6 个模型（T5、UnifiedQA、DeBERTa 等） | 10 个是 / 否类题集（不含） | 比零样本准确率平均高 4% |
| Kumaran 等，Causal evidence that language models use confidence to drive behaviour | Nature Machine Intelligence 2026 | 本次只读了摘要页；细节引用本目录 2026-11-16 的记录 | 内部把握与门槛 | 是 | 否（写入把握方向发生在决定答不答那一步，不是对已给回答的核查） | 不答率随写入方向升降 | Gemma 3 27B | 四选一题加弃答选项（不含） | 弃答率在 7% 到 66.5% 之间变（引自既有记录） |
| Li 等，Inference-Time Intervention（ITI） | NeurIPS 2023 | 方法与结果（另见本目录 ITI 方法记录） | 不做答后自判；探针在 TruthfulQA 问答对上选注意力头 | 是 | 否（写入在作答时） | "I have no comment" 随力度增多 | LLaMA-7B、Alpaca、Vicuna | TruthfulQA（含）、NQ、TriviaQA、MMLU | Alpaca 真实 × 信息量 32.5% → 65.1%（表 2） |
| Bürger 等，Truth is Universal | NeurIPS 2024 | 摘要 | 二维真假子空间上的分类器（TTPD） | 是 | 否 | 只报检测（谎言检测） | Gemma-7B、LLaMA2-13B、Mistral-7B、LLaMA3-8B | 事实陈述集与真实场景谎言（不含） | 准确率 94% |
| Feng 等，Don't Hallucinate, Abstain | ACL 2024 长文 | 全文 | 多种：自反思提示"The above answer is: A. True B. False"、隐层探针（只读题目）、外部校验器、自一致、多模型合作 / 竞争 | 部分（隐层探针基线） | 否 | 判假就不答 | Mistral-7B、LLaMA2-70B、ChatGPT | MMLU、K-Crosswords、Hellaswag、Propaganda（不含） | Mistral-7B MMLU：自反思 可靠准确率 .498、弃答准确率 .495；竞争法 .735、.640（表 1） |
| Zhang 等，Self-Alignment for Factuality | ACL 2024 长文 | 方法与结果 | P(True) 式自评（SELF-EVAL），并用 SK-Tuning 训练加强 | 否 | 否 | 用自评分数造偏好数据做 DPO 训练，不在推理时不答 | LLaMA-7B、LLaMA2-7B | TruthfulQA（含）、BioGEN | TruthfulQA 生成真实 × 信息量 LLaMA2-7B 53.42%（正文第 4 节） |
| Zhang 等，TruthX | ACL 2024 长文 | 摘要（全文检索了 abstain / verify 等词，未见答后自判） | 不做答后自判 | 是 | 否（编辑在作答时） | 改答 | 13 个模型 | TruthfulQA（含） | 平均提高真实性 20%（摘要） |
| Liu 等，On the Universal Truthfulness Hyperplane | EMNLP 2024 主会 | 方法与结果 | 探针读模型对"题目 + 模型输出"的隐状态；对照 Self-Eval（问"Is the answer correct or wrong?"） | 是 | 否（文中写明把干预留作以后） | 只报检测 | LLaMA2-7B-chat 等 | 40 多个题集（含 TruthfulQA） | 只在 TruthfulQA 上训练：探针（LR）同分布 82.28%、跨分布平均 54.44%；Self-Eval 62.96%、63.31%（表 1） |
| Manakul 等，SelfCheckGPT | EMNLP 2023 主会 | 方法与结果 | 多次采样之间是否一致（BERTScore、问答、n 元、NLI、提示） | 否 | 否 | 只报检测 | GPT-3（text-davinci-003） | WikiBio 生成的传记（不含） | 句级 NonFact AUC-PR：提示法 93.42，NLI 92.50，GPT-3 平均对数概率 83.21（表 2） |
| Tian 等，Just Ask for Calibration | EMNLP 2023 主会 | 方法与结果 | 口头报概率（一遍或两遍）、"Is True" 概率（采样估计）、标签概率 | 否 | 否 | 只报把握（校准） | ChatGPT、GPT-4、Claude 等 | TriviaQA、SciQ、TruthfulQA 生成题（含） | ChatGPT TruthfulQA：'Is True' AUC 0.384，两遍口头报（Verb. 2S CoT）AUC 0.551（表 1） |
| Azaria & Mitchell，SAPLMA | Findings of EMNLP 2023 | 方法与结果 | 探针读模型读入或生成陈述时的隐层 | 是 | 否 | 只报检测 | OPT-6.7B、LLaMA2-7B | 自建真假陈述集，另有模型自己生成的陈述（不含） | 准确率 71%–83%（摘要） |
| Bayat 等，LITO | Findings of ACL 2024 | 全文 | 以 5 个力度做 ITI 各生成一答，LSTM 读各答末层隐状态均值，判各答对错 | 是 | 否（干预在作答；判对错的 LSTM 读的是被干预后作答时的隐状态，判这一步本身不写入） | 全判错就输出 "I have no comment"，否则输出判对且把握最高的那个 | GPT2-large/XL、Llama2-Chat-7B/13B、Vicuna-7B | NQ、SciQ、TriviaQA、TruthfulQA（含） | TA 分（真实性与准确率的几何平均），Llama2-Chat-7B TruthfulQA：原模型 45.1、ITI 最好一档 49.9、LITO 49.6（表 1） |
| Dhuliawala 等，Chain-of-Verification（CoVe） | Findings of ACL 2024 | 方法与结果 | 模型自己提核查问题、分开回答，再改写 | 否 | 否 | 改答 | Llama 65B | Wikidata 列表题、MultiSpanQA、传记（不含） | Wikidata 精度 0.17 → 0.36；FactScore 55.9 → 71.4（第 5 节） |
| Chen 等，Truth Forest | AAAI 2024 | 摘要 | 不做答后自判 | 是 | 否（写入在作答） | 改答 | Llama-2-7B 等 | TruthfulQA（含） | 真实性 40.8% → 74.5%（摘要） |
| Marks & Tegmark，The Geometry of Truth | COLM 2024 | 方法与结果 | 少样本真假判断提示"…This statement is:"，读 P(TRUE) − P(FALSE) | 是 | **是**：在判断这一遍里，把真假方向加到 / 减到陈述末尾词元的隐状态上 | 无（用来证明方向有因果作用：让模型把假判成真或反之） | LLaMA-2-13B/70B | 自建真假陈述集（不含） | 归一化间接效应（NIE，1 表示完全翻转），均值差方向在 LLaMA-2-70B 上多为 0.9–1.0（表 2） |

### 1.2 研讨会论文

| 论文 | 场所 | 读到 | 答后用什么信号判对错 | 内部表示 | 核查一步做干预 | 判断之后 | 模型 | 题集 | 主要数字 |
|---|---|---|---|---|---|---|---|---|---|
| Ren 等，Self-Evaluation Improves Selective Generation | NeurIPS 2023 研讨会（ICBINB，PMLR 239） | 全文 | 多选（Sample and Select）、逐个判对错 p(Yes)（Sample and Eval）、两者结合并加"以上都不是"选项 | 否 | 否 | 按分数从低到高不答（选择性生成） | PaLM-2 Large、GPT-3 | TruthfulQA（含）、TL;DR | PaLM-2 TruthfulQA：Sample and Eval 校准 AUC 73.79、选择性 AUC 58.19；序列似然 39.80、33.63（表 1） |
| Kossen 等，Semantic Entropy Probes | ICML 2024 研讨会（海报） | 摘要 | 探针从单次生成的隐状态预测语义熵 | 是 | 否 | 只报检测 | Llama-2-7B/70B、Mistral-7B、Phi-3 Mini | TriviaQA、SQuAD、BioASQ、NQ Open（不含） | 没有逐项核对数字 |
| Hoscilowicz 等，NL-ITI | arXiv 注释称被 Interspeech 2024 接收（没有在会议论文集核到） | 摘要 | 不做答后自判 | 是 | 否 | 改答 | — | TruthfulQA 等选择题（含） | MC1 比 ITI 相对提高 16% 以上（摘要） |

### 1.3 预印本（未经同行评审）

| 论文 | 读到 | 答后用什么信号判对错 | 内部表示 | 核查一步做干预 | 判断之后 | 模型 | 题集 | 主要数字 |
|---|---|---|---|---|---|---|---|---|
| Kadavath 等，Language Models (Mostly) Know What They Know（arXiv 2207.05221） | 全文 | P(True)；另把 5 个采样答案一起给模型看再判（"brainstormed ideas"） | 否（P(IK) 用另加的值头训练） | 否 | 只报把握；报告 P(True) > 0.5 子集上的准确率 | 800M–52B 自家模型 | TriviaQA、Lambada、GSM8k、HumanEval、算术；TruthfulQA 只用于选择题校准 | 数字只在图中（图 1、11、28–31），正文没有逐项表 |
| Zhou 等，The Hidden Signal of Verifier Strictness（arXiv 2605.20745） | 方法与结果 | 校验模型逐步检查别的模型写的数学解答 | 是 | **是**：在校验这一遍里，于每段校验前的分隔词元上加均值差方向（正确拒绝 − 错误接受） | 只报找错步的 F1，不涉及不答 | Qwen3-1.7B/8B、FARE-20B | ProcessBench、Hard2Verify（不含） | Qwen3-8B ProcessBench F1 65.9 → 72.3；FARE-20B Hard2Verify 41.5 → 51.1 |
| Kumaran 等，How LLMs Detect and Correct Their Own Errors（arXiv 2604.22271） | 方法与结果 | 先答，再问"Verify your answer. Correct? Y/N"，读口头判断、回答后换行词元处的激活 | 是 | **是，但只作因果证据**：在核查提示里做激活修补与均值消融，看判错能力是否恢复 / 被破坏 | 核查后让模型重答（不设不答） | Gemma 3 27B、Qwen 2.5 7B | TriviaQA、MNLI（不含） | 模型认可 98% 的对答、也认可 68% 的错答；错答内部探针 AUROC 0.958，对照 0.715；重答准确率 75.5% → 79.2% |
| Luo 等，Shaking to Reveal（SSP，arXiv 2506.02696） | 方法与结果 | 在 P(True) 式自评输入后接一段学出的"扰动提示"嵌入，读中间层表示在扰动前后的变化 | 是 | 部分：改的是输入嵌入（软提示），不是往残差流写方向 | 只报检测 | LLaMA-3-8B-Instruct、Qwen2.5-7B-Instruct | TruthfulQA（含）、TriviaQA、CoQA、TydiQA | TruthfulQA AUROC：SSP 73.43，Self-evaluation 53.18（LLaMA-3-8B，表 1） |
| Eshuijs 等，JUSSA（arXiv 2505.17760） | 摘要 | 外部裁判比较原回答与"被诚实向量引导后重新生成的回答" | 是（引导被评模型） | 否（写入的是被评模型，不是裁判的核查） | 只报检测 | 裁判 GPT-4.1、Claude Haiku | 自建操纵数据集（不含） | AUROC 0.893 → 0.946（GPT-4.1 裁判） |
| Zou 等，Representation Engineering（RepE，arXiv 2310.01405） | 方法与结果（第 4 节） | 读"诚实"方向的投影做谎言检测；选择题用读出的真实性给选项打分（LAT） | 是 | 否（加诚实向量用于控制输出，不是用于核查已有回答） | 改答（控制诚实）；检测只做示例 | LLaMA-2-Chat 7B/13B/70B | TruthfulQA MC1（含） | 读出（LAT）MC1 平均 60.7 对零样本 32.3（表 1）；对比向量控制 13B 54.0 对 35.9（表 2） |
| Yuan 等，Hidden Error Awareness in Chain-of-Thought（arXiv 2605.09502，ICML 2026 研讨会稿） | 摘要 | 探针读推理过程隐状态 | 是 | 试过引导、修补，均未能纠错 | 只报检测 | Qwen、Llama、Phi（1.5B–72B）、DeepSeek-R1 | 没查到题集名 | 探针 AUROC 0.95；口头把握 4.55 对 4.87（错 / 对） |
| Roy 等，Detection Without Correction（arXiv 2604.13068） | 摘要 | 探针读激活 | 是 | 否（引导用于纠错，在作答时） | 纠错失败 | GPT-2、Pythia、Qwen-2.5（117M–7B） | 没查到 | 7 个模型上引导都没有纠正幻觉 |
| Zhang 等，Reasoning Models Know When They're Right（arXiv 2504.05419） | 摘要 | 探针读推理模型隐状态判中间答案对错 | 是 | 否 | 提前结束推理 | 没查到 | 没查到 | 推理词元减少 24% |
| Wagner，Two Axes of LLM Abstention（arXiv 2607.08456） | 摘要 | 答案把握、P(IK)、P(True)、隐状态探针 | 是 | 否 | 两个门槛共同决定答不答 | 2B–14B 共 5 个 | CREPE（不含） | 假前提题上探针 AUROC 0.69–0.77，其他信号接近瞎猜 |
| Lu，Diagnosing Correctness Probes under Self-Judgement Confounding（arXiv 2607.16799） | 摘要 | 区分"客观对错"方向与"模型自认为对"方向 | 是 | 摘要未见 | 只做诊断 | 4 个指令模型（≤14B） | 数学、事实回忆；扩展到二元 TruthfulQA 与 MMLU | 摘要只给定性结论 |
| Jeong 等，Internal Flow Signatures（arXiv 2602.01897） | 摘要 | 读层间隐状态轨迹特征，GRU 判对错 | 是 | 检测不改模型；之后的修正步骤回退并钳住异常层的变化 | 改答 | 没查到 | 没查到 | 没查到 |
| Yang 等，CASAL（arXiv 2510.02324；检索结果显示 ICLR 2026 海报，未逐项核对） | 摘要 | 不做答后自判 | 是 | 否（把引导方向训练进一层的权重） | 不会的题不答 | 没查到 | 没查到 | 短问答幻觉降 30%–40% |
| Hegde，Steering Self-Evaluation（谷歌研究页 2025，没查到发表场所） | 摘要 | 找到"自评推理步骤"的方向 | 是 | 是（引导自评行为），没查到对准确率的影响 | 没查到 | 没查到 | 数学、医学 | 没查到 |

## 2. 第 1 问：答完之后让模型自判对错，再不答或改答

**Kadavath 等（预印本，全文）**。第 4.1 节：先采样一个回答，再用"Question… Proposed Answer… Is the proposed answer: (A) True (B) False"读 P(True)。第 4.2 节：把 5 个采样答案作为"brainstormed ideas"一起给模型看，再判其中一个，短答案题上明显变好。零样本 P(True) 校准差、常在 50% 附近，少样本后变好。图 11 下：报告"只保留 P(True) > 0.5 的回答"时的准确率高于整体准确率。第 3.2 节用同一格式判别人给的选项，第 5 节另训 P(IK)（模型能否答对这题的概率，用值头训练）。TruthfulQA 只在第 2–3 节的选择题校准里出现，P(True) 自判没有在 TruthfulQA 上做。

**Ren 等（NeurIPS 2023 研讨会，全文）**。第 2 节：每题采样 4 个答案。"Sample and Select"把它们变成多选题让模型挑；"Sample and Eval"逐个问对不对，读 p(Yes)，作者说这与 P(True) 相近；"Hybrid"先挑再判，并可加"以上都不是"。第 3 节：按分数排序，把最低的若干比例不答，用选择性 AUC 衡量。表 1（TruthfulQA，GPT-judge 判对错）：PaLM-2 上序列似然的校准 AUC 只有 39.80，低于 0.5；Sample and Eval 73.79，Hybrid 加"以上都不是"75.34。

**Feng 等（ACL 2024，全文）**。第 2.3 节"Self-Reflect"：答完后直接接"The above answer is: A. True B. False"，判为 False 就不答，这一做法引自 Kadavath 等。同文的基线还有：隐层探针（第 2.2 节，只读题目的隐状态）、外部校验器、温度校准、自一致门槛、"以上都不是"。他们提出的"合作"（其他模型给反馈，再由裁判决定）和"竞争"（给出冲突信息后看模型改不改答案）在 12 个设定中的 9 个可靠准确率最好（表 1）。第 3 节的指标里，有效可靠度 ER = (答对 − 答错) / 总题数，与"答对 +1、不答 0、答错 −1"的平均分是同一个量。四个题集不含 TruthfulQA。

**Tian 等（EMNLP 2023，方法与结果）**。比较让 RLHF 模型给把握的多种方法：标签概率、"Is True"概率（第 3 节：先采样一个答案，再问它对不对，因为闭源模型没有概率，用 10 次采样估计）、一遍口头报概率（1S）、两遍口头报（2S：先只给答案，第二遍再问把握）。表 1（ChatGPT）：TruthfulQA 上"Is True"的 AUC 0.384，低于 0.5；口头报的校准误差更小。只报把握，不做不答。

**Manakul 等 SelfCheckGPT（EMNLP 2023，方法与结果）**。对同一提示多次采样，看主回答的每句话在其他采样里是否被支持；有 BERTScore、问答、n 元、NLI、提示五种比法。表 2：句级"非事实"检测 AUC-PR 提示法 93.42。WikiBio 传记，GPT-3，不涉及不答，也不含 TruthfulQA。

**Dhuliawala 等 CoVe（Findings of ACL 2024，方法与结果）**。四步：起草回答；针对草稿提核查问题；分开回答这些问题（"factored"版本让每个问题单独回答，避免受草稿影响）；据核查结果改写。第 5 节：Wikidata 列表题精度 0.17 → 0.36，错误实体数 2.95 → 0.68，正确实体数 0.59 → 0.38。Llama 65B，只改答，不设不答。

**Asai 等 Self-RAG（ICLR 2024，方法与结果）**。表 1：四类反思标记（是否检索、检索段落是否相关、回答是否被支持、是否有用），用 GPT-4 标注后蒸馏给一个评判模型，再插入训练数据，让生成模型自己输出这些标记。推理时按标记概率的加权和做分段束搜索。读的是输出的特殊词元，不读内部表示，不设不答。

**Zhang 等 Self-Alignment for Factuality（ACL 2024，方法与结果）**。用 P(True) 式的自评（SELF-EVAL）给模型自己的多条回答打分，造偏好对做 DPO 训练；另用 SK-Tuning 加强自评。推理时不判不答。TruthfulQA 生成任务真实 × 信息量 LLaMA2-7B 53.42%。

## 3. 第 2 问：用内部表示判断模型自己回答的对错

**Orgad 等（ICLR 2025，全文）**。第 3 节：对比对数概率、P(True)（用 Kadavath 等的同一提示，附录 A）和探针。探针读的是作答这一遍里"确切答案词元"处的中间层隐状态，这个位置比最后一个词元好。表 1：Mistral-7B-Instruct TriviaQA 上探针 0.85，p(True) 0.66。第 6 节：用探针从 30 个重采样里挑分数最高的回答，准确率在"多数采样是错的"等类型上高出 30–40 点；作者写明这是诊断工具，不作为纠错方法提出。全文没有写入或引导，没有不答，题集不含 TruthfulQA。

**Azaria & Mitchell SAPLMA（Findings of EMNLP 2023，方法与结果）**。在模型读入或生成一句陈述时取隐层，训练前馈分类器判真假；另建了一份模型自己生成的陈述集（第 4 节），表 3 报告在这份集上的准确率。OPT-6.7B、LLaMA2-7B，准确率 71%–83%。

**Burns 等 CCS（ICLR 2023，摘要）**。无监督地找一个方向，使"命题"和"它的否定"在这个方向上的概率相加为 1。10 个是 / 否题集，比零样本平均高 4%。判的是题目给定的命题，不是模型自己的回答。

**Marks & Tegmark（COLM 2024，方法与结果）**。自建真假陈述集，发现大模型里真假是线性可分的；第 5 节说均值差探针（mass-mean，两类均值相减得到的方向）泛化与逻辑回归相当。第 6 节因果干预：在少样本真假判断提示里，把方向加到陈述末尾词元和句号处的隐状态上，看 P(TRUE) − P(FALSE) 的变化，用 NIE 衡量；表 2 均值差方向的翻转效果在 8 个条件中的 7 个最好。判的是给定陈述，不是模型自己的回答，不涉及不答。

**Chen 等 INSIDE（ICLR 2024，方法与结果）**。第 3.1 节：取多次采样回答的句向量，计算协方差矩阵的特征值（EigenScore）衡量一致性。第 3.2 节：采样时把倒数第二层超出 0.2% 分位的激活截断，"防止过度自信的生成"，表 2 说最多提高 AUROC 1.8 点。附录 A 表 4 在 TruthfulQA 上报告检测准确率。

**Kossen 等 SEP（ICML 2024 研讨会，摘要）**。探针从单次生成的隐状态预测语义熵，不必多次采样；对照包括直接预测对错的探针和 P(True)。

**Bürger 等（NeurIPS 2024，摘要）**。真假陈述的激活在一个二维子空间里可分（一维管真假，一维管肯定 / 否定的极性）；据此做的检测器在事实陈述和真实场景谎言上准确率 94%。

**Liu 等（EMNLP 2024 主会，方法与结果）**。在 40 多个题集上训练"真实性超平面"探针，读模型对自己输出的隐状态。Self-Eval 基线问"Is the answer correct or wrong?"并只允许解码 correct / wrong（第 3.1 节）。表 1：只在 TruthfulQA 上训练时，探针同分布 82.28%、跨分布 54.44%，Self-Eval 跨分布 63.31%；作者据此说专门针对单一题集的探针泛化差。结论一节写明干预留作以后。

**Park 等 TSV（ICML 2025，全文）**。见第 4 节。

**Kumaran 等 2604.22271（预印本，方法与结果）**。见第 4 节。

## 4. 第 3 问：在"自判 / 核查"这一步做因果干预，或把干预与答后核查结合

### 4.1 在核查这一遍里改动内部激活的

**Park 等 TSV（ICML 2025，全文）**。最接近的一项。生成完成后，把"提示 + 已生成回答"再过一遍模型，在一个中间层的残差流上加一个学出的向量 v（h ← h + λv，所有词元位置共用，第 4.1 节式 2），读末层最后一个词元的单位化嵌入，按"真 / 幻觉"两类原型打分（式 13）。v 的训练目标是让两类在嵌入空间里聚得紧、分得开：先用少量带标签样本（TruthfulQA 32 条），再用最优传输给无标签生成打伪标签、挑最有把握的 128 条加入（第 4.2 节）。这一遍里没有"这个回答对吗？"之类的提问，读的不是 P(True)。只报检测 AUROC，不做不答或改答（表 1）；第 4.3 节说去掉向量即可回到正常生成。图 3：λ = 5 左右最好，再大反而下降。

**Zhou 等 VerifySteer（预印本，方法与结果）**。对象是逐步检查数学解答的校验模型，校验的是别的模型写的解答（"the generator and verifier are distinct models"）。方向 = 正确拒绝的隐状态均值 − 错误接受的隐状态均值（第 4.2 节式 1–2），取每段校验前的"\n\n"词元；在校验这一遍里写入并保持范数不变（式 3）。"选择性"指先用探针估计这份解答可能错不错，可能错的加"严格"方向，可能对的加"宽松"方向（第 5.1 节）。表中 F1 高于提示优化和统一写入，与多次投票相当而计算量少 4–7 倍。不涉及问答题，不设不答。

**Kumaran 等 2604.22271（预印本，方法与结果）**。三步：先答并给口头把握；再问"Verify your answer. Correct? (Output ONLY a single letter, Y/N)"；再问"你认为正确答案是什么"重答（附录 A.2）。在核查提示里做了两种因果操作（第 4 节、附录 A.6–A.7）：先把答案词元的嵌入做均值消融，使判错能力 d′ 从 1.17 降到 0.09，再在某一位置某一层恢复原激活，看能恢复多少（最后一个答案词元在第 15 层约 100%，回答后换行词元在第 30 层约 74%）；以及单独消融某些位置。这些操作只用来说明信号在哪里起作用，没有用来提高核查准确率；文中只提出"监视回答后换行词元处的激活可能帮助选择性纠错"，没有做。

**Marks & Tegmark（COLM 2024）**。在真假判断这一遍里沿真假方向加减，能翻转模型对给定陈述的判断（第 6 节）。目的在于证明方向有因果作用，不是让判断更准；判的不是模型自己的回答。

**Luo 等 SSP（预印本，方法与结果）**。在自评输入"题目 + 回答 + 模板"里加一段扰动提示：先让模型生成一句只改语气的句子，再用一个两层 MLP 按样本改它的嵌入（第 4.2 节式 10–13），读中间层表示在扰动前后的余弦变化。改动发生在输入嵌入，不是往残差流写一个方向。只报检测：TruthfulQA AUROC 73.43，对照 Self-evaluation 53.18（表 1）。

**Chen 等 INSIDE（ICLR 2024）**。在为检测而做的多次采样中截断极端激活（第 3.2 节），是为检测服务的激活改动，但不是写方向，也不是对单个回答提问核查。

### 4.2 把干预与"答后判对错 / 不答"放在同一流程里，但干预不在核查一步

**Bayat 等 LITO（Findings of ACL 2024，全文）**。第 3 节：用 ITI 的方向，以 α = 5, 10, …, 25 五个力度各生成一个回答；把每个回答所有生成词元的末层隐状态取均值，按力度顺序送入一层 LSTM，输出每个回答是否正确（式 3–4）；若全判错就输出"I have no comment"，否则输出判对的回答里把握（词元概率几何平均）最高的那个（第 3.2 节式 5）。判对错用的是作答时（已被干预）的隐状态，没有单独的自判提问，判这一步不写入。训练标签由 DeBERTa NLI 模型或 GPT-4 给出（第 4.2 节）。表 1 TruthfulQA 上 Llama2-Chat-7B：ITI 最好一档 49.9，LITO 49.6；"最大把握 > 0.6"基线 49.4。

**Hedström 等 MERA（ICML 2025，方法与结果）**。探针读作答时答案词元的激活，预测出错程度；只在预测出错超过门槛时，沿探针方向按闭式解给出的力度写入（第 3 节式 7），否则不写。文中的"abstain"指不做干预（第 5 节），模型仍然作答。是二分类、选择题类任务，不是问答题，不含 TruthfulQA。

**Kumaran 等（Nature Machine Intelligence 2026，摘要 + 本目录既有记录）**。写入把握方向改变弃答率；写入发生在答题决定时，不是对已给出的回答再核查。

**Li 等 ITI 及后续（NL-ITI、Truth Forest、TruthX）**。干预都在作答时，没有答后自判一步。ITI 第 4 节说真实性随力度上升时"I have no comment"增多（详见本目录 ITI 方法记录）。

**Zou 等 RepE（预印本，第 4 节）**。用诚实方向的投影逐词元做谎言检测（第 4.3.2 节，只给示例）；把诚实向量加到激活上控制输出（第 4.3.3 节，表 2 在 TruthfulQA MC1 上）。检测与控制是两个分开的演示，没有"先答、再用加了方向的一遍去核查"的流程。

**Eshuijs 等 JUSSA（预印本，摘要）**。写入的是被评的模型（让它生成一个"更诚实"的对照回答），裁判不被写入。

**Yuan 等（预印本）、Roy 等（预印本）**。探针能检测出错，但沿探针方向引导都没能纠错（两篇的摘要）；引导用在作答上，不在核查一步。

**Yang 等 CASAL（摘要）**。把"不会就不答"的引导训练进一层权重，没有答后核查。

### 4.3 检索记录

检索日期：2026-11-22（北京时间）。用到的通用关键词：activation steering self-verification hallucination abstain；self-evaluation steering vector truthfulness P(True) intervention；steering LLM self-correction verification step representation engineering；truth direction intervention P(True) calibration；post-hoc verification internal states abstention LLM answer correctness probe；Mechanistic Error Reduction with Abstention；self-verification activation steering verifier hidden states；steering vector LLM judge self-critique truthfulness；representation editing self-reflection hallucination detection abstain TruthfulQA；steering vector abstention "I don't know" knowledge-aware refusal；intervene hidden states verification prompt "is the answer correct"；activation steering self-evaluation correctness probe selective prediction calibration；steer LLM latents hallucination detection；activation steering during self-evaluation prompt AUROC P(True)；steering vector LLM-as-a-judge factual verification；generate then verify activation intervention selective answering；steering "I don't know" unknown questions knowledge boundary。

在检索到并读过的这些论文里，没有读到这样一项做法：模型先作答，再用"这个回答是真的还是假的？"的提问让它自判，并在这一遍自判里沿真假方向往残差流写入来提高自判的准确率，再据此不答或改答。最接近的是 TSV（在对"题目 + 回答"的检测这一遍里加学出的向量提高检测 AUROC，但没有自判提问、不读 P(True)、不做不答）、VerifySteer（在校验模型的校验这一遍写入均值差方向，提高找错 F1，但对象是数学解答、校验者不是作答者、不做不答）和 Kumaran 等 2604.22271（在自判提示里做激活修补，只作因果证据）。这只说明检索到的范围，不等于不存在。

## 5. 与本步骤逐项相比

| 本步骤的要素 | 读到有相同或相近做法的 | 差别（事实） |
|---|---|---|
| 先答，再把"题目 + 回答"交给同一模型问真假 | Kadavath 等、Ren 等、Feng 等、Tian 等、Orgad 等的 P(True) 基线、Liu 等的 Self-Eval 基线、Kumaran 等 2604.22271、Zhang 等 Self-Alignment | 都不写入方向 |
| 读"真 / 假"的概率倾向 | Kadavath 等（P(True)）、Ren 等（p(Yes)）、Tian 等（'Is True'）、Marks & Tegmark（P(TRUE) − P(FALSE)） | Tian 等用采样估计概率；Marks & Tegmark 判的是给定陈述 |
| 判为假就不答 | Feng 等（Self-Reflect）、Ren 等（按分数不答）、LITO（全判错才不答）、Wagner（两个门槛） | LITO 判对错用的是作答时的隐状态和 LSTM，不是自判提问 |
| 判为假就换方式重答 | Kumaran 等 2604.22271（核查后重答）、CoVe（改写）、Orgad 等（用探针从重采样里挑） | 这几项的核查或挑选都不写入 |
| 在判断那一遍写入方向 | TSV（加学出的向量，为了分得更开）、VerifySteer（加均值差方向，为了调严格度）、Marks & Tegmark（加真假方向，为了证明因果）、Kumaran 等 2604.22271（修补 / 消融，为了证明因果）、SSP（改输入嵌入）、INSIDE（截断激活） | TSV 没有自判提问、读嵌入原型；VerifySteer 是数学逐步校验、校验者与作答者是不同模型；Marks & Tegmark 与 Kumaran 等不以提高判断准确率为目的；SSP、INSIDE 不是写方向 |
| 方向用 DAS（因果干预式的梯度搜索）在真假判断任务上学 | 读到的写入方向来源：TSV 用带标签样本 + 伪标签训练的分离目标；VerifySteer 与 Marks & Tegmark 用均值差；ITI 系列用探针权重或均值差；RepE 用主成分；MERA 用探针权重 | 这些论文里没有读到用 DAS 学方向 |
| 与不写入的原模型、均值差方向比较判断的 AUROC | TSV 与不写入时比（第 5.3 节消融、表 3，LLaMA-3.1-8B：不加 TSV 只训类原型时接近 50%，加 TSV 后 TruthfulQA AUROC 提高 28.7 个百分点）；Marks & Tegmark 比较了均值差、逻辑回归、CCS 三种方向的因果效果 | 读到的工作没有在"自判提问 + 写入"的设置下做这组比较 |
| 记分答对 +1、不答 0、答错 −1 | Feng 等的有效可靠度 ER = (答对 − 答错) / 总题数（第 3 节，引自 Whitehead 等 2022） | 其他论文多用 AUROC、选择性 AUC、TA 分、可靠准确率 |
| 题集为 TruthfulQA | Ren 等、Tian 等、TSV、SSP、LITO、INSIDE（附录）、Liu 等、ITI 系列、RepE、Self-Alignment | Kadavath 等、Feng 等、Orgad 等、Kumaran 等 2604.22271、VerifySteer、MERA 都没有在 TruthfulQA 上做答后自判 |

## 6. 可以直接拿来比的现成基线（事实清单）

1. **P(True) 自判，不写入**（Kadavath 等第 4.1 节）。提示："Question: … Proposed Answer: … Is the proposed answer: (A) True (B) False The proposed answer is:"，读" (A)"与" (B)"的概率。Orgad 等附录 A 用的是同一模板，并加了"只看确切答案"的变体 p(True)-exact。原文提醒零样本时 P(True) 校准差、常在 50% 附近，少样本后改善。
2. **P(True) 加多个采样作对照**（Kadavath 等第 4.2 节）。先采 5 个答案，以"Here are some brainstormed ideas: …"列出，再问"Possible Answer: … Is the possible answer: (A) True (B) False"。
3. **逐个判对错 p(Yes) 与"先挑再判 + 以上都不是"**（Ren 等第 2 节、表 1）。按分数从低到高不答，报告校准 AUC 与选择性 AUC；在 TruthfulQA 上用 GPT-judge 判对错。
4. **自反思后判假就不答**（Feng 等第 2.3 节）。答完直接接"The above answer is: A. True B. False"，选 B 就不答；同文用 ER 衡量，等同于 +1 / 0 / −1 的平均分。
5. **两遍口头报把握**（Tian 等，Verb. 2S）。第一遍只答，第二遍问"这个答案正确的概率是多少"；ChatGPT 在 TruthfulQA 上 'Is True' 的 AUC 0.384，Verb. 2S CoT 0.551（表 1）。
6. **按回答的对数概率定把握**。序列似然、按长度归一的序列似然（Ren 等表 1：PaLM-2 TruthfulQA 校准 AUC 39.80、50.09）；词元对数概率的均值 / 最小值及只看确切答案的版本（Orgad 等表 1）；词元概率几何平均加门槛 0.6（LITO 的"最大把握 > T"基线，第 5.4 节）。
7. **作答时隐状态上的探针**（不写入）。确切答案词元处的线性探针（Orgad 等）；读陈述隐层的前馈分类器（SAPLMA）；跨 40 多个题集训练的超平面（Liu 等）。
8. **在检测那一遍加学出的向量**（TSV，第 4 节）。中间层残差流加 λv，读末层末词元嵌入，按原型打分；TruthfulQA 用 32 条带标签样本。
9. **作答时多力度干预再挑选**（LITO，第 3 节），以及作答时按探针门槛写入（MERA，第 3 节）。

## 7. 没有核到 / 不确定

- Kadavath 等是预印本，P(True) 的逐题集 AUROC 只在图里，正文没有表，本文件没有抄数字。
- Kumaran 等 Nature Machine Intelligence 2026：这次出版社页面需要登录，只读了 arXiv 2603.22161 的摘要页，细节（Gemma 3 27B、弃答率 7%–66.5%）引自本目录 2026-11-16 的记录。
- TSV 的 28.7 来自表 3 消融（不加 TSV 一行对加 TSV 一行），表 3 的逐格数字没有抄录；它是"提高 28.7%"还是"28.7 个百分点"，原文写作"improves AUROC by 28.7%"，本文件按百分点理解，没有逐格核对。
- SelfCheckGPT、Self-RAG、SAPLMA、INSIDE、Liu 等、RepE、Self-Alignment、VerifySteer、Kumaran 等 2604.22271、SSP 读了方法与主要结果，没有逐节读全文。
- CCS、SEP、Bürger 等、NL-ITI、Truth Forest、TruthX、JUSSA、CASAL、Yuan 等、Roy 等、Zhang 等 2504.05419、Wagner、Lu、Jeong 等、Hegde 只读了摘要。其中 SEP 的数字、Hegde 的模型与结果、Jeong 等与 Roy 等的题集都没查到。
- NL-ITI 的发表场所只见 arXiv 注释与二手检索结果，没在会议论文集核到。CASAL 是否为 ICLR 2026 只见检索结果中的会议海报页，没打开核对。SEP 的研讨会身份来自检索结果。
- Hegde 的 Steering Self-Evaluation 是谷歌研究页上的条目，没找到原文 PDF。
- Lu 2607.16799 是否做了写入或引导，摘要未提，没查到。
- MERA 的 arXiv 版本里，第 4.1 节与表 1 对 LLaMA 版本号的写法不一致（3.2 与 3.1），照原文记。

## 8. 来源

- Kadavath 等：<https://arxiv.org/abs/2207.05221>
- Ren 等：<https://proceedings.mlr.press/v239/ren23a/ren23a.pdf>
- Manakul 等：<https://aclanthology.org/2023.emnlp-main.557.pdf>
- Dhuliawala 等：<https://aclanthology.org/2024.findings-acl.212.pdf>
- Feng 等：<https://aclanthology.org/2024.acl-long.786.pdf>
- Asai 等：<https://arxiv.org/abs/2310.11511>，<https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html>
- Tian 等：<https://aclanthology.org/2023.emnlp-main.330.pdf>
- Zhang 等 Self-Alignment：<https://arxiv.org/abs/2402.09267>
- Azaria & Mitchell：<https://aclanthology.org/2023.findings-emnlp.68.pdf>
- Burns 等：<https://arxiv.org/abs/2212.03827>，<https://iclr.cc/virtual/2023/poster/11480>
- Marks & Tegmark：<https://arxiv.org/abs/2310.06824>
- Orgad 等：<https://arxiv.org/abs/2410.02707>
- Chen 等 INSIDE：<https://arxiv.org/abs/2402.03744>
- Kossen 等：<https://arxiv.org/abs/2406.15927>
- Bürger 等：<https://arxiv.org/abs/2407.12831>
- Liu 等：<https://aclanthology.org/2024.emnlp-main.1012/>
- Li 等 ITI：<https://arxiv.org/abs/2306.03341>
- NL-ITI：<https://arxiv.org/abs/2403.18680>
- Truth Forest：<https://arxiv.org/abs/2312.17484>
- TruthX：<https://arxiv.org/abs/2402.17811>
- LITO：<https://aclanthology.org/2024.findings-acl.737.pdf>
- RepE：<https://arxiv.org/abs/2310.01405>
- TSV：<https://arxiv.org/abs/2503.01917>，<https://icml.cc/virtual/2025/poster/45122>
- MERA：<https://arxiv.org/abs/2510.13290>，<https://proceedings.mlr.press/v267/hedstrom25a.html>
- VerifySteer：<https://arxiv.org/abs/2605.20745>
- Kumaran 等 2604.22271：<https://arxiv.org/abs/2604.22271>
- Kumaran 等 NMI：<https://www.nature.com/articles/s42256-026-01293-x>，<https://arxiv.org/abs/2603.22161>
- SSP：<https://arxiv.org/abs/2506.02696>
- JUSSA：<https://arxiv.org/abs/2505.17760>
- Yuan 等：<https://arxiv.org/abs/2605.09502>
- Roy 等：<https://arxiv.org/abs/2604.13068>
- Zhang 等 2504.05419：<https://arxiv.org/abs/2504.05419>
- Wagner：<https://arxiv.org/abs/2607.08456>
- Lu：<https://arxiv.org/abs/2607.16799>
- Jeong 等：<https://arxiv.org/abs/2602.01897>
- CASAL：<https://arxiv.org/abs/2510.02324>
- Hegde：<https://research.google/pubs/steering-self-evaluation-interpreting-llms-reasoning-across-domains-and-languages/>

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-22 16:14 | 建立本文件 | 赵磊（Opus 5.5 代笔） |

# 文献侦察候选清单 · Evidence vs Commitment 方案

- 生成：2026-10-24（PDT）/ 北京时间 2026-10-25
- 范围：2025-01 ～ 2026-10 的顶会论文 + 有分量的 arXiv 预印本；已排除方案里"已在读"的 45 篇
- 方法：arXiv API 约 110 组关键词查询（覆盖 truthfulness steering / 两方向 / 弃答方向 / DAS / 读层-写层 / 已知-未知门控 / SDT-Youden / honesty-lie detection / commitment 等）+ WebSearch 约 40 轮（含 OpenReview、ACL Anthology、NeurIPS/ICML/ICLR 会议页、知乎/机器之心）
- 所有 arXiv ID 均经 arXiv API `id_list` 回查标题/摘要核对；会议信息取自 arXiv comment 或会议官网，取不到的标"仅 arXiv"；作者机构取不到的标"待核"
- 术语：steering（激活引导 · 在残差流上加一个向量改模型行为）；probe（探针 · 用线性分类器读隐状态里有没有某信息）；abstention（弃答 · 说"我不知道"/拒绝下结论）；answerability（可答性 · 这题到底有没有答案）；SDT = Signal Detection Theory（信号检测论 · 把"分得清"与"倾向于说是/说否"拆开）

---

## 主表（按冲突风险从高到低，共 30 条）

| arXiv ID | 题目 | 作者首位 + 机构 | 发表处 | 一句话相关性 | 冲突风险 | 推荐是否全文读 |
|---|---|---|---|---|---|---|
| 2510.01782 | Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks（Refusal Index） | Wenbo Pan（香港城市大学；通讯 Xinfeng Li，NTU） | **ICLR 2026 主会** | "答得准"与"该拒才拒"是两种独立能力；用"允许拒答 / 强制作答"两遍评测估拒答-错误秩相关 RI；改提示词拒答率大变、RI 不动、共同作答题正确性一致 >90% ⇒ **提示只动准则不动判别，是我们 H2 的行为学先例**；SimpleQA 为主数据集。不做方向/干预/证据侧 | 中（C1 承诺侧行为层先例；C3 门控必须证明 ΔRI>0） | 已全文读（2026-10-26，来自赵磊 Epistemic_Awakening 文献库），笔记 notes/2510.01782_refusal_index_knowledge_aware_refusal.md |
| allen-zhu24a | Physics of Language Models: Part 3.1, Knowledge Storage and Extraction | Zeyuan Allen-Zhu（Meta FAIR）+ Yuanzhi Li（MBZUAI） | **ICML 2024 主会** | 受控合成传记数据：模型能逐字背下全部事实却 0% 答得出问题，除非预训练数据经过改写/打乱增广；线性探针显示差别在"知识是否线性编码在实体名 token 上"。"存了≠提取得出"的最干净因果证据。不涉及弃答/方向/干预 | 低（前提引用：知道≠说出的训练侧根因） | 摘要级引用即可；赵磊库 A1 级有全文 |
| 2601.07422 | Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations | Wen Luo（北大 + 微软亚研） | **ACL 2026 主会** | 真实性探针信号来自两条通路：Question-Anchored（依赖问-答信息流）与 Answer-Anchored（答案自身的证据）；注意力敲除 + token patching 分离；与知识边界相关；12 模型 4 数据集（TriviaQA、PopQA、NQ 等）。关系到我们 E 方向"证据从哪来" | 低–中（E 侧：证据方向可能是两条通路的混合） | 待读（E 方向解释部分写作前读） |
| 2602.14080 | Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality | Nitay Calderon（Google Research + Technion） | **ICML 2026 主会**（PMLR v306） | "知道（编码）≠ 说得出（召回）"的大规模行为学证据：13 模型 450 万回复，前沿模型 95–98% 事实已编码但 26–34% 直接问说不出；四选一 verification 上反向题不难、闭卷生成才难；thinking 救回 40–65%。SimpleQA 只作判分提示词来源。**不做方向/干预/弃答**，不占我们 claim，但打到老师方案"按 baseline 对错切已知/未知"的做法 | 低（支持前提 + 修正已知/未知定义） | 已全文读（2026-10-26 补），笔记 notes/2602.14080_empty_shelves_recall_bottleneck.md |
| 2503.15299 | Inside-Out: Hidden Factual Knowledge in LLMs | Zorik Gekhman（Technion + Google Research） | COLM 2025 | 同一团队前作：用探针打分对比"内部知道"与"1000 次采样也生成不出"，平均缺口 40%，9% 的题内部识别出答案却从未生成；EntityQuestions 约 1,700 题，Llama-3-8B / Mistral-7B / Gemma-2-9B。"知道≠说出"的内部表示版 | 低 | 待读（与 2602.14080 一起作"知道≠说出"前提引用） |
| 2603.09906 | Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs | Gekhman 团队（Google） | 仅 arXiv（2026-03） | 用 SimpleQA-Verified 1,000 题子集，thinking 通过"计算缓冲 + 事实启动"扩大参数知识召回边界；是 2602.14080 的伴生文 | 低 | 不必全文，摘要级引用即可 |
| 2608.20794 | Knowing but Not Saying: Preventing Factual Access Failures in LLM SFT via Recall-Anchored Distillation | Haodong Chen（机构待核） | 仅 arXiv（2026-08） | 领域 SFT 后模型仍能在选择题上认出答案却闭卷答不出（"知道但不说"），提出召回锚定蒸馏；MedMCQA 三个骨干 | 无 | 不读，占坑级 |
| 2609.32964 | The Commit-Abstain Circuit: Why Language Models Hallucinate Instead of Abstaining | Vy Nguyen（RMIT，待核） | NeurIPS 2026 主会 | 把幻觉定义为 unsupported commitment（无依据的承诺），用 causal gating 定位 commit/abstain 决策的稀疏头+MLP 电路；早层累积承诺、晚层弃答信号"修正不足"；在电路激活上训练轻量策略把 false abstention 降 2.5×、跨基准迁移。与本方案"承诺方向"概念正面撞车，且已抢占 commit-vs-abstain 叙事 | 高 | 必读 |
| 2606.29490 | Reported Confidence in LLMs Tracks Commitment More Than Correctness | Dharshan Kumaran（Google DeepMind） | 仅 arXiv | 两阶段"先答→再决定提交/弃答"范式：verbal confidence 跟踪 commitment，token log-prob 跟踪 correctness（作者称 answer-evidence signal）；去掉共享方差后二者解耦；Gemma 3/4 机制分析显示 post-answer 状态提前编码弃答决定。几乎就是"证据 vs 承诺"二分，只是落在置信度而非 steering | 高 | 必读 |
| 2607.08456 | Two Axes of LLM Abstention: Answer Correctness and Question Answerability | Benedikt J. Wagner（City St George's, Univ. of London） | 仅 arXiv | 弃答应分两轴：答案对错 vs 问题可答性；answer-confidence 只看前者、hidden-state probe 只看后者，缺口不随规模缩小；提出双阈值校准策略（0.75 vs 0.31 coverage）。与本方案"已知/未知分开、两个独立分数"高度重合 | 高 | 必读 |
| 2608.29109 | Recognition-Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions | Yucheng Du（机构待核；合作者 Xiyang Hu） | EMNLP 2026 主会 | 单一线性 recognition（识别"此题无解"）方向与 safety-refusal 方向近乎正交（1.7B–70B 一致、预训练已存在）；沿 recognition 方向 steering 可双向、剂量响应地改变弃答；结论"routing failure 而非 encoding failure"。= 证据方向与（拒答）承诺方向分离的直接先例 | 高 | 必读 |
| 2606.24952 | Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models | Cosimo Galeone（Alomana） | 仅 arXiv | 直接量化"最佳检测方向"与"最佳控制方向"夹角：虚构实体检测 AUC=1.0，但与产生拒答的方向 cos≈0.12（约 83°）；4 模型 3 家族稳定、指令微调前后不变；向 refusal 方向旋转 15° 可部分桥接。同时打了"读≠写"和"知识方向≠弃答方向"两枪 | 高 | 必读 |
| 2509.22449 | Detecting (Un)answerability in Large Language Models with Linear Directions | Maor Juliet Lavi（Tel Aviv Univ.；合作者 Mor Geva） | EACL 2026 主会长文 | 用"activation addition 对弃答行为的影响"来挑 unanswerability 方向（按写效果选方向，而非按 probe 精度），再把隐状态投影上去做分类；加/消该方向可因果控制弃答，并迁移到科学争议/主观性引起的不可答。本方案"弃答方向 + 以干预效果选层"有直接前身 | 高 | 必读 |
| 2510.02324 | Hallucination reduction with CASAL: Contrastive Activation Steering For Amortized Learning | Wannan (Winnie) Yang（机构待核） | 仅 arXiv | 按 known/unknown 划分问题，构造两条 diff-in-means 向量（未知→弃答、已知→强化作答），再把 steering 摊销进单层子模块权重；明确讨论强度与已知题 over-refusal 的权衡。= 本方案"按输入门控、已知/未知打不同方向"的训练版 | 高 | 必读 |
| 2604.03867 | Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM Alignment | Soham Gadgil（Univ. of Washington） | EMNLP 2026 主会 | 理论+实证：最优 steering 层随输入变化；学一个"输入表征→层"映射（W2S），ID/OOD 均胜固定层基线。直接冲击"自动选写层"贡献点，而且比本方案多了逐输入选层 | 高 | 必读 |
| 2604.15557 | Predicting Where Steering Vectors Succeed | Jayadev Billa（机构待核） | 仅 arXiv | 免训练逐层诊断 LAP（用 unembedding 做 logit-lens 可达性 A_lin）预测 steering 有效性 ρ≈0.86–0.91、选层 ρ≈0.63–0.92；给出"diff-of-means 有效 / 需非线性 / 无法 steer"三体制；演示"中层启发式"失效。与"免训练分数选干预层"几乎同题 | 高 | 必读 |
| （无 arXiv；见 NeurIPS 2025 proceedings 页 112079–112101） | LayerNavigator: Finding Promising Intervention Layers for Efficient Activation Steering in Large Language Models | Hao Sun（中科院信工所，待核） | NeurIPS 2025 主会 | 用"可区分性 + 一致性"两项给每层 steerability 打分排序，复用计算 steering 向量时已有的激活，零额外数据、免训练；支持多层选择。与本方案"免训练分数自动选层"正面重叠，需对比其分数是否已能预测"读层≠写层" | 高 | 必读（proceedings PDF） |
| 2609.01048 | Lagged Coupling: Internal Representations Become Readable Before They Become Causal | Xining Xun（机构待核） | 仅 arXiv | Pythia 全系列（160M–12B）预注册研究：线性可读性 step 1k 即饱和（AUROC≥0.99），但沿同一 probe 方向 steering 在 43/48 格子里等价于零；"read-before-write" 11/11、无反例；写入量 <0.11% headroom。为"探针最准≠干预最有效"提供训练维度的系统证据 | 高 | 全文读 |
| 2610.17641 | Beyond Bias Scores: Answerability Confounds in Activation Steering with Gradient-Constrained Direction Design (GCD) | Weitong Li（机构待核） | 仅 arXiv | 对比 steering 把"是否回答（answerability）"与"偏好"混在一起：BBQ 分数变化中 answerability 通道在 17% 配置里超过一半、11% 里占全部；两方向近正交；提出带 unknown-logit 一阶约束的方向设计，答题率失真降 14×。= "steering 混入弃答成分"这一本方案核心动机的独立证据 | 高 | 全文读 |
| 2608.12852 | Falsehood and Impossibility Are Different Directions in an AI's Representation of Language | Yoon Pyo Lee（机构待核） | 仅 arXiv | Gemma 3 4B：truth 方向与 impossibility（必然为假/不可能）方向近正交；truth probe 分不开 impossible vs false（AUC 0.20），impossibility probe 分 necessary vs contingent falsehood AUC 1.00（layer 15）；SAE 复现同几何。标题句式与本方案撞车，"真假 vs 可答性"是另一种两方向切分 | 高 | 全文读 |
| 2510.13290 | To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models (MERA) | Anna Hedström（J.P. Morgan AI Research） | ICML 2025 | 同时优化干预方向、校准"何时/多大力度"steer，无把握时弃答，并有"不劣化"保证；可叠加在现有 steering 方法上。选择性 steering + 弃答的正式框架，评审必然要求对比 | 中 | 全文读 |
| 2602.05234 | Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions (CDAS) | Yuntai Bao（浙江大学，待核） | ICLR 2026 | 把 DAS 的 distributed interchange intervention 改成双向 steering 方法（分布匹配目标代替概率最大化），AxBench 评测。rank-1 DAS 当 steering 估计器的方法学先例——"用 DAS 估方向"本身不再新 | 中 | 全文读 |
| 2609.00760 | A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals | Yuri Son（Hanyang Univ.，待核） | EMNLP 2026 主会 | 知识型拒答（KR）与安全型拒答（SR）共享一条 refusal 方向但上层分化：KR 对齐不确定性/知识表征、SR 对齐策略表征；提出 commit-then-specify（先承诺拒答、再决定理由）。给"承诺方向"一个竞争性解释 | 中 | 全文读 |
| 2610.30363 | Probing and Steering for Safety-Relevant Behaviours in Large Language Models | Nimun Bajwa（QMUL，待核） | 仅 arXiv | 2 估计器×3 池化×逐层，每个方向既当检测器又当 steering 向量：logistic 读得好、diff-of-means 写得好的 read/write 分离在 deception/high-stakes 成立、在 sycophancy 反转；"steer 最佳层可远早于 detect 最佳层"。与"读层≠写层"直接相关的最新对照 | 中 | 全文读 |
| 2604.13068 | Detection Without Correction: A Robust Asymmetry in Activation-Based Hallucination Probing | Dip Roy（IIT Patna，待核） | 仅 arXiv | 7 模型 3 家族：probe 能检测幻觉，但沿 probe 方向 steering 7/7 失败；>410M 的模型输出置信度在检测上还胜过 probe。给"为何不能直接拿 probe 方向写"提供反例，也会被拿来质疑任何 probe→steer 链路 | 中 | 摘要 + 实验节 |
| 2510.24222 | HACK: Hallucinations Along Certainty and Knowledge Axes | Adi Simhi（Technion） | 仅 arXiv（待核） | 幻觉沿"知识"与"确定性"两轴分类、按模型构造数据；用 steering 验证知识轴（只有模型有参数知识时 steering 才有效）。= 本方案"已知题/未知题分开报、纠正/损伤/转弃答三分"的评测前身 | 中 | 全文读 |
| 2603.14893 | LLMs as Signal Detectors: Sensitivity, Bias, and the Temperature-Criterion Analogy | Jon-Paul Cacioli（机构待核） | 仅 arXiv | 预注册 168k 试次，用完整 SDT（unequal-variance 拟合、criterion、z-ROC）把 LLM 事实判别拆成 sensitivity 与 bias；指出 ECE 等校准指标混淆二者；z-ROC 斜率 0.52–0.84。本方案"Youden's J vs 标签偏置"的 SDT 框架已有人先铺 | 中 | 全文读 |
| 2608.13167 | TRAPSBench: Vision-Language Models Encode but Fail to Express Epistemic Restraint | Fnu Pramono（机构待核） | COLM 2026 | probe 从隐状态解码可答性 AUROC 0.91，单层 void direction steering 可因果诱发/抑制弃答；提出 PECS 指标同时要求"能答答对、不能答弃答"。多模态版的"编码有、表达无 + 弃答方向" | 中 | 摘要 + 方法节 |
| 2605.22007 | Hallucination as Commitment Failure: Larger LLMs Misfire Despite Knowing the Answer | Jewon Yeom（机构待核） | 仅 arXiv | 0.8B–72B：16–47% 的 Instruct 幻觉发生时正确概念已有大量概率质量，且比例随规模单调上升；差别在"承诺如何分布"而非"有没有表征"；预生成隐状态可检测。把"有证据但承诺错了"做成实证叙事 | 中 | 全文读 |
| 2609.06951 | Steering Interference Reflects the Model's Defaults, Not the Behavior Directions | Srikanth Malla（机构待核） | 仅 arXiv | 24 行为×10 模型：任何 steering（含无语义的等范数向量）都把模型推向其默认行为——首当其冲是 refusal/sycophancy；24 个可线性解码行为只有 20 个真能改输出。暗示"truth steering 顺带增加弃答"可能是范数伪影，威胁"承诺方向"的因果解释 | 中 | 全文读 |
| 2604.03754 | Testing the Limits of Truth Directions in LLMs | Angelos Poulis（Boston Univ.） | BlackboxNLP 2026 | truth 方向强烈依赖层、任务类型（事实题早层、推理题晚层）、难度；一句"评估正确性"的指令就显著改变 truth probe 几何与泛化。直接威胁"用 system prompt 对比估承诺方向"时与证据方向的可分性 | 中 | 全文读 |
| 2603.25052 | Closing the Confidence-Faithfulness Gap in Large Language Models | Miranda Muqing Miao（UPenn） | 仅 arXiv | 校准信号与 verbalized confidence 方向线性且正交（3 模型 4 数据集）；两阶段自适应 steering：先读内部准确度估计，再把表达拉齐。结构上等于"读证据、写表达"的门控 steering，只是对象是置信度 | 中 | 摘要 + 方法节 |
| 2606.12160 | A Controlled Study of Decoding-Time Truthfulness Methods on Instruction-Tuned LLMs | Ao Sun（机构待核） | 仅 arXiv | 六重控制（OOD 训练、多 judge、简单解码基线、混杂、bootstrap、seed）下 ITI 等 15 种方法在 instruct 模型 TruthfulQA 上增益消失；列出 5 种评测敏感性。若本方案在 TruthfulQA 报增益，评审会按此要求控制 | 中 | 全文读（评测章） |
| 2504.02904 | How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence | Hongzhe Du（UCLA） | COLM 2025 | truthfulness 方向在 base/post-trained 间高度相似且可迁移干预；refusal 方向则不同、迁移差；confidence 差异不能归于 entropy neurons。给"证据方向稳定、承诺方向受后训练塑造"提供先验 | 中 | 摘要 + 相关节 |
| 2609.21996 | A Lie Detector Test for Language Models: Reading Knowledge a Model Won't Reveal | Hiskias Dingeto（机构待核） | 仅 arXiv | PIR：把 Concealed Information Test（隐匿信息测试 · 看嫌疑人对真细节的反应）搬进模型，读出模型"认得"的答案；在 prompted deception、sandbagging、password-lock 下识别率 0.85–0.93，unlearning 后降至未知水平。"知道但不说"的证据侧读出法 | 中 | 摘要 + 方法节 |
| 2608.02957 | Inverted Detection and Control in Steering Vectors | Max Torop（Northeastern，待核） | 仅 arXiv | 高判别力的 steering 向量可能把行为推向反面（ISV）；给出不需生成的判别法，翻转符号后 ITI 在 27/30 实验提升（Gemma 3 12B / Qwen 2.5 14B / Olmo 3 7B）。再证"读方向≠写方向"，并提供修补 ITI 的基线 | 中 | 摘要 + 实验节 |
| 2512.08121 | Balanced Accuracy: The Right Metric for Evaluating LLM Judges — Explained through Youden's J statistic | Stephane Collot（机构待核） | EACL 2026 Industry | 论证 Youden's J（= Balanced Accuracy 的线性变换）是选判别器的理论正确指标，对类别不平衡与正类选择不敏感。本方案若主打 Youden's J，此文是最近的方法学引用/竞争 | 中 | 摘要 |

---

## 附表：边缘候选（风险低或为方法学/评测配套，未进主表；按主题分组）

**A. 已知/未知门控、选择性 steering、自适应强度**
- 2508.17621 Steering When Necessary (FASB) · NeurIPS 2025 · 跟踪生成中内部状态决定"是否/多强"干预 + 回溯；TruthfulQA
- 2512.03661 Dynamically Scaled Activation Steering (DSAS) · Apple · 仅 arXiv · 把"何时 steer"与"如何 steer"解耦，逐层逐输入缩放
- 2504.09466 AdaSteer · EMNLP 2025 主会 · 沿 rejection 与 harmfulness 两方向按输入自适应系数（安全域的"两方向+门控"同构）
- 2609.02089 IDEEA · EMNLP 2026 Findings · 免训练输入依赖 steering（按 head 聚类匹配），TruthfulQA
- 2606.28589 DynaSteer · ICML 2026 · 推理链中只在高熵分叉处 steer 并可回滚，明说"朴素 steering 会误伤正确轨迹"
- 2609.38962 USteer · NeurIPS 2026 主会 · 用置信度对激活的梯度做免训练 steering 减幻觉
- 2608.14465 YOPO · 仅 arXiv · steering 写入会干扰 sufficiency 方向的读出，训练小网络重建写前残差——"写破坏读"的直接实验
- 2610.07270 Does the Model Use the Feature? · 仅 arXiv · 已知/未知弃答 latent 的 installation vs removal 不对称
- 2607.13568 Graded Entity-Familiarity Readouts … Refusal Steering · 仅 arXiv · 一维 familiarity 方向在已知/未知实体上单调双向移动拒答率
- 2604.14324 Purging the Gray Zone (GeoDe) · ACL 2026 Findings · 用 truth 超平面距离当弃答置信度做去噪微调

**B. 读层≠写层 / probe≠steer 方法学**
- 2605.05715 Decodable but Not Corrected by Fixed Residual-Stream Linear Steering · 仅 arXiv · 医疗 QA，29 配置 steering 全为零效应
- 2607.00158 Readable but Not Controllable · 仅 arXiv · 医疗幻觉神经元级可读不可控
- 2608.07528 The Knowing-Saying Gap: When Probes See Errors that Confidence Misses · 仅 arXiv
- 2605.25151 Representation Without Control · 仅 arXiv · 行为敏感、线性可读、因果可控三者不共现
- 2602.06801 On the Non-Identifiability of Steering Vectors · 仅 arXiv · 单层白盒下 steering 方向存在大等价类（威胁"这就是那条方向"的叙事）
- 2610.19359 Competing-Target Test for Activation Steering · EPFL · 仅 arXiv · 证明两条向量不可互换的检验协议——本方案应做
- 2608.08159 When Is a Steerable Concept Representation Real? · 仅 arXiv · 单位/读出/操作点混杂审计
- 2609.07876 LLM Layers Immediately Correct Each Other · NeurIPS 2025 · 相邻层互相抵消，解释为何 steering 需大幅放大
- 2610.16084 The Sharp Directions Are Against You · NeurIPS 2026 · CAA 向量的曲率尖锐分量导致反向 steer
- 2602.01716 Mechanistic Indicators of Steering Effectiveness · 仅 arXiv · 用熵/KL 预测 steering 成败
- 2606.11599 When is Your LLM Steerable? · 仅 arXiv · 从早期解码状态预测 under/over-steer
- 2608.08829 Deployable Per-Instance Multi-Layer Activation Steering · ARR 2026-08 · 逐实例多层选择
- 2601.19375 Selective Steering · 仅 arXiv · 按类别对齐符号判别选层

**C. 真方向几何、上下文/指令敏感性**
- 2506.08572 The Geometries of Truth Are Orthogonal Across Tasks · Apple · 仅 arXiv · 跨任务 truth probe 几乎不共享支撑
- 2601.06599 How Context Shapes Truth · ACL 2026 主会 · 上下文改变 truth 向量方向/幅度
- 2609.10739 The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes · 仅 arXiv · 顺从语境下"真值"与"任务规定答案"标签重合→probe 失真（估证据方向的采样警示）
- 2510.09033 Do LLMs Really Know What They Don't Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness · ACL 2026 Findings（待核）· "证据方向"可能只是"回忆方向"
- 2607.16741 The Anatomy of a Truth Direction · 仅 arXiv · 免训练 SVD 主方向，14 模型 6 家族
- 2505.21800 From Directions to Cones · 仅 arXiv · 真值为多维锥
- 2506.00823 Probing the Geometry of Truth · ACL 2025 Findings
- 2608.03035 Language Models Encode the Contextual Truth of Propositions · 仅 arXiv · 表征与输出分离区分两种 sycophancy
- 2610.04512 Correctness Is a Direction · 仅 arXiv · 事实/领域/校准真实性方向近正交
- 2601.07422 Two Pathways to Truthfulness · ACL 2026 主会 · 真实性信号的两条信息通路（问题锚定 vs 答案锚定），与知识边界相关——标题易与"两方向"混淆，需在相关工作中区分

**D. 承诺/预承诺、置信度机制**
- 2605.06723 When Does a Language Model Commit? · 仅 arXiv · 有限答案承诺码在可见答案前稳定，线性可读但 steering 不可靠
- 2603.01437 Post-Hoc Reasoning in CoT: Decoding and Steering Pre-Committed Answers · 仅 arXiv
- 2604.15400 Hallucination as Trajectory Commitment · 仅 arXiv
- 2603.17839 How do LLMs Compute Verbal Confidence · Kumaran · 仅 arXiv（已读 2603.22161 的姊妹篇）
- 2604.22271 How LLMs Detect and Correct Their Own Errors · Kumaran · 仅 arXiv
- 2511.13240 Knowing What You Know Is Not Enough: LLM Confidences Don't Align With Their Actions · 仅 arXiv · 行为层的"证据≠承诺"
- 2604.01457 Wired for Overconfidence · COLM 2026 · 口头过度自信的电路与推理时再校准
- 2605.08942 Decomposing and Steering Functional Metacognition · 仅 arXiv · 多个元认知状态可分别解码/可分离地 steer

**E. 评测与 SDT**
- 2610.11986 FearCaut-Qwen · 仅 arXiv · 用 SDT 把 steering 的效果拆成 d' 不变、criterion 移动——与本方案"判别力 vs 偏置"同一刀法（VLM 域）
- 2510.04302 When Guessing is Rewarded · NeurIPS 2026 D&B · 区分"错答过度自信"与"弃答"的评测指标
- 2605.17028 PARALLAX · NeurIPS 2026 投稿 · 幻觉检测基准的构造伪影
- 2608.28930 The Hallucination Signal Is a Mean Shift · EMNLP 2026 主会 · 幻觉信号由单一均值位移主导
- 2602.06256 Steering Safely or Off a Cliff? · EACL 2026 主会 · steering 的 specificity 三维评测（含 overrefusal / faithfulness hallucination）
- 2509.13450 SteeringSafety · ICML 2026 · 含 hallucination / epistemic integrity 视角与 conditional steering

**F. honesty / 两方向的结构类比**
- 2507.11878 LLMs Encode Harmfulness and Refusal Separately · 仅 arXiv（待核）· "判断方向 vs 行为方向"分离的安全域原型，本方案叙事的最佳类比
- 2512.07667 Depth-Wise Activation Steering for Honest Language Models · ICLR 2026（待核）· MASK 上多层高斯调度 honesty steering
- 2604.08169 Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence · 仅 arXiv · 按 logistic 边界只对越界 token 干预，MASK honesty 53.6→81.2%
- 2605.03907 Steer Like the LLM (PSR) · ICML 2026 · 把 prompt steering 形式化为 activation steering——与"用 system prompt 对比估承诺方向"直接相关
- 2604.09839 Steered LLM Activations are Non-Surjective · ICLR 2026 workshops · steering 推到 prompt 不可达区域
- 2508.04530 StyliTruth · ICLR 2026（待核）· 正交 deflation 分离 style/truth 子空间
- 2502.12446 MAT-Steer · ACL 2025 · 多属性（含 truthfulness）steering 向量强制正交
- 2506.08359 REAL · 仅 arXiv · 用 VQ-AE 定位 truthfulness steering 模块，比 ITI 平均 +20%
- 2506.11088 One SPACE to Rule Them All · NeurIPS 2025 · 事实性/忠实性幻觉共享子空间（与"解耦"反向的论点）
- 2509.18058 Strategic Dishonesty · ICLR 2026 · probe 检测/steer 策略性不诚实
- 2606.12618 "Did you lie?" Evaluating Lie Detectors · 仅 arXiv · 现有 lie detector 在信念验证过的模型上大幅掉点
- 2603.05494 Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation · 仅 arXiv

---

## 覆盖说明与负结果

1. **未找到**：用"不确定就说不知道 vs 必须给确定答案"这类 system prompt 对比来估计弃答/承诺方向的论文。最接近：CASAL（用 known/unknown 问题对比）、2605.10664 GCAD（从 system-prompt 对注意力的贡献提取 steering 信号）、2605.03907 PSR（prompt steering 形式化为 activation steering）。这一估计法目前看仍是空位。
2. **未找到**：把 Youden's J 用于 truthfulness steering 评测的论文；J 只出现在 LLM-judge 选择（2512.08121）与 probe 阈值选取（2606.02628）。但 SDT 的 sensitivity/bias 拆分已被 2603.14893、2610.11986、2609.18204（已读）用上。
3. **DAS 用于 truthfulness / hallucination**：只有 CDAS（2602.05234，通用 steering 方法）；没有 truth-specific 的 DAS 文章。
4. **中文社区**：知乎/机器之心/量子位未检索到 2026 年 truthfulness steering 的专题报道，只有 ITI、"模型知道的比说的多"等旧文。
5. **会议扫描**：ACL 2026 主会/Findings、EMNLP 2025/2026、NeurIPS 2025/2026、ICLR 2026、ICML 2025/2026、COLM 2025/2026、EACL 2026 均已覆盖到上表相应条目；EMNLP 2026 刚公布接收（2608.29109、2609.00760、2604.03867、2602.02132、2603.27518、2608.28930、2609.03887），NeurIPS 2026（2609.32964、2609.38962、2610.16084、2510.04302、2609.37616）需等 camera-ready 再核最终版本。

---

## 总结（≤200 字）

最可能动摇新颖性的是五篇：**2609.32964**（NeurIPS'26，Commit-Abstain 电路，已抢占"承诺"叙事）、**2606.29490**（Kumaran，证据信号 vs 承诺信号的同一二分）、**2608.29109**（EMNLP'26，识别方向⊥拒答方向且可 steer 弃答）、**2606.24952**（检测方向 vs 控制方向夹角，读≠写 + 知识≠弃答）、**2607.08456**（弃答两轴、双阈值策略）。另有三篇直接压"免训练选写层"贡献：**2604.15557**（LAP）、**LayerNavigator**（NeurIPS'25）、**2604.03867**（EMNLP'26）。本方案剩余的差异化空位：rank-1 DAS 同时估两条方向、system-prompt 对比估承诺方向、Youden's J + 三分评测。

# 本轮全文阅读记录与方向依据

2026-11-04。只把读完正文、参考文献和所附附录的论文用于方法结论；优先正式会议版本。以下为主审及独立复核合并记录，页码指PDF物理页。原文件与提取文本保留在本地 `from_gpt/references/`，不以摘要或搜索片段替代全文。

这不是为05列出的17篇逐篇背书。与本次取舍直接相关的论文如下；其他文献没有在本轮完整复核，因此不借它们下排他性结论。

| 论文与正式来源 | 全文覆盖 | 核验后可用的事实／需纠正处 |
|---|---|---|
| [Finding Alignments… / DAS，CLeaR 2024](https://proceedings.mlr.press/v236/geiger24a.html) | 28/28页，附录A–E | §3.5允许一般可微目标，本文实验使用反事实交叉熵；分类与BERT `[CLS]` 是实验设置，不能升级成DAS必须使用真假标签的定义。pp7–9、12、22–23 |
| [Interpretability at Scale / Boundless DAS，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6a8b109d4d4fd64c75e94aaf85d9697-Abstract-Conference.html) | 22/22页，附录A.1–A.7；另核官方旧补充附录5页 | 价格区间判断用单词Yes/No反事实监督。p17 A.3显示来源原答案与交换后的反事实目标可不同，故不能直接互换“复制源分布”和“正确反事实标签” |
| [RAVEL，ACL 2024](https://aclanthology.org/2024.acl-long.470/) | 19/19页，附录A–C | p15 B.3“首词近似”属于RLAP；p16 B.4才是DAS/MDAS。干预输入实体末词元。MDAS还约束非目标属性保持，不能遗漏这一半解耦目标 |
| [LLMs Know More Than They Show，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a712d461e57201efe35d429a6f1731c1-Abstract-Conference.html) | 34/34页，附录A–F | 表1的0.85/0.76/0.71是Mistral在TriviaQA的检测结果；并非Llama/Qwen的干预收益。§4、附录C保留同技能迁移；§6是候选重排，不能推论“96题训练必死记” |
| [Controllable Context Sensitivity and the Knob Behind It，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c9d780d1e2d57d4b70e807608a72501b-Abstract-Conference.html) | 30/30页，附录A–I | 05遗漏的相关先例：§3.4直接监督问答内容答案，在提示末位学单层秩1交换方向，§4.4删意图句后设定投影控制。但附录I要求一词回答，任务是上下文/先验取舍，不是长回答真实性 |
| [ReFT，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html) | 55/55页，附录A–J与检查清单 | §3.2对全部目标输出训练，主要干预提示位置。LoReFT常识/算术为前7后7及全层；指令微调表7 p24实际为前5后5，层3/9/18/24。“必然塞知识、只能当上限”没有被证明 |
| [Concept DAS / CDAS，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/704357127afabbd5a6eb979a57810767-Abstract-Conference.html) | 55/55页，附录A–K | §3.3训练匹配反事实上下文分布，§4推理设定投影坐标；不等于放大自身坐标。p5明确不是严格因果变量定位。TruthfulQA在§5.1作为能力保持测验，不能说文中完全未用该数据集，也不是直接真实性生成实验 |
| [CAA，ACL 2024](https://aclanthology.org/2024.acl-long.828/) | 19/19页，附录A–L | §4.2与附录G有单层开放回答的幻觉行为对照；附录H/Table12的TruthfulQA则是二选一正确答案概率。不能把它混成官方MC1，也不宜声称自由生成控制一概必须多层 |
| [ITI，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html) | 80/80页含全部案例、附录B–F；本线程前轮完整阅读，本轮核对同一文件哈希并重读方法、表1–5及附录 | 答案末位估头方向、逐步加法有先例；p80附录F明确估头激活不用生成阶段的标准示例提示。表3的42.3与34.8分别使用α20与α15，不能拼成同强度配置 |
| [TruthX，ACL 2024](https://aclanthology.org/2024.acl-long.483/) | 42/42页，附录A–F、F.1–F.38全部案例；本轮复核既有全文记录 | §3.3与表6的k是注意力/前馈模块数；原生表示上的编辑量经过编码解码，不能直接等同我们残差层叠加。附录C提供经典示例提示；F中仍有混合拒答、答非所问和改坏原答案，固定拒答词不保证判分无歧义 |

合计10篇正式PDF、384页；另核对Boundless官方旧补充附录5页。页数包含参考文献、案例与检查清单，**不把页数当成证据强度**；ITI沿用本线程已完成的全文阅读，本轮做同版核验，不声称重新逐页读了第二遍。来源文件哈希见 [`paper_sources.json`](paper_sources.json)。

## 对05五个问题的回答

1. **A还是B：本轮不把二者设为必选二选一。** 用户已允许先做均值差。A的真假自评后缀增加了一次任务迁移；B是学习低秩控制器，不能预先判为只记忆。新增Knob先例还表明，问答内容监督不必先变成真假自评，但其控制变量不能直接改名为本项目的E。
2. **位置与目标：先把反事实定义清楚。** 若目标是正确标签，则应优化该标签；若目标是复制源分布，需要说明源分布为何代表所需反事实。答案末位或精确答案位置“易检测”，不等于从那里学出的方向能改善尚未生成的答案。
3. **单层/多层：先单层诊断，之后再做预定窗口对照。** 各层作用和激活尺度不同，相同α不能保证总扰动可比。多层对照应同时报告实际扰动大小及候选预算，不能凭原论文的模块数断定本项目窗口大小。
4. **开场白：主线推荐保留E24原句。** 这是为了保住当前跨实验的可解释性。经典示例提示可以独立比较；装进聊天模板后应另命名协议。没有精确复现原模型、评分与数据切分时，不能直接与旧论文数字排名。
5. **顺序：修确定错误→三题真实线路核验→再谈扩大。** 把E/C解开；实际作答率须从真实回答测，不能新增自报指标就算解决。DAS训练目标/应用方式的改动留到第二阶段，不同时引入提示、位置、目标、多层四种变化。

## 关键证据的补充限度

**Knob代码核验**：作者公开 `encode_answer` 和 `compute_loss` 只编码一个答案词元、对提示末位输出做交叉熵，与一词提示一致。因此它支持“直接内容监督”，不支持“已验证整段长答案目标”。[作者实现](https://github.com/kdu4108/context-vs-prior-finetuning/blob/main/analysis/circuit_utils/utils.py#L152)、[干预训练库](https://github.com/jkminder/nnpatch/blob/main/nnpatch/subspace/interventions.py#L148)。正式PDF附录已完整读取；官方补充ZIP下载不完整，没有声称读完该代码包，公开代码也不冒充会时锁定版本。

**ReFT记忆实验的边界**：附录F确实研究刻意长时间训练下的记忆能力，并承认预训练记忆可能参与；它没有检验本项目96题，更没有证明所有小样本答案监督都会失去泛化。其独立测试与定性小样本例子也不能反过来保证我们会成功。这里是拒绝超出证据的定性，不是在替ReFT作效果承诺。

**CDAS的边界**：附录E.2关于不同分布损失的比较仅在两个概念上进行；不能据此认定我们的KL一定应换JSD。本文使用“设定坐标”的部署方式，和本项目“相对中心放大自身坐标”不同，是需要区分的机制；其优势也随任务和模型变化。

**TruthX实现核验**：作者公开代码按模块索引读取正负中心，经同一编码/解码模型得到依赖当前状态的编辑量。因此“多模块编辑有效”有依据；“十模块就是我们十个固定残差方向”的等同没有依据。[作者实现，`truthx.py`](https://github.com/ictnlp/TruthX/blob/main/truthx.py#L260)。本文没有由论文一个模型的多层曲线预测Llama-3.1或Qwen3必然同样获益。

本次新增检索没有取得“与我们同任务、同读出、同操作的DAS自由生成真实性通用配方”这一层级的证据；这是本轮检索与全文覆盖的边界，**不是宣称全领域不存在**。

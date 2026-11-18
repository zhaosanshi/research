# 查重：用“内部读得出的 − 嘴上说出的”差值事先预测干预收益（排期 D1）

撰写：赵磊（Opus 5.5 代笔）。北京时间 2026-11-19。检索与读全文由 Opus 5.5 子代理完成，表中 11 篇都读了正文，只记事实；创新点由博士判断。

2026-11-19（北京时间）检索。只记别人做了什么、和我们差在哪；建议单列在文末。

**我们的计划用法**：用写入 E 的同一根方向量出"读出判别力 − 输出判别力"的差值，在新题集上跑之前写下对"写 E 后净分（答对 − 答错）变化"的预测，跑完对照。

**术语**：AUROC（曲线下面积 · 0.5 是瞎猜，1 是真假全分开）；ρ / r（秩相关 / 相关系数 · 两组数排序或高低是否一致）；arXiv（预印本网站 · 论文没经过同行评审也能挂）；workshop（研讨会 · 主会旁的小会，审稿较松）；post-hoc（事后的 · 看完结果才分析）；go/no-go（做不做的判断）；unknown（不知道）；probe（探针 · 在模型内部状态上训的小分类器）；steering（激活写入 · 往隐状态里加一个方向来改输出）；mean difference（均值差 · 两类样本平均状态之差的那根方向）；logit lens（输出透镜 · 用模型自己的输出层去读中间层）；d′ / SNR（可分度 / 信噪比 · 两类中心离得多远、相对各自散得多开）；calibration（校准 · 只挪判定门槛，不改排序）；few-shot（少样本提示 · 提示里先给几个例题）；GV-Gap（验证−生成差 · 模型挑自己答案的本事比直接答高出多少）。数据集：TruthfulQA（真实性问答集）、BBQ（偏见问答集）、TriviaQA（常识问答集）、NQ（谷歌搜索真实问题集）、PopQA（按实体热度分层的事实问答集）。

**检索范围**：本地文献库 75 篇按关键词在全文里查过；网上只用通用关键词。下表 11 篇都读了正文（附录多为抽读），按权威度排：顶会主会 > workshop > 预印本。另有 5 篇只读了部分正文，按规矩不列入表：arXiv 2410.02707、2603.25052、2606.24952、2607.29062、2606.20959。

| # | 题目 | 发表处 · 年 | 链接 | 他们做了什么（差值或预测怎么用） | 与我们的差别 |
|---|---|---|---|---|---|
| 1 | Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions（KAPPA） | ICML 2026 主会 | [arXiv:2509.23782](https://arxiv.org/abs/2509.23782) | 选择题上量"知识探针准确率 − 模型准确率"（TruthfulQA +19.6～+21.3 点，BBQ-Age Llama +29.3）；正文在差值大的 6 个题集上评估干预 KAPPA（用同一对探针把"要选哪项"的坐标换成"哪项对"的坐标）；附录 E.1 另报差值小的知识题集：一致率升、准确率只小变，并写知识探针准确率本身低"limiting the potential upper bound"；唯一的正式相关是"子空间夹角 vs 差值"（Llama ρ = 0.976） | 量差值与写入用同一组探针，这点与我们相同；没用差值对收益做预测或相关检验；指标是选择题准确率，无弃答、无净分 |
| 2 | To Steer or Not to Steer? Mechanistic Error Reduction with Abstention（MERA） | ICML 2025 主会 | [arXiv:2510.13290](https://arxiv.org/abs/2510.13290) | 探针预测"这题会错多少"，按预测值逐题定写入力度；在校准集上带统计保证地挑阈值，没有阈值显著提升就整体不写 | "写不写"靠在校准集上直接试，不靠差值；分类题准确率；模型 ≤ 3B |
| 3 | Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models | ICLR 2025 主会 | [arXiv:2412.02674](https://arxiv.org/abs/2412.02674) | GV-Gap = 按模型自评重新加权后的得分 − 原得分，即"自评过滤"这一步的收益本身；推出它减去蒸馏误差后是后续微调收益的下界；作者写分析"primarily post-hoc"，事先预测"能否自我改进"留作未来工作；事实题 NQ 上差值 < 1% | 差值在行为层面（提示模型自评），不读内部；干预是重排、微调，不是激活写入 |
| 4 | Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations | EMNLP 2025 主会 | [doi:10.18653/v1/2025.emnlp-main.187](https://doi.org/10.18653/v1/2025.emnlp-main.187) | 逐题按"语义不确定度 − 说出的不确定度"定沿"说不确定"方向写的力度，只写被判为幻觉的回答；TriviaQA、NQ、PopQA 上自信幻觉相对降约 30%，正确率小降 | 差值逐题定力度，没有数据集级的收益预测；写的是语气方向，不是真假方向；无净分 |
| 5 | Inside-Out: Hidden Factual Knowledge in LLMs | COLM 2025 主会 | [arXiv:2503.15299](https://arxiv.org/abs/2503.15299) | 内部探针与外部打分的排序能力差（平均相对 40%）；采 1000 个答案用探针挑，比贪心相对 +12%；附录 A.12 推导排序能力与"采 n 挑 1"成功率的关系，并写明差值更大不保证更受益 | 干预是采样重排，不是写入；没有事先预测再检验 |
| 6 | Inference-Time Intervention: Eliciting Truthful Answers from a Language Model（ITI） | NeurIPS 2023 主会 | [arXiv:2306.03341](https://arxiv.org/abs/2306.03341) | TruthfulQA 上探针准确率比生成准确率高约 40%，作为动机；按探针准确率挑注意力头写入 | 差值只作动机，没用来预测收益 |
| 7 | Understanding (Un)Reliability of Steering Vectors in Language Models | ICLR 2025 Workshop | [arXiv:2505.22637](https://arxiv.org/abs/2505.22637) | 36 个行为数据集：沿均值差写入方向的可分度 d′、激活差与写入向量的余弦一致度能预测写入成效 | 只看"沿写入方向读得多清楚"，不减输出；行为倾向，不是答对答错 |
| 8 | When Do Internal Probes Beat Reading the Answer? Miscalibrated Readouts and Behavior-Concealed Knowledge in Language Models | arXiv 预印本 2026 | [arXiv:2609.04582](https://arxiv.org/abs/2609.04582) | 行为 < 输出层 YES−NO 差 < 探针的"信息阶梯"；按三者关系分三种情形并写明各自哪种修法有效；事先预言 few-shot 是靠挪门槛修好的，配对测量证实（0.6B：50% → 约 82%） | 修法是校准、提示，作者写明没做激活写入；合成逻辑题，≤ 8B；定性分区，不是数值预测 |
| 9 | Predicting Where Steering Vectors Succeed（LAP） | arXiv 预印本 2026 | [arXiv:2604.15557](https://arxiv.org/abs/2604.15557) | 用 logit lens 准确率事先预测均值差写入能否成（ρ = +0.86～+0.91）、在哪层成；go/no-go 阈值在 3 个模型上定、2 个模型上验；"探针 − logit lens"只作定性分区 | 预测量不减模型实际输出；收益是目标词概率变化，单词元补全 |
| 10 | Signatures of Steerability in Activation Space of Language Models | arXiv 预印本 2026 | [arXiv:2609.14151](https://arxiv.org/abs/2609.14151) | 38 个选择题集 × 6 个模型：写入向量的 SNR 与写入成效相关 r = 0.76 | 同第 7 行 |
| 11 | Do Probing and Steering Favor the Same Directions Within and Across Language Models? | arXiv 预印本 2026 | [arXiv:2611.19499](https://arxiv.org/abs/2611.19499) | 读得最好的方向不是写得最好的方向（真假概念上最佳探针方向的写入效果只有均值差的 21–64%）；假设与判据事先写定；PopQA 上往"unknown"写，弃答增加 | 不涉及用差值预测收益 |

## 对照我们用法的四个要素（只限上表 11 篇）

- **用写入的同一读头量差值**：KAPPA 量差值和写入用同一组探针；第 7、10 行用写入向量本身的可分度，但不减输出。
- **差值是"读出 − 输出"**：第 1、3、4、5、6、8 行各有一种（知识探针 − 模型、自评 − 生成、语义不确定 − 说出的不确定、内部 − 外部排序、探针 − 生成、探针 − 输出层 − 行为）；第 9 行的"探针 − logit lens"减的是输出层读出，不是模型实际答案。
- **收益口径是净分**：11 篇都没用"答对 − 答错、弃答记 0"作收益；用的是准确率、目标词概率、幻觉率、弃答百分点。
- **事先写下、新数据上检验**：第 8 行有一条定性的事先预言并证实；第 9 行阈值换模型验证；第 2 行在校准集上带统计保证；第 11 行事先写定判据，但测的不是差值预测收益。表中没有一篇对"差值 → 写入收益"做事先登记的数值预测。

## 建议（单列）

- 写 D2 时把第 1、5、8、9 行作最近的对照来引：同读头但不预测（1）、差值大不保证受益（5）、阶梯与修法预言（8）、只看读出预测写入（9）。
- 可仿第 8 行加一级"只看输出层"的读出，把"只挪门槛就能拿到的收益"和"必须写入才能拿到的收益"分开报。
- 收益除净分外同时报准确率或 ΔAUROC，便于和第 1、2、6 行对照。

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-19 | 建立本文件 | 赵磊（Opus 5.5 代笔） |

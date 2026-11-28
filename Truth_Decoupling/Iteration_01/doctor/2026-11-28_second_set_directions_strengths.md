# 第二套题上的方向和力度：沿用 TruthfulQA 的，还是每套题各自定

撰写：赵磊（Fable 5.2 代笔）。起始日期：2026-11-28（北京时间）。这一页只问一件事，是[简化方法意见稿](2026-11-28_simple_rule_routes.md)第 1 节的展开。文献部分只列事实和原文；看法单列在第 5 节，标明是看法。

**请博士裁定：FalseQA（第二套题）上用的 E、C 两个方向和它们的力度，是直接沿用在 TruthfulQA 上学到、选好的那一套，还是用 FalseQA 自己的题重新学方向、重新选力度？**

现状：到目前为止，FalseQA 上的全部结果都是沿用的。博士裁定之前，实验先按沿用继续做（赵磊 11-28 暂定）。

## 1. 方法，和现在的方向、力度从哪来

每道题都用同一个普通问法。先得到只拧 E 的回答；E 没答时，再看原模型（不拧任何方向）的回答。然后按下表决定交什么：

| 原模型 | 只拧 E | 最后交什么 |
|---|---|---|
| 答了 | 答了 | 交 E 的回答。两份回答不一样时也交 E 的，不看原模型答了什么 |
| 没答 | 答了 | 交 E 的回答 |
| 答了 | 没答 | 不答。原模型的回答不交；它只用来判断“这次不答是 E 造成的” |
| 没答 | 没答 | 同时拧 E 和 C，把原题再问一次。答了就交这份回答；还是没答，就不答 |

- E：管回答内容对不对的方向。C：管敢不敢答的方向，这里往敢答推。两个方向都是学出来的。
- TruthfulQA 的 817 道题分三份：96 道学方向（约 12%），192 道调参（约 23%），529 道测试（约 65%）。
- 每个模型的 E 层、E 力度、C 层、C 力度，都在那 192 道调参题上选；E 的力度同时受 MMLU 掉分的限制。
- FalseQA 上，方向、层、力度原封不动拿过来，没有用 FalseQA 的任何一道题来学方向或选方法的力度。

| 模型 | E 层 | E 力度 | C 层 | C 力度 |
|---|---|---|---|---|
| Llama-3.1-8B | 16 | 3.5 | 19 | 2 |
| Qwen3.5-9B | 15 | 2 | 16 | 0.5 |
| Qwen3.8-27B | 28 | 1.75 | 33 | 1 |
| Gemma-3-12B | 24 | 2.5 | 25 | 2 |
| Gemma-3-27B | 28 | 1 | 31 | 2 |

## 2. 三种做法

| 做法 | 方向在哪学 | 力度在哪选 | FalseQA 自己的题用来做什么 | 还要做的事 |
|---|---|---|---|---|
| A 沿用（现状） | TruthfulQA 的 96 道题 | TruthfulQA 的 192 道调参题 | 只用来测试 | 没有 |
| B 方向沿用，力度各自选 | TruthfulQA 的 96 道题 | FalseQA 自己的调参题 | 一部分选力度，其余测试 | C 的力度已经能离线算（第 4.3 节）。E 的力度要在 FalseQA 的调参题上新跑几档；选出的档和现在不同，FalseQA 的回答就要重新生成、重新判 |
| C 方向和力度都各自定 | FalseQA 自己的训练题 | FalseQA 自己的调参题 | 一部分学方向，一部分选力度，其余测试 | 用 FalseQA 的官方训练集（1,187 对题）造训练材料；五个模型重学 E 和 C，重选层和力度；FalseQA 的回答全部重新生成、重新判；程序先交 GPT 审。另外要验证：FalseQA 上学出的 E 和 TruthfulQA 上学出的 E 是不是同一个方向 |

做法 C 的分法可以照 TruthfulQA：约一成题学方向、两成题选力度、其余测试。

## 3. 已发表的工作怎么做（事实）

五篇都在 2026-11-28 对照论文 PDF 原文核对过，引文逐字。两种做法都有，用在不同的场合。

**沿用、不再调：第二套题用来检验“换一套题还管不管用”**（原文叫 out-of-distribution generalization，分布外泛化）

| 论文 | 做了什么 | 原文 |
|---|---|---|
| ITI（Li 等，NeurIPS 2023）第 5.3 节 | TruthfulQA 上学的方向和超参数，直接用到 Natural Questions（搜索引擎里的真实问题）、TriviaQA（知识问答）、MMLU（多学科选择题） | “we apply ITI—using the activation shift directions and hyperparameters learned from TruthfulQA—to three different datasets”；“this is a true zero-shot evaluation ... as we do not tune any prompts, hyperparameters, or learn new truthful directions” |
| TruthX（Zhang 等，ACL 2024）第 4.3 节 | TruthfulQA 上训练的编辑模型，直接测 Natural Questions、TriviaQA、FACTOR（事实判断题） | “we directly use the TruthX model trained on TruthfulQA to assess its out-of-distribution generalization” |
| LoFiT（Yin 等，NeurIPS 2024）表 5 | 在 TruthfulQA 上训练之后不再训练，直接测 TriviaQA、Natural Questions、MMLU | “we first fine-tune Llama 2-7B-chat on TruthfulQA and then evaluate 0-shot on three out-of-domain question-answering benchmarks” |
| SEA（Qiu 等，NeurIPS 2024）第 4.6 节、表 5 | 用 BBQ（偏见测试题）的示例算出的编辑，直接测另一套偏见测试题 CrowS-Pairs | “the post-editing improvements can be generalised across new tasks, provided they share some similarities”；“We further assess all SEA methods using BBQ’s demonstrations on the CrowS-Pairs dataset” |

**每个任务各自学、各自选：几个任务都是主结果，而且任务本身不同**

| 论文 | 做了什么 | 原文 |
|---|---|---|
| LoFiT（NeurIPS 2024）表 6、附录 D | 三个主任务 TruthfulQA（真实性）、MQuAKE（知识更新后的多步推理）、CLUTRR（亲属关系推理），各有自己的训练、验证、测试划分，各自训练、各自搜超参数。作为对照的 ITI，力度也在每个数据集的验证集上调 | “we did not use the same set of hyperparameters throughout; instead, we tuned the following hyperparameters with grid search”；“We tuned α on a validation set of each dataset and we found that α = 15 worked robustly across all settings” |
| SEA（NeurIPS 2024）附录 H | 真实性：超参数在 TruthfulQA 上定。公平性：编辑用 BBQ 训练部分的示例算 | “All hyperparameters are determined with two-fold cross-validation on TruthfulQA”；“1000 pairs of demonstrations randomly sampled from our training split of BBQ to obtain the editing projections for fairness” |
| SADI（Wang 等，ICLR 2025）第 4.3 节 | 11 个任务上，最优超参数随任务变化很大；建议在每个任务自己的验证集上搜 | “Hyperparameters are task-specific. ... it is recommended to search optimal hyperparameters using data from the validation sets specific to each task” |

从原文能读出的几点（事实）：

- 沿用的四处，第二套题在原文里都是“泛化检验”，主结果仍在学方向的那套题上。ITI 表 4 的数：Natural Questions 46.6 → 51.3，TriviaQA 89.6 → 91.1，MMLU 35.71 → 40.16。
- 各自学的三处，每个任务都是主结果，任务考的不是同一种能力（真实性、公平性、推理、分类等）。
- LoFiT、SEA 各自在一篇里两种都用了。
- 一套题内部怎么分：ITI 用 5% 的题学方向、5% 的题选超参数（“Up to this point, we use 10% of TruthfulQA—81 questions in total”），正式结果用两折交叉验证（题分两半，轮流一半用来学和选、另一半测试）；TruthX 一半题做训练和验证（3:1），另一半测试；LoFiT 分成 326 / 82 / 407。

各模型力度怎么选的 18 篇逐篇记录在[这里](../novelty/prior_work/2026-11-20_steering_strength_per_model_survey.md)。

## 4. 我们自己的数（事实）

每个模型一行。方括号是按题对重抽的 95% 区间；加粗表示区间整个在 0 以上。计分：交出的回答答对 +1、答错 −1，不答或含糊 0。对手单 C 的力度在 FalseQA 的 100 道开发题上选，这对对手有利。

**4.1 沿用（做法 A），FalseQA 留出 600 题（读过的题）**

| 模型 | 方法 减 单 E | 方法 减 单 C |
|---|---|---|
| Llama-3.1-8B | **+106 [+87, +125]** | **+77 [+33, +122]** |
| Qwen3.5-9B | **+23 [+13, +34]** | +4 [−30, +39] |
| Qwen3.8-27B | **+123 [+97, +148]** | **+139 [+106, +173]** |
| Gemma-3-12B | **+84 [+67, +101]** | **+96 [+54, +138]** |
| Gemma-3-27B | **+32 [+15, +51]** | **+121 [+91, +151]** |

**4.2 沿用（做法 A），FalseQA 没用过的题（确认实验，先判的 100 道）**

方法和各模型的力度全部冻结之后，在 FalseQA 官方测试集里从没用过的 386 对题上新生成、新判分（[排期 Q](../discussion/work_queue/2026-11-28_queue_q.md)）。下表是先判的 100 道（50 对）；其余的题正在判，判完后这张表换成全部 772 道的数。100 道只看方向，每 100 道几分的差别分不出，所以这张表不加粗。

| 模型 | 方法 减 单 E | 方法 减 单 C | 方法净分 | 单 E 净分 | 单 C 净分 | 原模型净分 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | +15 [+7, +22] | +3 [−13, +19] | +38 | +23 | +35 | +25 |
| Qwen3.5-9B | +5 [+1, +10] | +5 [−9, +19] | +57 | +52 | +52 | +53 |
| Qwen3.8-27B | +20 [+10, +29] | +19 [+4, +33] | +50 | +30 | +31 | +30 |
| Gemma-3-12B | +14 [+8, +20] | +24 [+8, +39] | +50 | +36 | +26 | +26 |
| Gemma-3-27B | +5 [−3, +13] | +9 [−4, +23] | +43 | +38 | +34 | +35 |

- 比单 E、比单 C 的十个差值里，为正的有 10 个。
- 判分设置与留出 600 题那一轮相同（同一个裁判模型、同一份判分约定）；读表程序先在留出 600 题上重算，与存档逐项相同才读新题；另有一个不共用代码的重算程序，结果逐项相同。

**4.3 只把 C 的力度改成在 FalseQA 自己的 100 道开发题上选（做法 B 里能离线算的那一半），留出 600 题**

| 模型 | 沿用的 C 力度 | FalseQA 自选的 C 力度 | 自选时：方法 减 单 E | 方法 减 单 C |
|---|---|---|---|---|
| Llama-3.1-8B | 2 | 2 | **+106 [+87, +125]** | **+77 [+33, +122]** |
| Qwen3.5-9B | 0.5 | 0.5 | **+23 [+13, +34]** | +4 [−30, +39] |
| Qwen3.8-27B | 1 | 1 | **+123 [+97, +148]** | **+139 [+106, +173]** |
| Gemma-3-12B | 2 | 1 | **+59 [+44, +74]** | **+71 [+28, +114]** |
| Gemma-3-27B | 2 | 0.5 | **+23 [+12, +34]** | **+112 [+85, +139]** |

- 三个模型两种选法选出同一档，数字与 4.1 相同。另两个模型选出的档不同，区间仍整个在 0 以上。

**4.4 沿用过来的 E，在 FalseQA 上做了什么（只拧 E 减 原模型，留出 600 题，每对一道前提错、一道前提正常）**

| 模型 | 前提错的 300 道 | 前提正常的 300 道 |
|---|---|---|
| Llama-3.1-8B | +101 | −66 |
| Qwen3.5-9B | +45 | −43 |
| Qwen3.8-27B | +63 | −37 |
| Gemma-3-12B | +118 | −101 |
| Gemma-3-27B | +103 | −15 |

- 沿用过来的 E，五个模型在前提错的题上都加分，在前提正常的题上都扣分。

**4.5 没有做过的**

- E 的力度在 FalseQA 上重选（做法 B 的另一半）；E、C 两个方向在 FalseQA 上重学（做法 C）。所以“各自定之后数会不会更好”，现在没有数据。
- 按工作记录，此前在别的题集上也都是沿用，没有查到在别的题集上重学方向或重选力度的记录。

## 5. 我的看法（看法，不是事实）

建议正文用做法 A（沿用），并加一句 4.3 的结果作稳健性说明。理由：

1. 两套题考的是同一种能力：问题带误导时如实回答。这和第 3 节“沿用”那一类场合对得上。这一条是判断，博士可能有不同的看法：TruthfulQA 是常见误解，FalseQA 是前提错误，也可以看成两种题型。
2. 沿用是更严的考法。FalseQA 的题一道都没有用来学方向或选方法的力度，方法的数不会因为调参而偏高。
3. E 的力度本来就被能力线（MMLU 掉分）卡着，这条线不随题集变。
4. 做法 C 要按天算，并且带出一个新问题：两套题上学出的 E 是不是同一个方向。

做法 A 的代价：沿用的 E 在 FalseQA 前提正常的题上扣分（4.4）。在 FalseQA 上重选 E 的力度（做法 B）也许能减小这项代价，没有试过。

## 6. 请博士裁定

1. A、B、C 选哪一种？
2. 如果选 A：FalseQA 在论文里摆成“第二套主结果”，还是摆成“换一套题的泛化检验”？第 3 节里沿用的四处都是后一种摆法。
3. 如果选 B 或 C：FalseQA 的题怎么分？照 TruthfulQA 的比例（约 12% 学方向、23% 调参、65% 测试），还是别的分法？

结果文件：[`routes_1128_20261128.json`](../results/write_positions/routes_1128_20261128.json)（4.1、4.3、4.4），[`out_confirm_1128_q100_20261128.txt`](../results/write_positions/out_confirm_1128_q100_20261128.txt)（4.2）。

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-28 | 初稿：问题、三种做法、五篇论文的原文、我们的数、看法、请博士裁定的三问；确认实验先放 100 道的数 | 赵磊（Fable 5.2 代笔） |

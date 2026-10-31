# 审核点 1：E 方向用什么方法估。DAS 与均值差的来龙去脉

> 2026-10-31。给 GPT 看。先讲我们做了什么、看到了什么、哪里可疑，再列这次新写的代码和要你审的问题。相关结果在 `results/01_E估计器/`，代码在 `code/`。

## 0. 三句话总结

1. 原方案把 rank-1 DAS（Distributed Alignment Search · 分布式对齐搜索 · 用训练的办法找一个方向，把句子在这个方向上的坐标从"真"换成"假"，看模型输出会不会跟着翻）定为估 E 方向的主方法，均值差和线性探针只作对照。
2. 我们按方案做了。五个模型、三轮越来越公平的对照，DAS 没有一次显著赢过均值差，于是主方法换成均值差，DAS 退为对照。
3. 现在回头审这个决定，有四处站不稳：三个主模型之一 Gemma-3-12B 从没比过 DAS，而它恰好是 E 推不动的模型；我们的 DAS 比方案规格缩水（单种子、无截尾、无平衡加权）；最终读出口径下 DAS 一次没比过；我们此前引用 CDAS 论文来支持"DAS 推不动"，引过头了。另外，"DAS 在深层换分量近乎完美、推了却没用"这个现象需要一个能验证的解释。

## 1. 背景：E、C、2×2 与读出方式

- 模型每一层算完后传给下一层的向量叫残差流（residual stream）。我们取提示词最后一个 token 位置的残差流 h（Llama 是 4096 维）。
- **E 方向 wE**：真陈述和假陈述在某层的 h 的差异方向。**C 方向 wC**：同一句话配"自信"系统提示和配"谨慎"系统提示时 h 的差异方向。
- **推法**。E 用放大式 `h' = h + α · w wᵀ (h − c)`，c 是真假两类均值的中点：只放大 h 在 w 上相对中点的偏离，w 的正负号无关。C 用加法 `h' = h ± α · gap · w`。放大式的来由：加法式推 E 会把所有句子一起推向"真"，判别力不涨、偏置涨，违反 2×2 对 E 行的要求。
- **读出**。两个独立上下文各问一次。问法 A（问题级）："你知道这道题的答案吗？只答 Yes 或 No"，Yes 对 No 的概率给**作答率**。问法 B：强制判每句话 True / False，给**判别力 J = TPR − FPR**、AUROC、**偏置 B = 判 True 的比例**。推方向时作用在提示词末 token 位置。
- **2×2 主张**：推 E → J 升、作答率不变、B 不变；推 C → 作答率变、J 和 B 不变。
- **数据与切分**：TruthfulQA 817 道题，每题一句真话一句假话。96 对估方向，192 对调参（选层、选强度），529 对测试一次。
- 本审核点只关心 **E 行**：用哪种方法估出来的 wE，推了之后 J 和 AUROC 涨得多、作答率和 B 不动。

## 2. 三种估计器是什么

| 估计器 | 做法 | 我们的实现 |
|---|---|---|
| 均值差（mean difference, MD） | 真句 h 的平均值减假句 h 的平均值，归一化 | `core/directions.py: mean_diff_direction`；96 对 192 句 |
| 线性探针（logistic-regression probe, LR） | 训一个分真假的逻辑回归，取权重方向（标准化后再除回标准差） | `scripts/e23_knob_sweep.py --e-estimator probe` |
| rank-1 DAS | 见下 | `core/directions.py: train_das` |

**DAS 在做什么（人话）**。拿一对句子：假句（base）和同一题的真句（source）。在第 l 层，把假句的 h 在方向 w 上的那**一个坐标**换成真句在同一方向上的坐标，其余 4095 个坐标不动，然后要求模型对这个"换了一个坐标的假句"的回答，跟它读真句时的回答一样。训练的目标就是让这两个回答分布的差距最小。公式：

```
z_base = h_base · w ,  z_src = h_src · w
h'_base = h_base + (z_src − z_base) · w
L = (1/T) Σ_{t<T} KL( p_src(· | y_<t) ‖ p_int(· | y_<t) )
```

我们的实现细节：rank 1；T = 8，teacher forcing 用的是 source 自己贪心生成的 8 个 token；前向 KL；Adam，学习率 1e-3，10 个 epoch，梯度裁剪 1.0；只换末 token 位置；E 的对子 base = 假句、source = 真句，96 对；单种子 42。文件 `das_E_L<l>_r1_s42.pt`，训练后另算 `mean_gap`（训练对上 z_src − z_base 的平均值）。

**推的时候三种方向用的是同一个算子**：同一个放大式、同一个中点 c（估计集真假两类均值的中点）、同一批层、同一组强度 α。所以后面的差别不是推法造成的。

## 3. 我们做了什么、看到了什么

### 3.1 时间线

1. **10-25 / 26**：按方案，DAS 只在 LA-Score（方案里免训练的选层打分）挑出的前 3 层训（TruthfulQA 上是第 18、17、13 层）。Llama 上比均值差弱。有人指出层可能选错，于是补 12 到 16 层：DAS 12–14 层联合推 AUROC +0.040，同配置均值差 +0.076。
2. **10-27**：跨模型对照。Mistral-7B、Qwen3-4B、Qwen3-1.7B、Qwen3-8B，同层同网格，DAS 没有一格赢均值差。
3. **10-27 / 28，公平协议**：DAS 每一层都训（1 到 30 层）；训练数据与均值差同一份 96 对；候选在测试集的 A 半挑、在 B 半报，再对调取平均；三种指标各选一次。另做了"一半一半"：408 对估方向、另一半测。
4. **10-30，新读出协议**：换到两上下文读出后，在 Llama、Qwen 上再比一次（问法 A 当时还用"sure"措辞，后来改成了问题级问法，见 §5c）。

### 3.2 结果

**表 1：旧协议（单上下文强制判断），推 E 后 AUROC 的增益。** 文件：`results/01_E估计器/llama3.1-8b/select_layer_auroc.md`、`select_layer_acc.md`，其他模型 `multilayer_{das,meandiff}.md`。

| 模型 | 均值差 | DAS | 说明 |
|---|---|---|---|
| Llama-3.1-8B，全层公平对照，AUROC | L13 α5 **+0.043** [+0.036, +0.049] | L12 α5 +0.030 [+0.018, +0.039] | 差 −0.014 [−0.026, −0.001]，均值差显著赢 |
| Llama-3.1-8B，同上，全题正确率 | +0.039 | +0.051 | 差 +0.005 [−0.029, +0.033]，平手 |
| Qwen3-8B | +0.047 | +0.016 | |
| Mistral-7B | +0.011 | +0.001 | |
| Qwen3-4B | +0.024 | −0.012 | 推 DAS α=3 时 AUROC −0.135 |
| Qwen3-1.7B | +0.074 | +0.005 | DAS 方向是偏置杆，B → 0.998 |
| Gemma-3-12B | 零效果 | **没跑** | 见 §5a |

**表 2：新读出协议（E22，问法 A 用"sure"措辞），α=3，变化量相对基线。** 文件：`two_stage_sure.md`（均值差）、`two_stage_das.md`（DAS）。

| 模型 | 方向 | ΔJ | ΔAUROC | ΔB | Δ作答率 |
|---|---|---|---|---|---|
| Llama | 均值差 | **+0.136** [+0.093, +0.173] | **+0.077** [+0.061, +0.092] | +0.058 | +0.114 |
| Llama | DAS | +0.075 [+0.029, +0.115] | +0.036 [+0.021, +0.049] | +0.103 | +0.157 |
| Qwen3-8B | 均值差 | **+0.122** [+0.081, +0.167] | +0.015 [−0.001, +0.031] | +0.022 | −0.154 |
| Qwen3-8B | DAS | +0.072 [+0.033, +0.112] | −0.021 [−0.036, −0.004] | +0.054 | −0.173 |

读法：DAS 判别力涨得少、偏置漂得多、作答率漏得也不少。两家作答率都在动，是当时"sure"措辞的极性伪影（"你确定吗"被模型当成"你觉得它是真的吗"来答），换成问题级问法后均值差 E 的作答率变化降到 +0.007（不显著）。DAS 在新问法下没有重测。

**Gemma-3-12B（最终协议，`gemma3-12b/knob_sweep_all.md`、`knob_sweep_probe.md`）**：均值差 E 和探针 E 都是零效果。但探针能从 Gemma 中层读出真假的 AUROC 是 0.950，模型嘴上只有 0.856，说明知识在里面。

### 3.3 逐层现象（Llama，`multilayer_das_all.md` / `multilayer_meandiff_all.md`）

- DAS 只在第 9 到 13 层推了有效，峰在 12。**从第 14 层起 DAS 的训练损失已降到 0.03**，也就是换一个坐标就能让模型对假句的回答和读真句时几乎一样，**但拿同一个方向去放大，判别力一点不涨**。
- 均值差在 13 到 16 层一段都有效，峰在 13。

## 4. 我们的解释，以及怎么验证

"换"和"推"考的是两种不同的东西。换分量考的是：这个方向里是否装着模型的最终判断。推考的是：放大这个方向，判断能否变得更正确。深层的 w 装的是**嘴**，也就是模型已经做出的决定；把决定换掉，嘴当然跟着换，所以损失极低；可把一个已经做出的决定放大，错的还是错的，所以推了没用。能让判断更正确的方向必须装着**嘴还没用上的证据**，那在更早的层。

这个解释有一个不花 GPU 的检验，照 CDAS（ICLR 2026）附录 J 的做法：把方向 w 直接过模型的最终归一化层和输出层，`Δlogits = W_U · Norm(w)`，看它最想吐出哪些词。**预测：Llama 第 14 层以后的 DAS 方向会直接吐出 True / False 这类标签词（嘴），第 9 到 13 层的不会；均值差方向逐层看也应有同样的分界。** 这就是 `scripts/diag_direction_unembed.py`。

## 5. 四处可疑

- **(a) Gemma 没比过 DAS。** 三个主模型里 E 唯一不成立的是 Gemma-3-12B，它恰好是唯一没比过 DAS 的。而且 Gemma 是"知识在、方向推不动"的情形（探针 0.950 对嘴 0.856），正是 DAS 这类"训练出能改输出的方向"的方法该发挥作用的场景。
- **(b) 我们的 DAS 比方案规格缩水。** 方案要求三个随机种子（42、43、44）、对归一化后的激活差取 10% 截尾均值、按类别和长度平衡加权。我们单种子、无截尾、无平衡加权。"DAS 输了"对 DAS 可能不公平。
- **(c) 最终口径下 DAS 一次没比过。** 表 2 是问法 A 还用"sure"措辞时跑的。换成问题级问法、全层扫描、192 对调参这套最终口径后，三个模型都没再比过 DAS。
- **(d) 引用过头。** 我们此前写"CDAS（ICLR 2026）证明原版 DAS 目标做 steering 几乎无效"。精读后发现：那篇论文批的"DAS"是"交换干预 + 交叉熵拟合固定标签词"，而我们的 DAS 是"交换干预 + KL 逼近 source 的输出分布"，其实更接近它改良后的 CDAS；并且该论文没有任何真假判断任务。所以"DAS 不如均值差"只能靠我们自己的数据撑。

## 6. CDAS 与我们的 DAS 差四处

| | 我们的 DAS | CDAS | 本次实现为可选项 |
|---|---|---|---|
| 散度 | 前向 KL | 全词表 JSD（Jensen–Shannon divergence · 对称版 KL，两边都不许塌掉）；他们消融 JSD 0.860 > 反向 KL 0.840 ≫ 前向 KL 0.560 | `--das-loss jsd` |
| 方向性 | 只训 base ← source 一向 | D⁺ 和 D⁻ 两向同训，同一个 w 既要能"变真"也要能"变假" | `--das-bidir` |
| 位置 | source 取末 token，只换 base 末 token | source 取回复前模板 token，钳到 base **所有**位置 | `--das-positions all`（source 仍取末 token，钳到 base 所有位置） |
| 推理 | 放大式 | 钳到概念类平均投影 a = β·ᾱ | **不用于 E**：对真/假判别任务，钳到真类均值会抹掉信息、变成偏置杆 |

另外 CDAS 的一条数据支持我们的做法：同一目标同一数据只换层，Gemma-2-9B 第 20 层 0.992、第 31 层 0.518，**训练损失不是推得动的代理，选层只能看推的效果**。

## 7. 这次新写的代码（请重点审）

全部默认值不改变旧行为，旧文件名不变。

### 7.1 `code/core/directions.py`

- `_interchange_hook_factory(W_fn, src_act_holder, positions)`：新增 `positions="all"`，把 source 末 token 的投影 z_src 钳到 base 所有位置：`delta = (z_src[:, None] − h @ W) @ Wᵀ`。对子是逐条编码的，没有 padding。
- `_divergence(int_logits, src_logits, kind)`：`kl` 为原来的前向 KL；`jsd` 用对数空间算混合分布 `logm = logsumexp([logp_int, logp_src]) − log 2`，再 `0.5·KL(p_int‖m) + 0.5·KL(p_src‖m)`。依赖 `F.kl_div(input=log q, target=log p, log_target=True) == KL(p‖q)`。
- `multistep_kl_loss(..., loss, positions, base_key, source_key)`：名字没改；`base_key / source_key` 让调用方交换两句的角色。
- `train_das(..., loss, bidirectional, positions, pair_weights, trim)`：双向时 D⁺ 用 source 的贪心 token 做 teacher forcing，D⁻ 用 base 自己的贪心 token，各乘 0.5 分别 backward 再一起 step（峰值显存只有一项）；`pair_weights` 逐对乘到损失上；`trim` 只作用于 `mean_gap`（DAS 里唯一的"激活差均值"）。
- 新增 `length_bins`、`balanced_pair_weights`（按 (类别, 句长三分位) 逆频率加权、均值归一到 1）、`das_file_name`、`load_das_direction`（同层同变体的所有种子，先按与参考方向的点积对齐符号，再平均、归一化；`mean_gap` 随符号一起翻）。

### 7.2 `code/scripts/e1_directions.py`

新参数 `--das-seeds 42 43 44`、`--das-loss`、`--das-bidir`、`--das-positions`、`--das-balance`、`--das-trim 0.1`、`--das-variant`、`--trim`。变体文件名加后缀，例如 `das_E_L12_r1_s43_jsd-bidir-allpos.pt`；默认变体文件名与旧版完全一致。平衡加权的"类别"取记录里 `meta.category`（TruthfulQA 自带的 38 个类别），"长度"取真假两句 token 数之和的三分位。多种子时记录种子间 |cos|。

### 7.3 `code/scripts/e23_knob_sweep.py`

`--e-estimator das --das-dir <目录> [--das-variant <后缀>]`：每层读入该层所有种子的 DAS 方向，与均值差方向对齐符号后平均，其余流程与均值差完全相同（同中点、同算子、同规则选层）。某层没有 DAS 文件则该层整体丢弃并打印。`--plan-split` 跑原方案的 32 / 64 / 其余切分。

### 7.4 `code/scripts/diag_direction_unembed.py`（新）

对每层的均值差 E、均值差 C 和每个 DAS 方向文件，算 `W_U · Norm(±w)` 的前 10 个 token、True / False / Yes / No / Unknown 的名次与标准化 logit、`cos(w, u_True − u_False)`。某方向在任一符号下有 True 或 False 进前 20 名，就标为"嘴"。

### 7.5 我自己看到的风险点，请你判断

1. `positions="all"` 把陈述本身的 token 也钳了，这是不是 CDAS "one-to-many" 的正确读法？CDAS 的 source 值取自回复前的模板 token，我们取自末 token。
2. JSD 的 `kl_div` 参数顺序和 `log_target` 用法。
3. 双向训练两次 backward 再 step，与一次 backward 等价吗？`y_base` 用 base 的贪心 token 是不是 D⁻ 的正确 teacher。
4. 平衡加权归一到均值 1，再配 Adam，权重尺度是否还有意义。
5. "截尾均值"只作用在 `mean_gap` 上，是否符合"对归一化后的激活差取 10% 截尾均值"的本意？（方案里这句对 DAS 的含义本身不清楚。）
6. 多种子取平均，还是按调参集效果挑一个种子？哪个更站得住。
7. e23 里 DAS 方向配的中点 c 仍是估计集真假两类均值的中点。DAS 自己没有中点，这样配合理吗？
8. 诊断脚本忽略了 Gemma 的 logit softcapping（单调，不改名次）；"前 20 名算嘴"是拍的。

## 8. 运行计划与成本

0. **诊断（几分钟）**：Llama、Qwen 现有的全层 DAS 方向与均值差方向过输出层。
   `python3 scripts/diag_direction_unembed.py --model llama3.1-8b --e1-dir results_tqa/E1/llama3.1-8b --das-dir results_das/E1/llama3.1-8b`
1. **按方案规格重训 DAS**，三个模型全层：
   `python3 scripts/e1_directions.py --exp E1_directions_tqa --model <m> --das-tags E --das-pairs 96 --das-layers <1..L−2> --das-seeds 42 43 44 --das-balance --das-trim 0.1`
   估算：Llama 30 层 × 3 种子约 3 到 4 小时单卡，Gemma-12B 46 层 × 3 种子约 9 小时单卡，四卡分层并行。
2. **最终口径下重比**：`e23_knob_sweep.py --tune-pairs 192 --e-estimator das --das-dir ... [--das-variant bal-trim0.1]`，与均值差、探针各一轮，`e23_merge.py` 合并，同一规则选层。
3. **CDAS 式变体**（`--das-loss jsd --das-bidir --das-positions all`）先在 Llama 单种子跑，看方向和效果有没有变化，再决定要不要扩到三个模型。
4. **泛化测试**（下一个审核点）：各种 E 都搬到 HaluEval、BBQ、知识集，看谁搬得动。均值差 E 已知搬不动；DAS E 没测过。

## 9. 请你回答

1. §7 的代码有没有会导致错误结论的 bug？按 §7.5 逐条判断，并说"可以跑 / 改了 X 再跑 / 不能跑"。
2. §3 的三轮对照，哪一轮对 DAS 不公平？如果你是审稿人，会要求补哪个对照才认"均值差优于 DAS"？
3. §4 的解释和诊断，逻辑上成立吗？诊断结果出来后，什么样的图样算支持、什么样算推翻？
4. "DAS 在某些模型上略逊于均值差没关系，只要更能泛化、效果也可以，就比均值差强"：这个标准怎么定义成可检验的？§8 第 4 步够不够？
5. 如果 Gemma 上 DAS 也推不动，还有哪些估计器或训练目标值得试（给具体做法）？如果 Gemma 上 DAS 推得动而均值差不行，主方法该怎么定、怎么写？
6. 多重比较：全层 × 多强度 × 多估计器在 192 对上选最大值，报到 529 对上，现在的"选层误差 + 测试集 oracle"够不够？

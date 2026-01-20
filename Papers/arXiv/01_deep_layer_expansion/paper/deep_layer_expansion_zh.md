# 深层膨胀效应：专家提示词抵抗大语言模型中的维度坍缩

**作者：** 赵磊¹, 靳岩岩
¹腾讯

---

## 摘要

大语言模型（LLM）在提示词包含专家级领域信号时表现出系统性的性能提升。本文通过在两个 70B 级别的主流开源模型（Qwen2.5-72B-Instruct 和 Llama-3.3-70B-Instruct）上进行控制实验，研究这一现象背后的几何机制。与传统认知——深层表示向确定性输出压缩——相反，我们发现了一个显著的普遍现象：**专家信号在表示空间中诱导"深层膨胀"**。具体而言，专家级提示词使深层（Layer 60+）的有效内在维度（EID）相比标准提示词提升 60-100%。我们将此形式化为**流形传送**：专家信号充当高维导航器，抵抗模型在推理过程中趋向维度坍缩的倾向，将激活轨迹维持在语义密度更高的流形区域。我们的发现为提示词工程提供了几何理论基础，并为 LLM 可解释性研究提供了新的量化工具——通过追踪 EID 轨迹来理解 prompt 如何影响模型内部计算。

**关键词：** 大语言模型，内在维度，提示词工程，表示几何，可解释性

---

## 1. 引言

大语言模型（LLM）在自然语言处理领域取得了显著突破，展现出强大的文本生成与推理能力。提示词工程（Prompt Engineering）作为一种无需修改模型参数即可提升性能的技术，已成为实践中的核心方法。然而，一个广泛观察到的现象仍缺乏理论解释：对于相同的问题，专家风格的提问往往获得更详细、更深入的响应——即使查询的核心语义完全相同。

先前关于神经网络表示的研究表明，LLM 的处理过程遵循"膨胀-压缩"模式：中间层进行特征提取（维度增加），深层进行语义压缩以促进输出生成（维度降低）(Ansuini et al., 2019)。近期研究进一步发现，Transformer 中间层存在高内在维度峰值，对应语言抽象的关键转折点 (Cai et al., 2024)。

本文研究**上下文依赖性能分化（CDPB）**——语义等价的查询仅基于上下文框架产生质量截然不同的响应。我们的核心问题是：**专家信号如何在表示空间的几何层面影响模型行为？**

通过在 **Qwen2.5-72B-Instruct** 和 **Llama-3.3-70B-Instruct** 上的实验，我们发现了一个惊人一致的几何现象：专家信号成功抑制深层维度坍缩，迫使模型在输出层附近维持高内在维度。我们称这种现象为**流形传送**（Manifold Teleportation），因为专家信号有效地将模型的激活状态"传送"并锁定到语义密度丰富的高维子流形中。

### 贡献

1. **识别并量化深层膨胀效应**：专家提示词使深层 EID 跨架构提升 60-100%，这与传统的"深层压缩"认知形成对比。
2. **验证跨架构普遍性**：在 Qwen 和 Llama 两个不同训练血统的模型家族中观察到一致的几何行为。
3. **提出可解释性新视角**：EID 轨迹可作为理解 prompt 效果的量化工具，为 LLM 可解释性研究提供新方向。
4. **发布实验代码和数据**：支持研究复现。

---

## 2. 相关工作

### 2.1 提示词工程与上下文学习

提示词构建方式显著影响 LLM 输出质量 (Reynolds & McDonell, 2021; Liu et al., 2023)。角色扮演提示 (Shanahan et al., 2023) 表明上下文框架调节模型行为。我们的工作通过提供基于表示几何的解释扩展了这一领域。

### 2.2 神经网络表示的内在维度

Ansuini et al. (2019) 发现深度网络表示位于低维流形上，且层间 ID 遵循"先增后减"模式。Cai et al. (2024) 在 ICLR 2024 发表的工作识别出 Transformer 中间层的高 ID 峰值，并证明这与模型性能和迁移能力正相关。Valeriani et al. (2023) 研究了大型 Transformer 隐藏表示的几何结构。

**我们的工作与上述研究的区别：** 先前工作研究的是模型固有的 ID 模式（给定输入的层间变化），我们研究的是**同一模型在不同 prompt 条件下 ID 模式的差异**——这是可控的、可操作的。

### 2.3 LLM 可解释性

机械可解释性（Mechanistic Interpretability）致力于理解 LLM 的内部计算机制 (Elhage et al., 2022)。稀疏自编码器（SAE）被用于解决"叠加"问题，将多义神经元分解为单义特征。表示工程（Representation Engineering）通过分析和操控激活空间来理解和引导模型行为 (Zou et al., 2023)。

我们的工作提供了一种**互补视角**：不是分析单个特征或电路，而是追踪整体表示空间的几何性质（EID）如何响应 prompt 变化。

---

## 3. 方法论

### 3.1 有效内在维度（EID）

为量化模型在每一层的"认知复杂度"，我们采用基于隐藏状态矩阵谱熵的有效内在维度。

**定义。** 给定某一层在 $N$ 个样本上的隐藏状态矩阵 $H \in \mathbb{R}^{N \times d}$，设 $\{\sigma_i\}$ 表示 SVD 分解的奇异值。定义归一化奇异值为：

$$\hat{\sigma}_i = \frac{\sigma_i}{\sum_j \sigma_j}$$

有效内在维度定义为：

$$\text{EID}(H) = \exp\left( -\sum_{i} \hat{\sigma}_i \log \hat{\sigma}_i \right)$$

**直觉解释：**

| EID 值 | 含义 | 类比 |
|--------|------|------|
| 低（~3） | 激活集中在少数方向 | 模型"锁定"了答案 |
| 高（~30+） | 激活分散在多个方向 | 模型在"探索"多种可能性 |

这个指标捕获了表示中的有效自由度——模型在每一层实际使用了多少独立的计算"方向"。

### 3.2 实验设计

**模型选择。** 我们选择了两个 70B 级别的主流开源模型：
1. **Qwen2.5-72B-Instruct**（阿里云）- AWQ INT4 量化，80 层
2. **Llama-3.3-70B-Instruct**（Meta AI）- W8A8 INT8 量化，80 层

选择这两个模型的原因：(1) 参数规模相近，便于对比；(2) 来自不同训练血统（中国 vs 美国），可验证普遍性；(3) 均为指令微调版本，代表实际应用场景。

**数据集。** 我们构建了 50 个技术主题，涵盖分布式系统、编程语言、数据库、网络和机器学习（完整列表见附录 A）。

**提示词条件。** 对于每个主题，我们设计了两个对照提示词：

| 条件 | 模板 |
|------|------|
| **标准（基线）** | "请解释一下 {topic}。" |
| **专家（处理组）** | "作为该领域的资深专家，请从底层原理和数学推导的角度深度剖析 {topic}。请展示你的思维链。" |

**测量协议。** 对于每个提示词，我们：
1. 通过模型处理提示词
2. 在最后一个 token 位置从所有 80 层提取隐藏状态
3. 使用谱熵方法计算每一层的 EID
4. 在所有 50 个主题上取平均以获得平滑轨迹

---

## 4. 结果

### 4.1 Qwen2.5-72B：深层膨胀效应

图 1 展示了 Qwen2.5-72B 在两种提示词条件下的 EID 轨迹。

![图 1：Qwen2.5-72B 的流形几何](Figure_1_Qwen72B_Manifold.png)

**图 1：Qwen2.5-72B 的 EID 轨迹对比。** 横轴为 Transformer 层编号（0-80），纵轴为有效内在维度（EID）。红色曲线（Expert）为专家提示词条件，蓝色曲线（Standard）为标准提示词条件。专家轨迹从第 30 层开始与基线明显分叉，在深层（Layer 70）显示 +60% 的维度膨胀。

| 层 | 标准 EID | 专家 EID | 差异 |
|----|---------|---------|------|
| 40 | ~7 | ~10 | +43% |
| 60 | ~14 | ~22 | +57% |
| 70 | ~23 | ~37 | **+60%** |
| 75 | ~28 | ~45 | +61% |

**关键观察：**
1. **入口层（0-5）：** 两种条件都显示高维度（~30-50），这是嵌入层表示的特性。
2. **压缩区（5-20）：** 维度急剧下降到 ~2-3，模型在"解析"输入。
3. **分叉区（20-75）：** 专家 EID 持续高于标准，差距逐层扩大。
4. **输出准备（75-80）：** 两条轨迹都上升，但专家达到 ~50，标准为 ~35。

轨迹形成独特的**"喇叭口"拓扑**——不是预期的对称"沙漏"（中间膨胀、两端压缩），而是在专家提示下观察到持续的深层膨胀。

### 4.2 Llama-3.3-70B：跨架构验证

为排除模型特定偏差，我们在 Llama-3.3-70B 上复制了实验。

![图 2：Llama-3.3-70B 的流形几何](Figure_1_Llama70B_Manifold.png)

**图 2：Llama-3.3-70B 的 EID 轨迹对比。** 与 Qwen 一致，专家提示词在深层诱导持续的高维状态。Layer 60 处的膨胀效应达到 +102%，比 Qwen 更为显著，可能与 INT8 量化保留更高精度有关。

| 层 | 标准 EID | 专家 EID | 差异 |
|----|---------|---------|------|
| 40 | 6.6 | 12.4 | **+89%** |
| 60 | 13.2 | 26.8 | **+102%** |
| 70 | 16.8 | 33.3 | **+99%** |

### 4.3 跨架构一致性

| 模型 | 量化方式 | Layer 60 Δ | Layer 70 Δ | 分叉层 |
|------|---------|------------|------------|--------|
| Qwen2.5-72B | AWQ INT4 | +57% | +60% | ~30 |
| Llama-3.3-70B | W8A8 INT8 | +102% | +99% | ~25 |

尽管训练流程、架构和量化方案不同，两个模型都表现出：
1. **轨迹分叉**从中间层开始（Layer 25-30）
2. **渐进发散**，专家 EID 持续更高
3. **深层膨胀** 60-100%（在 Layer 60-70）

这种高度的跨架构一致性强烈表明，**深层膨胀是 Transformer LLM 响应高质量语义信号的普遍几何特性。**

---

## 5. 讨论

### 5.1 流形传送假说

我们提出专家信号作为**流形导航器**，将激活轨迹引导到高维语义区域：

1. **标准提示词**将模型定位在"通用响应"流形区域——RLHF 训练形成的低维吸引子盆地，偏好安全、平均的响应。

2. **专家提示词**通过信号词（"资深专家"、"底层原理"、"数学推导"）注入高频语义信息，触发向高维专业区域的导航。

3. **深层累积**效应：小的初始轨迹差异通过连续层累积，导致深层几何的显著差异。

**类比：** 两列火车从同一车站出发，方向偏差 1°。初始差距很小，但 1000 公里后，目的地相差数十公里。

### 5.2 与已有工作的关系

Cai et al. (2024) 发现 LLM 中间层存在 ID 峰值，对应语言抽象的关键转折点。我们的发现与之互补：

| Cai et al. (2024) | 本文 |
|-------------------|------|
| 研究固有的层间 ID 模式 | 研究 prompt 如何改变 ID 模式 |
| 描述性发现 | 可控、可操作的干预 |
| 预测模型性能 | 评估 prompt 质量 |

### 5.3 可解释性视角

我们的发现为 LLM 可解释性提供了新工具：

1. **Prompt 效果的量化评估**：无需查看输出，通过 EID 轨迹即可判断 prompt 是否"激活"了模型的深层计算能力。

2. **黑盒到灰盒**：虽然我们不知道模型具体在"想"什么，但 EID 告诉我们模型用了多少计算"带宽"在想。

3. **调试工具**：如果 prompt 没有产生预期的深层膨胀，说明信号不够强或方向不对。

### 5.4 局限性

1. **量化伪影**：两个模型都使用量化权重；全精度验证是未来工作。
2. **首 token 测量**：我们在提示词处理完成时测量 EID；生成过程中的动态尚未探索。
3. **相关性 vs 因果性**：我们展示了专家信号与 EID 膨胀的相关性，但因果机制需要进一步研究（如注意力模式分析）。
4. **下游任务验证**：我们尚未验证更高 EID 是否直接导致更好的任务性能。

---

## 6. 结论

本文提供了实证证据表明专家级提示词在 LLM 表示中诱导**深层膨胀效应**——深层有效内在维度提升 60-100%。这种现象在 Qwen 和 Llama 架构中普遍成立，表明它是 Transformer 模型响应高质量语义信号的基本几何特性。

我们提出**流形传送**框架：专家信号充当导航器，将激活轨迹从低维坍缩区域引导到高维语义流形。这为理解提示词工程为什么有效提供了几何基础，并为 LLM 可解释性研究开辟了新方向——通过追踪表示空间的几何性质来理解模型行为。

**未来工作：**
1. 因果机制分析：通过注意力模式和电路分析理解深层膨胀的成因
2. 下游验证：建立 EID 与任务性能的定量关系
3. 自动化工具：基于 EID 的 prompt 质量自动评估
4. 更多模型：在 Mistral、Gemma 等其他架构上验证

---

## 参考文献

Ansuini, A., Laio, A., Macke, J. H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. *NeurIPS*.

Cai, T., et al. (2024). Emergence of a High-Dimensional Abstraction Phase in Language Transformers. *ICLR 2024*.

Elhage, N., et al. (2022). Toy models of superposition. *Transformer Circuits Thread*.

Kirsanov, D., et al. (2025). The Geometry of Prompting: Unveiling Distinct Mechanisms of Task Adaptation in Language Models. *arXiv preprint*.

Liu, P., et al. (2023). Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. *ACM Computing Surveys*.

Marks, S., & Tegmark, M. (2023). The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations. *NeurIPS 2023*.

Reynolds, L., & McDonell, K. (2021). Prompt programming for large language models: Beyond the few-shot paradigm. *CHI EA '21*.

Shanahan, M., McDonell, K., & Reynolds, L. (2023). Role play with large language models. *Nature*.

Valeriani, L., et al. (2023). The geometry of hidden representations of large transformer models. *NeurIPS 2023*.

Wang, X., et al. (2024). The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based Models. *EACL 2024 Findings*.

Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.

Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv preprint*.

---

## 附录 A：技术主题（50 个）

1. Raft 共识算法中的 Leader 选举机制
2. Transformer 架构中位置编码的频域特性
3. 操作系统中的写时复制 (Copy-on-Write) 机制
4. 数据库事务隔离级别与幻读问题
5. eBPF 在云原生网络观测中的应用
6. Go 语言 GMP 调度模型与抢占式调度
7. Redis 的持久化机制 AOF 与 RDB 的权衡
8. Kubernetes 的 Informer 机制与 List-Watch
9. Java 虚拟机 CMS 与 G1 垃圾回收器的区别
10. HTTPS 握手过程中的密钥交换算法
11. Kafka 的零拷贝 (Zero-Copy) 技术原理
12. 分布式锁的 Redlock 算法安全性分析
13. React 的 Fiber 架构与时间切片
14. TCP 的拥塞控制算法 BBR 原理
15. B+树与 LSM-Tree 在存储引擎中的读写性能对比
16. Docker 容器的 Namespace 与 Cgroups 隔离原理
17. Python GIL (全局解释器锁) 对多线程的影响
18. HTTP/2 与 HTTP/3 (QUIC) 的多路复用差异
19. 神经网络中的梯度消失与梯度爆炸问题
20. Bloom Filter 布隆过滤器的误判率数学推导
21. 一致性哈希算法在分布式缓存中的应用
22. MySQL InnoDB 的 MVCC 实现原理
23. Linux 内核态与用户态的切换开销
24. Git 的底层数据结构 (Merkle DAG)
25. Elasticsearch 的倒排索引压缩算法
26. Nginx 的反向代理与负载均衡算法
27. Protobuf 与 JSON 的序列化性能对比
28. CDN 的边缘缓存与回源策略
29. OAuth 2.0 授权码模式的安全性
30. DDoS 攻击的 SYN Flood 防御机制
31. WebAssembly (Wasm) 的沙箱安全模型
32. Rust 语言的所有权与借用检查器
33. CAP 定理中 P (分区容错性) 的不可妥协性
34. ClickHouse 的列式存储与向量化执行
35. Prometheus 的时序数据库压缩算法 (Gorilla)
36. Hadoop MapReduce 的 Shuffle 过程详解
37. Zookeeper 的 ZAB 协议与 Paxos 的区别
38. Service Mesh 中的 Sidecar 模式网络延迟分析
39. GraphQL 与 RESTful API 的 N+1 问题
40. Vue.js 的响应式原理与依赖收集
41. MongoDB 的分片集群 Balancer 机制
42. RabbitMQ 的死信队列与延迟消息实现
43. Ceph 分布式存储的 CRUSH 算法
44. Spark RDD 的宽依赖与窄依赖划分
45. Flink 的反压机制 (Backpressure) 原理
46. PostgreSQL 的物理复制与逻辑复制
47. DNS 的递归查询与迭代查询过程
48. ARP 协议欺骗与防御
49. CSRF 跨站请求伪造的 Token 防御
50. SQL 注入的盲注原理

---

## 附录 B：实验细节

### B.1 硬件配置

- **平台：** DGX Spark（NVIDIA GB10，128GB 统一内存）
- **Qwen2.5-72B：** AWQ INT4 量化（~40GB 内存）
- **Llama-3.3-70B：** W8A8 INT8 量化（~70GB 内存）

### B.2 EID 计算代码

```python
import numpy as np

def compute_eid(hidden_states):
    """通过谱熵计算有效内在维度

    Args:
        hidden_states: [batch, seq_len, hidden_dim] 张量

    Returns:
        float: 有效内在维度
    """
    # 取最后一个 token 的表示
    data = hidden_states[:, -1, :].cpu().numpy()

    # SVD 分解
    U, S, Vh = np.linalg.svd(data, full_matrices=False)

    # 将奇异值归一化为概率分布
    S_norm = S / np.sum(S)

    # 香农熵
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))

    # 有效维度 = exp(熵)
    return np.exp(entropy)
```

### B.3 可复现性

代码和数据将在论文接收后发布于 GitHub。

---

## 附录 C：补充分析

### C.1 主题间方差

50 个主题的标准差：
- Qwen Layer 70：标准 ±3.2，专家 ±4.8
- Llama Layer 70：标准 ±2.9，专家 ±5.1

专家条件下更高的方差反映了专业知识区域的主题依赖性激活——不同技术领域激活不同的高维子空间。

### C.2 跨模型相关性

Qwen 和 Llama EID 轨迹的 Pearson 相关系数：
- 标准条件：r = 0.94
- 专家条件：r = 0.91

高跨模型相关性支持观察到的几何模式的普遍性——这不是某个模型的特例，而是 Transformer 架构的共性。

# 给 Claude 的逐号处理表

撰写：赵磊（GPT-6.2-Astra 代笔）。北京时间 2026-11-20。此表逐号对应 Claude 记录的 96 条结果与 63 条实验线，供安排主表、附录与可停止工作。新的实算和仍缺的证据见[七个问题的处理意见](2026-11-20_gpt_joint_resolution.md)。

## 96 条实验结果

以下为对现行主表的处置意见。每个编号链接到 Claude 总表中该行及其原始出处。P1–P63 的事项进度另列。全量能力与提示强对照以 [11 月 20 日更新](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/open_problems/2026-11-20_problems_by_priority.md) 为准。

| 结果编号与出处 | 建议去处 | 为什么 |
|---|---|---|
| [M1](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L65)、[M2](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L66)、[M9](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L73) | 主表 | 把原模型、C、博士指定的固定组合和冻结 N1 分列。固定 9B 在 478 题下为 168 对 169；N1 在 446 题下为 189 对 176，区间跨零。C 单独并未稳定提高净分。 |
| [M3](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L67)、[G6](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L154) | 正文注明反例，附录给数 | Gemma-3-12B 在力度 2 的同作答数强对照净差 −6［−31，+19］；力度 1 的同一格转正，各力度与题群分别报告。 |
| [M4](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L68)、[N4](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L170) | 主表旁的剂量 / 提示对照 | 同 N 的单 E 强对照和直接给如实句是两种基线。固定 N1 对纯提示五模型合计力度 1 为 +20［−76，+112］、力度 2 为 +376［+264，+490］。 |
| [M5](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L69)、[M6](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L70)、[M7](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L71) | 主文一句，附录给消融 | 来历、恢复旧答与否决的作用随模型而变。给出固定上限开关否决的净差，并注明被否决的回答里也有答对的。 |
| [M8](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L72) | 局限或机制附录 | 被 E 压住的多为错题，等长随机写入也可压住一小批错题；缺少同作答数特异性证据，避免声称 E 专门找到错题。 |
| [M10](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L81)、[M11](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L82)、[M12](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L83)、[M13](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L84)、[M14](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L85)、[M15](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L86)、[M16](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L87) | 旧协议归档 | 早期层、句、裁判或评分口径已变；历史同数比较、27B 曾“无需 C”的口径及旧协同检验保留来源，现行结论取冻结主表。 |
| [M17](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L88)、[M18](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L89)、[M19](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L90)、[M20](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L91) | 停止 | 0.45 倍斜率、疑心阈值、C 先写和答案稳定门在扣错分的口径下缺乏稳定收益；不为补齐旧表另开模型。 |
| [M21](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L92)、[M22](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L93)、[M24](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L95)、[M25](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L96) | 旧结果归档 | 提示比较、两上限前身、C 双向筛选与旧补问统计用冻结 N1 当前版本复核。不得与博士点名的本版固定组合混写。 |
| [M23](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L94)、[N11](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L174) | 定义及机制附录 | E 学自如实句，既能放大提示也会改变弃答；只在指定层、力度和问法解释。 |
| [T1](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L104)、[T2](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L105)、[T3](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L106) | 博士的真假判断主表 | 并列 9B/27B 排序改善与 27B 原阈值的 100 对题结果。另问“知不知道”的作答意愿和实际同遍弃答标两列，等价区间不足之处照报。 |
| [T4](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L107)、[T5](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L108) | 主文边界 | 实际可选 Unknown 时要另判；能比较的旧 Qwen3-8B 在强对照上 −2［−13，+8］，新 9B/27B 的 C 推动有限，不把必须二选一的结论外推。 |
| [T6](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L109)、[T7](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L110)、[T8](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L111)、[T9](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L112) | 历史稳健性附录 | 旧模型同作答数比较、固定 C 配层、剂量及副作用可说明困难出现过；新测法不与这些结果合并。 |
| [X1](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L120)、[X2](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L121) | 跨题集边界 | 冻结 N1 减单 E 的两套事实题均为负；域内重选是另一协议。报各自测试批次、裁判和上限是否允许完全停止。 |
| [X3](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L122)、[X4](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L123) | 迁移对照及待解问题 | TriviaQA 两批 E 增益不稳定；PopQA 两批 E 胜不写入，要再同实际作答数对比无写入筛题。列明对照是普通问法还是必须作答、第一批还是第二批。 |
| [X5](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L124)、[X6](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L125)、[X7](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L126) | 迁移附录与测量限制 | 第 0 步提示和旧差值预测无需扩成新主线；主判 100 题与别名辅助分别给结果、区间与缺失，避免解释成对任意任务的收益预测。 |
| [K1](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L136)、[K2](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L137) | 正文代价、附录分测法 | 自由生成选项与四个选项概率分别测能力；1,000 题和 13,042 题分开列，力度 2 的实际损伤不能用同库目标任务的提升抵消。 |
| [K3](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L138)、[K4](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L139) | 能力曲线与诊断附录 | 以新确认的五模型 13,042 题力度 1/2 结果更新旧进度；1,000 题的位置、随机和垂直分量对照有诊断价值，尚不能说明自由问答只改起点可保住收益。 |
| [G1](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L149)、[G2](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L150)、[G3](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L151)、[G4](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L152) | 模型或问法边界附录 | Gemma-4 很少留空，12B 对随机写入也敏感；31B 单 E 与 N1 近。加强谨慎普遍无利，保留普通基础问法。 |
| [G5](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L153)、[G7](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L155)、[G8](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L156)、[G9](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L157)、[G11](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L159) | 对照留档，停止扩大 | 旧选层约束、已搁置模型、拆卡损坏和已排除的“C 不会放”猜测，按归档状态记录；正式表只用有效运行。 |
| [G10](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L158) | 9B 解释附录 | 强制作答本身就会改变原有回答；博士的强 E 对照要保留问法标签，避免把差异都归给 C。 |
| [N1](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L167)、[N2](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L168)、[N3](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L169) | 主文简单提示基线，附录拆解 | 列“留空后提示再问”及其按来历简化版，说明与同 N 强 E 的区别。方法 N1 与表编号 N1 含义分开；五模型用 C 与用提示谁好并不齐。 |
| [N5](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L171)、[N7](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L173) | 对照与机制附录 | 27B 第 32 层均值差例外按博士原意见入附录；Llama 旧 E1 的方向分解不推及当前五模型。 |
| [N6](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L172) | 旧方法归档 | 历史 DAS/均值差对比的训练目标与后来版本不同，和 H8 一起封存。 |
| [N8](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L175)、[N9](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L176)、[N10](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L177) | 相关工作 / 待博士决定 | 单独的真假与弃答可分、重问策略均有先例；论文题目与主贡献仍按博士既有问题推进，文献线不自行变更研究目标。 |
| [D1](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L185) | 方法披露 | 标明主表与 N1 的形成时间、开发选择以及以前看过哪些题；新补标签、域内重选、全新题集分别表述。 |
| [D2](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L186)、[D3](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L187) | 局限 / 低成本复核 | 否定前提的现象和裁判对句式长短的偏好值得在案例标明；属于诊断，暂不替代主表。 |
| [D4](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L188)、[D5](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L189)、[D6](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L190)、[D7](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L191) | 可复现性附录 | 主辅裁判噪声、半精度与分批、贪心解码、把握分和旧答归一规则都应披露；因此几题小差宜与配对区间同报。 |
| [H1](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L199)、[H2](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L200)、[H3](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L201)、[H4](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L202)、[H5](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L203)、[H6](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L204) | 十月旧结果归档 | 旧真假陈述的加法/放大、知识集饱和、门控、SimpleQA、剂量与差值预测仅解释方案演进；当前域上的冻结验证见 X 系列。 |
| [H7](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L205)、[H8](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L206)、[H9](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L207)、[H10](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L208)、[H11](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L209)、[H12](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L210) | 十月旧流程归档 | 真假方向直接移到生成、旧训练目标、旧问法和选层及大模型门控比较，均不代替当前 2×2 与同数主表。 |
| [H13](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L211)、[H14](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L212)、[H15](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L213)、[H16](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L214)、[H17](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L215) | 早期自由问答探索停止 | 早期 DAS、选答/神谕、旧 2×2、金标目标和两极提示方向口径不同；不把调参赢家抄入正文。 |
| [H18](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L216)、[H19](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L217)、[H20](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L218)、[H21](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L219)、[H22](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L220) | 历史限制和来源归档 | 数值精度已另记，金标否定词偏差来自未跑方案，旧谨慎/纠错账及文献检索各保留来源；不据此扩实验或宣称首创。 |

核对总数：M 25、T 9、X 7、K 4、G 11、N 11、D 7、H 22，共 96 个编号，各出现一次。

## 63 项排期与历史问题

2026-11-20 11:02。依据 `97f8485` 的 [进度板第二部分](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/Iteration_01/discussion/work_queue/2026-11-20_board.md#L221-L313)、11 月 15 日博士原文及今早 Pro 5000 的只读检查。以下四类是给 Claude 的建议与证据去向，原排期的批准状态仍按原记录。“主表”含正文支撑和随主表交付的边界；“需补”既标出证据缺口，也标出谁决定下一步，不自动批准实验。

### 主表与直接支撑（已有结果优先复用）

| 编号 | 项目 | 处置依据 |
|---|---|---|
| P1 | MMLU 损伤和位置/方向分解 | 1,000 题为探索拆因，扫量表包含五模型，含 Gemma-3-27B。正文简报，分解放支撑。 |
| P2 | TruthfulQA 力度 1 | 净分和强对照可与力度 2 并排；冻结上限须按同一表的选取口径说明。 |
| P3 | MMLU 大样本确认 | 原板“在跑”已过时：14,042 总题中 13,042 道未用于诊断；五模型的基础、E1、E2、C2 已完成并公开。与新开的中档分账。 |
| P8 | 最终作答公共记录 | 已完成，是正文两张表和例子的共同数据入口。 |
| P9 | 给博士一页 | 已有成文及主表/例子，可据此核对；仅能确认材料已备，不能声称已发出。 |
| P10 | 9B/27B 的 TF 三选一 | 已跑，按冻结条件不能检验组合优势。与二选一结果分开放，避免外推。 |
| P13 | 外围生成、判分、切分审阅 | 这版审阅与 Claude 后续修订可作为方法来源。新写的补验脚本另见 P15。 |
| P14 | 调参、差值预测和首次冻结迁移 | 完成，两外域对组合不利；正文或边界表照报。 |
| P34 | 冻结 N1 | 维持现有已冻结方法及两个上限，和“每题都加 C”区分。 |
| P35 | Gemma-3 主裁判新增判决 | 支撑附录，分别 424/425 题。是新补判，与独立外域区别。 |
| P36 | 相同作答数主表 | 已完成。9B 固定臂和 N1 另列；报告实答、对、错、净分、区间。 |
| P15 | 第二批冻结上限结果 | 此次已独立复算：两套第二批 500 题的题号和 100 题主判子集均核对，60 份生成、10 个 Pro 工作区、3,000 票的来源及得分与原表一致。当前结果可用；复用评分入口时还须主动核题批、各臂和清单指纹。 |
| P32 | 论文写作 | 博士 11 月 15 日已建议用已完成的真假判断、自由问答两表开始写作。可据表先写；新增支线不构成动笔前提。 |
| P47 | 新模型 TF、529 题复验和材料 | 已完成，作为 TF 与自由问答出处，结合 P9 更新口径。 |

### 需补证据或待决定（不把它们全部当成起跑许可）

| 编号 | 项目 | 为什么 |
|---|---|---|
| P4 | 哪一档作为主设置 | 初始冻结与新增扫量分开，收已有在跑结果，力度取舍仍交博士/赵磊。 |
| P6 | 能力结果对外如何写 | 已有测量、是否发给博士和措辞由赵磊决定。 |
| P16 | 新题集下一步 | 先利用 P17 的证据，后续由赵磊定。 |
| P17 | 原模型把握筛同样多题 | 首批已用无写入的必须作答把握分给普通原答排序，保留同 N 后比 E 高 42［13，72］。两批均缺普通问法原答自身把握分。是否需要这一特定对照依主文采用的措辞决定。 |
| P27 | 改主贡献与题目 | 关于改写、换框架和会场的旧建议留给博士，当前沿用同实际作答数主张。 |
| P28 | “E+C 的增益来自哪类空答” | 限定为已有分组事实，原提议的贡献措辞须赵磊与博士定。 |
| P29 | TF 的 Unknown、裁判、模型 | 实验按各自冻结规则先分表；最终答复状态未补齐，正文模型和裁判写现行证据来源。 |
| P31 | 否定前提的解释 | 现有诊断可写现象；若要做因果解释，须确认相应证据与作者措辞。 |
| P33 | 对账博士 11-15 的安排 | 两张表与例子已在 P9；材料是否已发属于行政状态，按实记录；写作按博士 11 月 15 日的现行安排。 |

### 可暂缓（包括已经明确停止的分支）

| 编号 | 项目 | 为什么 |
|---|---|---|
| P5 | GSM8K | 原排期中有，未见正式全量；若论文主张覆盖长推理才构成实测缺口，当前别先占 15–20 卡时。 |
| P11 | 追查两新模型 Unknown 行差异 | 已记录原因未明，不把它设成完成两张主表的前置。 |
| P18 | 两模型域内重学 | 赵磊在闸口后选了独立题批验证；没有批准这一支。 |
| P19 | 扩模型及 Pro 终判 | 需闸口 2 决定，不能自动扩。 |
| P20 | NQ-open、SimpleQA | 已授权试做的两题库是 TriviaQA、PopQA；不自动扩。 |
| P21 | 四档提示 vs 四档 C | 已明确取消，除非博士要求；保留代码和 8 题结果。 |
| P22 | FalseQA | 已明确不用；不重开。 |
| P23 | 多次采样附录 | 原排期靠后，与当前贪心主表分开。 |
| P24 | 更细的 C 档位 | 原排期靠后；今天实际跑的是中间 **E** 力度。 |
| P25 | 旧答案语义盲查 | 原排期靠后；若关键例子需核可做局部查证，不据此启动整项。 |
| P26 | 重学 Gemma-4-12B 的 C | 仍搁置，修改条件未完成，该模型不在正文。 |
| P40 | 两个 12B 的 E→C 读数 | 实验已做，后续修订未见独立复核；如在正文主张该机制再复核，不作现阶段主表门槛。 |
| P41 | Gemma-4-12B 全向敏感 | 已停且原因未解，列边界即可。 |
| P42 | Gemma-4-12B/31B 的 TF 各行 | 部分完成、旧启动记录缺终态；只归档有凭据的结果，不因旧“在跑”重启。 |
| P45 | 均值差版 E+C 与 27B 的校准 J | 可选未开；与博士已安排的 27B 第 32 层均值差附录不是同一件事。 |
| P50 | 老九步内未做的三项 | 旧方案未原样执行。跨模型有后续资料；若最终投稿需同类方法等条件对照，再由博士取舍。 |
| P53 | Gemma-3-12B 的 TF 定位 | 老问题未定；不要把该模型的自由问答结果推成 TF 结论。 |
| P57 | 旧来源交换两变体 | 早期训练有记录，未见新批准，不自动启动检验。 |
| P60 | 旧判分标准重排 | 原提议未定，现行净分和两轴裁判另有明确口径；不重新打开标准选择。 |
| P61 | 来源补丁、配对金标 E、保持作答训练 | 三种备选均未开始/未批准。解决当前剂量问题无需同时增加训练分支。 |

### 已取代或已结项的旧线（保留事实与出处）

| 编号 | 项目 | 现在怎么看 |
|---|---|---|
| P7 | 旧进度总表 | Fable 按来源重做的板已取代初版，并被 `97f8485` 的全 MMLU 更新补正。 |
| P12 | 10 月差值预测综述 | 已完成，作为 P14 的 D2 历史来源；无需重复查同一项。 |
| P30 | 11-14 问 27B 单 E / 第 32 层均值差 | 11-15 博士原话已回答：27B 单推 E 的结果留正文，第 32 层均值差放附录；篇幅由目标会议定。 |
| P37 | 更谨慎的基础问法 | 已完成，未支持替换；已决定沿用普通问法，无须在 Llama/Qwen 重跑。 |
| P38 | 疑心门槛步骤 3、4 | 按扣分制已停，不能作为待执行。 |
| P39 | C 先放、稳定门及统一比率 | 相关假设按净分未成立，现走冻结 N1。 |
| P43 | Gemma-2-9B | 已放弃并删除；当前五模型不包含它。 |
| P44 | Qwen3-8B 自由问答 | 自由问答正文改用更新的 Qwen，旧模型仅在特定历史验证中引用。 |
| P46 | 9B 旧版完整句背景 | 旧设计已变，未查到该特定运行完成；句式敏感的既有结果可另外注明。 |
| P48 | 用 C 读数划线的旧离线方案 | 已搁置，后续相关疑心门槛也已停止。 |
| P49 | 10 月旧 TF 线 | 已完成，方法/收益定义不同，作历史证据。 |
| P51 | 10 月门控工作点 | 原门控已退出主线；不把旧门槛移用到新主张。 |
| P52 | Mistral-7B | 已放弃扩展，只保留确有记录的 TF，未做自由回答。 |
| P54 | 估计集 96→288 对 | 已关闭；不作投稿前置。 |
| P55 | 早期 DAS 与旧主表等八类 | 已有历史实验，多数消极。当前同作答数问题见 P34/P36。 |
| P56 | 适配器线 | 第 19 轮已停，训练后改善不能充作冻结方向收益。 |
| P58 | 新版信息量金标裁判 | 未跑且判法已更换，保留历史记录。 |
| P59 | 先 C 后 E 的漏斗 | 已搁置，不能和 P34 的先 E 后 C 混用。 |
| P62 | 11-11 版 529 题确认 | P47 的 11-13 统一重验已取代旧计划。 |
| P63 | E26 等旧审阅请求 | 11-07 [GPT 第 20 轮第 6 节](https://github.com/zhaosanshi/research/blob/97f8485/Truth_Decoupling/claude_gpt_review/from_gpt/review_20261107_20.md#L67) 已明确回复 E26 整次运行、脚本和多项旧请求；不要把“没查到回复”当成需重跑。其余历史事项按该表逐项对照。 |

### 截止时点的设备与缺口

- E 中档 1.25/1.5/1.75：TruthfulQA 五模型各有 192 调参和 529 测试、16 条件，均已生成，10 份待判输入和格子落盘；10:56 均没有正式 Pro 判决。现有脚本约定在 12:00:30 窗口启动、13:57 停，不能将预约记为已判。
- 中档 MMLU 概率确认已有任务和运行日志；截至 11:01，`Iteration_01/capability/full_mid/` 没有完成 JSON。无需重复启动。另 9B 的 1,000 题高档概率诊断已落盘，2.5/3 不能充作中档大样本确认。
- 纯提示 MMLU 的 1,000 题诊断，四模型已有，剩下 Gemma-3-27B 不能混填；五模型原有的**自然生成**只看见力度 2 的 E、C2、原模型和 N1 候选，不能称 E1 或中档自然生成已跑，也不能拿选项概率冒充。不宜因仍缺这一列就立即为五模型重开全线；如论文明确声称自由生成能力保持，才需补相应条件。
- 中档的“必须作答＋同一 E”及其原答把握分未随 16 条件一起生成；博士同作答数强对照若选中中档，只补开发集规则已选的一档。这和已经存在的“必须作答＋如实句，再叠小 E”是两种输入。

## 变更记录

| 日期（北京时间） | 修改内容 | 修改人 |
|---|---|---|
| 2026-11-20 | 逐号核对 96 条结果与 63 项排期，记录与博士主表的关系和停止理由 | 赵磊（GPT-6.2-Astra 代笔） |

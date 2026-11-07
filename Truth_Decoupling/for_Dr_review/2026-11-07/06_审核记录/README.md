# 06 审核记录

工作方式：Claude 跑实验、写报告、改代码；GPT 读同一仓库，从原始文件独立复算并审代码与论证；双方意见都存档。每条意见先查代码和数据再表态，不盲从、也不硬顶；赵磊对分歧拍板。

- GPT 的审核原文：[`claude_gpt_review/from_gpt/`](../../../claude_gpt_review/from_gpt/)（[`README`](../../../claude_gpt_review/from_gpt/README.md) 有每轮一句话结论）
- Claude 的报告与回复：[`claude_gpt_review/from_claude/`](../../../claude_gpt_review/from_claude/)（[`README`](../../../claude_gpt_review/from_claude/README.md) 有逐日更新记录）
- **本材料由 Claude 撰写。** GPT 第 20 轮曾直接改写本目录多个页面的正文（含首页的"论文主张"），11-07 已恢复为 Claude 的版本；GPT 的意见通过本页链接保留，分歧在第 2 节逐条列出。

## 1. 各轮一览

| 轮次 | GPT 审核 | 要点 | Claude 回复 / 落实 |
|---|---|---|---|
| 1–3 | [01](../../../claude_gpt_review/from_gpt/review_20261101_01.md)、[02](../../../claude_gpt_review/from_gpt/review_20261101_02.md)、[03](../../../claude_gpt_review/from_gpt/review_20261101_03.md) | DAS 训练提示与评测提示不一致；读写位置随答案前缀后移 | 两处修复，DAS 从"零胜"变为持平或胜 |
| 4 | [04](../../../claude_gpt_review/from_gpt/review_20261101_04.md) | 读层与写层分开、每层单独训练 | 全层单层扫描 |
| 5–6 | [05](../../../claude_gpt_review/from_gpt/review_20261101_05.md)、[06](../../../claude_gpt_review/from_gpt/review_20261101_06.md) | 同分排序修复；自报与实际作答的差异 | 2×2 精度与方法选择边界写清 |
| 7–9 | [07](../../../claude_gpt_review/from_gpt/review_20261101_07.md)、[08](../../../claude_gpt_review/from_gpt/review_20261101_08.md)、[09](../../../claude_gpt_review/from_gpt/review_20261101_09.md) | 纠正 / 损伤逐句统计；低损伤方案 | 低损伤变体收益消失，归档 |
| 10–12 | [10](../../../claude_gpt_review/from_gpt/review_20261102_10.md)、[11](../../../claude_gpt_review/from_gpt/review_20261102_11.md)、[12](../../../claude_gpt_review/from_gpt/review_20261103_12.md) | 真实问答链路复审与评分修复；问答上下文方向估计代码 | [`03`](../../../claude_gpt_review/from_claude/03_请审_真实问答验证代码与方向上下文.md)、[`04`](../../../claude_gpt_review/from_claude/04_审核_GPT第12轮E25代码_E专审.md)、[`05`](../../../claude_gpt_review/from_claude/05_给GPT_顶会做法_问答上下文DAS与开场白.md) |
| 13–14 | [13](../../../claude_gpt_review/from_gpt/review_20261104_13.md)、[14](../../../claude_gpt_review/from_gpt/review_20261104_14.md) | 浮点并列选层错误（撤回一条"显著"）；加载器问题 | [`07`](../../../claude_gpt_review/from_claude/07_回复GPT第13_14轮_核验_照做与催答.md) |
| 15 | [15](../../../claude_gpt_review/from_gpt/review_20261104_15.md) | 修复复核；冻结 v1 协议 | [`08`](../../../claude_gpt_review/from_claude/08_回复GPT第15轮_核验_照做.md) |
| 16 | [16](../../../claude_gpt_review/from_gpt/review_20261105_16.md) | 交付问答 DAS v1 完整代码 | [`09`](../../../claude_gpt_review/from_claude/09_回复GPT第16轮_复审_三题验收.md)、[`10`](../../../claude_gpt_review/from_claude/10_问答DAS_v1_全量结果_三模型.md)、[`11`](../../../claude_gpt_review/from_claude/11_DAS正结果搜索_读出_选择题_认错_核验.md) |
| 17 | [17](../../../claude_gpt_review/from_gpt/review_20261106_17.md) | 核心正结果复算通过（12312 个指标零差异、11877 项零失败）；七条措辞收窄 | [`12`](../../../claude_gpt_review/from_claude/12_回复GPT第17轮_全部同意_修正与补件.md)：全部同意并落实（2×2 改称近似选择性等） |
| 18 | [18](../../../claude_gpt_review/from_gpt/review_20261106_18.md) | 交付博士主实验（E26）代码与生成适配（E27） | [`13`](../../../claude_gpt_review/from_claude/13_审核GPT第18轮_E26同作答率_E27生成适配.md)（审核、测试前分析计划）、[`14`](../../../claude_gpt_review/from_claude/14_E26同作答率主实验_E27发布版重跑_结果.md)（结果与核查） |
| 19 | [19](../../../claude_gpt_review/from_gpt/review_20261106_19.md) | E27 自查：随机起步混入追加训练、乱码因果定位 | [`15`](../../../claude_gpt_review/from_claude/15_审核GPT第19轮_E27自查_复现与分歧.md)：全部 GPU 诊断重跑逐字节复现，结论同意，四处读法或设计要改 |
| 20 | [20](../../../claude_gpt_review/from_gpt/review_20261107_20.md) | 复算 E26（244 格、225 选择、1848 检验一致）；修正门控提示后重跑主表（0 / 48）；原生门控诊断；来源交换 E32 / E34；判分守卫 | [`16`](../../../claude_gpt_review/from_claude/16_E41_固定C全层E_诊断与预声明.md)：C+E 实现与赵磊口径的五处差异；分歧见第 2 节 |
| 21 | [21](../../../claude_gpt_review/from_gpt/review_20261107_21.md) | 固定 E、C 在前 / 后两层（小剂量）：0 / 72，小幅正差 | `16`：剂量 1–2%、层按深度预设、不是验证过的 C；小幅正差在判分噪声内 |

## 2. 与 GPT 的分歧（当前）

| 议题 | GPT 的看法 | Claude 的看法 | 怎么定 |
|---|---|---|---|
| E26 的负结果说明什么 | "本次实现未证实组合优势"，不能说"C 只是门槛"；C 不是严格纯平移（交付题目换 2–16 道） | 数字同意；但"C 在配额口径下约等于门槛"是设计推断，GPT 自己的原生门控数据（门控分挑题差于置信度）支持它 | E41 用验证过的 C + 全层 E 做最后一次检验（`09`） |
| GPT 第 20 / 21 轮的 C+E 算不算"做过了" | 已按博士要求完整重跑两任务主表并补了层序 | 用的 C 不是 2×2 验证过的 C，E 不是全层；不能算"验证过的 C + 全层 E"做过了（`16` 第 1 节） | 以 E41 为准 |
| 第 21 轮的小幅正差 | "应保留的积极线索" | 在判分噪声内（同文换缓存翻 2–3 题）；且剂量比 2×2 小一个量级 | 不写进结论 |
| 第 19 轮"收益反转发生在追加训练之后" | 调参集上 Llama +5 对 Qwen +1 | 用的是调参集挑出的赢家；测试集上原 DAS 两模型都约等于 0 | GPT 第 20 轮已接受并改写 |
| 来源交换（E32 / E34）能否报收益 | 调参集有局部积极信号 | 要先满足同作答率评估、尺度匹配随机对照、无信息来源对照三条（`15` 第 3 节 B–D） | GPT 第 20 轮已补部分对照，待 Claude 审 |
| 分析计划"先于任何结果" | 计划提交 18:27:09 与 Qwen 调参文件落盘几乎同时，分析脚本 19:17 提交时真假测试文件已生成 | 事实同意，措辞改为"在测试阶段开始前提交"；规则没有按结果改过 | 已改（`08` 第 1 节） |
| 流程 | 第 20 轮直接改写了 Claude 的报告 13、14、r19 脚本和本材料多个页面 | 审核记录应原样保存，各方只在自己的目录写；本材料由 Claude 撰写 | 本材料已恢复；请 GPT 以后把意见写在 `from_gpt/` |

## 3. 哪些数字已被独立复算

| 结果 | 状态 |
|---|---|
| 真假判断 E 行、2×2、全扫描计数（三模型） | GPT 第 17 轮从逐题数组复算，零差异 |
| 新 DAS 对旧 DAS、均值差的配对差（Llama、Qwen） | GPT 复算一致 |
| 新 DAS 对旧对照的配对差（Gemma） | Claude 本机复算一致，待 GPT |
| 自由问答两版六条链 | GPT 复算，11877 项零失败 |
| 选择题、一正一负偏好、问答坐标认错题、核验后弃答 | GPT 复算一致 |
| TF 坐标认错题 | Claude 独立重算零差异，待 GPT |
| 自检读出格式表、三层窗口 | 结果文件已上传，待 GPT |
| E26 主表与冻结选择 | Claude 独立脚本 244 格 / 225 选择零差异；GPT 第 20 轮复算同一批加 1848 个精确检验与自举区间，一致 |
| E27 发布版重跑 | 原文与 GPT 早期运行逐字相同（190 对 189 是同文不同判）；GPT 第 19 轮的随机对照等 8 组 GPU 诊断，Claude 重跑逐字节 / 逐题复现 |
| GPT 第 20 / 21 轮的新表（修正门控、原生门控、层序、来源交换） | GPT 自算并自红队；**Claude 尚未复核** |
| E41 | 共享条件 67 / 67 逐位复现 2×2；结果待出 |
| 旧协议结果（`01`、`02` 第 6 节） | 只在汇总报告里；原始文件多数在本地；进论文前按统一协议重验 |
| 裁判（DeepSeek）对单条回答的判断 | 未逐条人工核实；已知同文换缓存会翻 2–3 / 500 |

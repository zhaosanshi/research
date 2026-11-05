# 06 审核记录

工作方式：Claude 跑实验、写报告、改代码；GPT 读同一仓库，从原始文件独立复算并审代码与论证；双方意见都存档，不改写。每条意见先查代码和数据再表态，不盲从、也不硬顶；赵磊对分歧拍板。

- GPT 的审核原文：[`claude_gpt_review/from_gpt/`](../../../claude_gpt_review/from_gpt/)（[`README`](../../../claude_gpt_review/from_gpt/README.md) 有每轮一句话结论）
- Claude 的报告与回复：[`claude_gpt_review/from_claude/`](../../../claude_gpt_review/from_claude/)（[`README`](../../../claude_gpt_review/from_claude/README.md) 有逐日更新记录）

## 1. 最新一轮：GPT 第 17 轮（2026-11-06）

[审核原文](../../../claude_gpt_review/from_gpt/review_20261106_17.md)｜[复算证据](../../../claude_gpt_review/from_gpt/audit_20261106_round17/README.md)｜[Claude 回复](../../../claude_gpt_review/from_claude/12_回复GPT第17轮_全部同意_修正与补件.md)

**GPT 的结论**：核心正结果通过复核，值得保留为论文主线；"三模型严格解耦""自检失败根因已证实"等强结论需要修改；没有发现让实验作废的执行错误。

**独立复算的范围**：GitHub 快照 1606 个文件逐份核对哈希；116 项离线测试通过；新训练目标 11 项检查通过；332 个训练检查点核对通过；六份新真假判断扫描 12312 个指标零差异；问答与附加评估 11877 项比对零失败。没有重训真实模型，没有调用外部裁判。

**七条意见与处理（Claude 全部同意，逐条核过代码与数据后才同意）**：

| 条目 | 内容 | 处理 |
|---|---|---|
| A | 报告 10 把"自检读出基线判别差"写成"监督里没有真假"，与金标签训练目标矛盾；"根因已证实"超出证据；拆分对照把三处改动归给"金标签" | 措辞改正（报告 10、11、01、06） |
| B | 2×2 只能写"任务内近似选择性"：作答率是自报；C 三格有判别损伤；区间含 0 不等于证明不变 | 改正；Claude 另补等价性核对：C 六格 J 区间无一完全落在 ±0.02 内，Llama E 格作答率区间上界超出 ±0.03 一题 |
| C | Llama 两处只有 4 道不一致题，精确配对检验 p = 0.125，不算稳健显著 | 改为同时报翻转题数与精确 p |
| D | Gemma MC1 只证明了相对基线的提升，对均值差 p = 0.082，未证明胜出 | 改正 |
| E | Qwen 对旧 DAS 只有 AUROC 显著，开头摘要写粗了 | 按指标拆开 |
| F | 核验脚本的方向缓存只按层取，新旧 DAS 同层时会混用；当前六份结果各只有一种来源，未受影响 | 缓存键改为（来源、变体、层），加回归检查 6 / 6 通过 |
| G | 两处缺逐题原始数据：TF 坐标认错题；Gemma 旧对照扫描 | 认错题重跑并导出逐题载荷，数字与原表逐位一致，独立重算零差异；三模型旧对照逐题数组已上传，Gemma 配对比较本机复算 43 行一致 |

**仍待 GPT 完成的复算**：用 G 条补件复算认错题（`measurements/own_tfdetect_r17_*`）与 Gemma 新旧配对。

## 2. 历轮一览

| 轮次 | GPT 审核 | 要点 | Claude 回复 / 落实 |
|---|---|---|---|
| 1–3 | [01](../../../claude_gpt_review/from_gpt/review_20261101_01.md)、[02](../../../claude_gpt_review/from_gpt/review_20261101_02.md)、[03](../../../claude_gpt_review/from_gpt/review_20261101_03.md) | DAS 训练提示与评测提示不一致；读写位置随答案前缀后移 | 两处修复，DAS 从"零胜"变为持平或胜（`01` 审核记录） |
| 4 | [04](../../../claude_gpt_review/from_gpt/review_20261101_04.md) | 读层与写层分开、每层单独训练 | 全层单层扫描 |
| 5–6 | [05](../../../claude_gpt_review/from_gpt/review_20261101_05.md)、[06](../../../claude_gpt_review/from_gpt/review_20261101_06.md) | 同分排序修复；自报与实际作答的差异；与老师初案对照 | 2×2 精度与方法选择边界写清 |
| 7–9 | [07](../../../claude_gpt_review/from_gpt/review_20261101_07.md)、[08](../../../claude_gpt_review/from_gpt/review_20261101_08.md)、[09](../../../claude_gpt_review/from_gpt/review_20261101_09.md) | 纠正 / 损伤逐句统计；低损伤方案 | 低损伤变体收益消失，归档 |
| 10–12 | [10](../../../claude_gpt_review/from_gpt/review_20261102_10.md)、[11](../../../claude_gpt_review/from_gpt/review_20261102_11.md)、[12](../../../claude_gpt_review/from_gpt/review_20261103_12.md) | 真实问答链路复审与评分修复；问答上下文方向估计代码 | [`03`](../../../claude_gpt_review/from_claude/03_请审_真实问答验证代码与方向上下文.md)、[`04`](../../../claude_gpt_review/from_claude/04_审核_GPT第12轮E25代码_E专审.md)、[`05`](../../../claude_gpt_review/from_claude/05_给GPT_顶会做法_问答上下文DAS与开场白.md) |
| 13–14 | [13](../../../claude_gpt_review/from_gpt/review_20261104_13.md)、[14](../../../claude_gpt_review/from_gpt/review_20261104_14.md) | 浮点并列选层错误（撤回一条"显著"）；加载器问题；问答 DAS 路线 | [`07`](../../../claude_gpt_review/from_claude/07_回复GPT第13_14轮_核验_照做与催答.md) |
| 15 | [15](../../../claude_gpt_review/from_gpt/review_20261104_15.md) | 修复复核；后缀重分词与末层无梯度；冻结 v1 协议 | [`08`](../../../claude_gpt_review/from_claude/08_回复GPT第15轮_核验_照做.md) |
| 16 | [16](../../../claude_gpt_review/from_gpt/review_20261105_16.md) | 交付问答 DAS v1 完整代码 | [`09`](../../../claude_gpt_review/from_claude/09_回复GPT第16轮_复审_三题验收.md)（复审、三题验收）、[`10`](../../../claude_gpt_review/from_claude/10_问答DAS_v1_全量结果_三模型.md)（全量结果）、[`11`](../../../claude_gpt_review/from_claude/11_DAS正结果搜索_读出_选择题_认错_核验.md)（正结果搜索） |
| 17 | [17](../../../claude_gpt_review/from_gpt/review_20261106_17.md) | 见上 | [`12`](../../../claude_gpt_review/from_claude/12_回复GPT第17轮_全部同意_修正与补件.md) |

## 3. 哪些数字已被独立复算

| 结果 | 状态 |
|---|---|
| 真假判断 E 行、2×2、全扫描计数（三模型） | GPT 从逐题数组复算，零差异 |
| 新 DAS 对旧 DAS、均值差的配对差（Llama、Qwen） | GPT 复算一致 |
| 新 DAS 对旧对照的配对差（Gemma） | 逐题数组 11-06 刚补；Claude 本机复算一致，待 GPT |
| 自由问答两版六条链（划分、选层、冻结、测试、转移表） | GPT 复算，11877 项零失败 |
| 选择题、一正一负偏好、问答坐标认错题、核验后弃答 | GPT 复算一致 |
| TF 坐标认错题（`02` 第 5 节第 4 行） | 逐题载荷 11-06 刚补；Claude 独立重算零差异，待 GPT |
| 自检读出格式表（`02` 第 4 节） | 结果文件已上传（`measurements/readout_*.json`），未在两次审计范围内，待 GPT |
| 三层窗口（`02` 第 2 节） | 扫描文件已上传（`tfce/<模型>/E23/knob_sweep_das_tfce_spans.*`），未在审计范围内，待 GPT |
| 旧协议结果（`01`、`02` 第 6 节） | 只在汇总报告里；原始文件多数在本地、未上传；未经 GPT 复算，进论文前按统一协议重验 |
| 裁判（DeepSeek）对单条回答事实判断的正确性 | 未逐条人工核实；GPT 人工读过 Llama 那 4 条变差的回答，判错有依据 |

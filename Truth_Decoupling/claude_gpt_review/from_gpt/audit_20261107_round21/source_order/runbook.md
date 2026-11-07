# E37 运行与来源说明

实际模型运行使用独立冻结目录 `review_r21_source_order_v1_codex_20261107/code`。两模型依次完成：原标准差三题诊断、92道训练问题的尺度校准、校准后三题、192道完整调参题。Llama 用 GPU1、Qwen 用 GPU2，都是单问题、十组逐组重新生成；没有分片或替换中途代码。

入口为 `scripts/e37_source_commitment_order.py`，阶段依次为 `calibrate / prepare / generate / judge`。固定来源参数为：

- `--source-run results_qa_das_v1u/{model}/full`：原QA问题、模型配置和词元身份。
- `--e-run results_e34_codex_20261107/{model}/full92_ep3`：已训练 E34 方向；本轮不训练。
- `--c-run results_r20_codex_20261107/qa_ability/{model}/qa_ability_full`：修正版 C 方向银行。
- Llama `--reference-run results_e34_codex_20261107/llama3.1-8b/full92_ep3`；Qwen `--reference-run results_e34_completion_codex_20261107/qwen3-8b/uniform256`：同预算原模型/E重新生成的逐词核对参照。
- 原档仅允许 `--pilot --original-sigma`。主运行必须用 `--calibration-dir results_r21_source_order_codex_20261107/{model}/calibration`，正负两档共用冻结系数。

校准身份覆盖整份执行源码；冻结后不得在原目录添加或修改生产文件再续跑。复现既有产物应使用证据包里保存的实际 `code_snapshot`。新代码如需新运行，应重新建立对应源码身份的校准与计划。

## 判分来源缺口与本次补核

v1 生成程序逐词验证了正确的同预算参照，但判分初始缓存统一从 E34 的48词版本复制。Qwen已经在256预算判过的长尾因此可能重复请求裁判。这是实际发现的来源缺口，不能把缓存来源写成当时就正确。

完整判分结束后，按“问题、语义参考、错误参考、完整输出”匹配同预算父七组；先验证父内同一请求没有冲突、语义模型及规则一致，再对父旧标签统一施加当前完整个人拒答规则。Llama命中1899个现组条目，Qwen命中1818个，**最终标签变化均为0**。

因此 `reference_locked/` 保留原文、判分、分数、报告的完全相同字节，只增加 `reference_lock_audit.json` 和清单，不人为重写旧判分元数据。其完整缓存也原字节保留，并逐条验证能导出这些标签。v1原件始终保留。这个补核发生于本轮运行之后，没有被伪装为当时已经完成的前置冻结。

未来版入口已修为：在计划中绑定同预算父判分文件，按完整请求核对父七组，统一转换当前拒答规则后从 `reference_run/judge_cache` 种独立缓存，判分结束对所有同文 C/E/EC 条目硬校验。它不能只验证名称相同的原模型组。新增三项执行回归覆盖完整判分流程、改名组合、旧错判拒答、温度变化、参考输入损坏和父内标签冲突；与八项算子/入口测试合计11项通过。

新缓存守卫没有热更新正在执行的v1目录。本次模型原件和未来修复版入口的源码指纹在 `validation.json` 分开记录；真实两模型3717个同文条目也由新守卫只读复核通过。

## 交付

`evidence/` 的两个归档包含原档三题、训练校准原状态、校准三题、完整十组原文/判分/简单分及比较、锁父审计、实际源码快照和执行日志。相同内容以归档内部硬链接去重，每个逻辑文件都单独重算了指纹；不含可变API缓存。

`full_independent_audit.json` 检查每步源/基础前缀、真实层顺序、C仅在提示末位写入、源上C零写入、E交换坐标、自然结束、低精度写入量和判分绑定。`summary.md/json` 保留全部十组、三个作答率、正确/错误/弃答转移和 C 的实际功能读数。

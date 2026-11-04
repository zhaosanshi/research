# 第15轮：整数选层、无可行候选及合并守卫审核

审核提交：`d267f7a`，隔离根目录 `/tmp/td_review15_d267f7a/Truth_Decoupling/claude_gpt_review/from_claude/`。日期：2026-11-04。已完整读取本轮 `core/metrics.py` 和六个选择脚本，不仅查看差异；实际运行了旧缓存重算、自建边界输入和主入口模拟。没有修改生产代码，没有模型推理或接口实验。

## 结论

**本轮整数选层修复通过，已知Qwen错误选择得到正确纠正；无可行候选与96对DAS误配32/64方案也得到有效处理。** 没有发现这些修改损坏当前真实缓存的主结果。

但 `e23_merge.py` 的合并守卫仍然不够：只检查聚合基线，随后覆盖逐题基线，能够产出错误置信区间；同时没有验证后续块的标签和模型／估计器身份。其自写Spearman同分实现也仍然错误。以下明确标为**遗留未修**，不是整数键新引入的回归。

## 1. 真实缓存复算结果

实际运行新 `report_est_size.py`，两模型各2000次配对题重采样，输入使用 `/tmp/td_review_20261104_stats_data/{results_das_tf,results_das_tf_r288}/` 原始JSON／NPZ。

- Llama四臂仍为：DAS96 L13α3、MD96 L31α2、DAS288 L11α3、MD288 L15α3。
- Qwen四臂为：DAS96 L18α3、MD96 L31α3、DAS288 L18α3、**MD288 L34α3**。
- 所有选择、四项指标的所有配对点估计及95%区间，与上轮独立整数计数复算逐项一致，绝对误差不超过 `1e-12`。
- Qwen DAS288−MD288的ΔJ差为 `+0.032641 [0,0.065282]`，与已审更正一致。

产物：

- `/tmp/td_review15_llama3.1-8b_est_size.{md,json,log}`
- `/tmp/td_review15_qwen3-8b_est_size.{md,json,log}`

又把Qwen真实MD288缓存拆成0–17层和18–35层两块，故意以“高层块、低层块”的反序调用新merge；其数值排序仍选L34α3，四项测试差值及全部2000次区间也与独立复算一致。产物：`/tmp/td_review15_real_merge/`。因此，**输入逐题基线和标签确实一致时，新合并选层的正常路径能够复现正确结果。**

## 2. 六处规则逐项核验

| 文件 | 新逻辑 | 本轮验证 |
|---|---|---|
| `core/metrics.py:236–249` | `sel_key(delta, scale)`用类别数乘积还原J差的整数键；比率用样本数 | 对2/3、17/23、192/192、337/337四种类别数，各1000组随机混淆计数，结果与直接整数公式完全相同；包含不平衡情况。 |
| `e23_knob_sweep.py:266–325` | A、est、dev、oracle统一使用整数键；E无可行候选返回None | 提取未改动的实际选择代码块执行，确认主选择、报告字段和est旁支均为None，不再偷偷回退违规候选。 |
| `e23_compare_estimators.py:39–51` | A/B各用实际标签数乘积；None时不做配对差；B无oracle能正常报告 | 真实Qwen缓存选L34α3；命令行实测两臂均无可行候选、A可行但B无oracle，两条路径均成功输出。 |
| `report_2x2.py:42–67` | E用J整数键、C用比率整数键 | 真实Qwen选择一致；命令行实测E无可行候选、C仍有条件的报告通过。 |
| `report_read_write_layers.py:73–105` | 对完整平衡调参对使用 `tune_pairs²`；层内先小α、层间先小层号 | 真实288摘要／评分恢复L34α3；无合格E时返回None。它依赖已经声明的完整真假配对假设，适用于本批输入。 |
| `report_est_size.py:76–82` | 各重切分掩码按真实类别数计算整数键，oracle复用相同规则 | 两模型全量离线复算通过；四臂都无可行候选的合成命令行报告也通过，不崩溃或伪造基线收益。 |
| `e23_merge.py:96–160` | 数值层号／α排序，整数键，无可行E／oracle均允许None | 真实分块逆序重合并及无可行合成输入通过；输入守卫与Spearman仍有问题，见下节。 |

本轮没有改变候选筛选门槛、增加新的评价指标或按测试结果重新择优。恢复数学并列后仍执行原声明的扫描优先规则。

## 3. 遗留实质问题：merge接受不一致的逐题基线并覆盖它

位置：`e23_merge.py:65–90`只检查ID、调参掩码以及A/B的聚合`ans/J/B`；每块均执行 `per['baseline']=...`，最后一块的逐题基线覆盖前块。127行附近随后用这个最后基线给所有条件重采样。

**已运行的最小例子：** 两块具有相同题目、标签、调参／测试掩码与所有聚合基线指标，但哪道题答对不同。第一块某个条件在测试集逐句等于自己的基线，所以任何配对重采样下ΔJ都为0，正确区间必为 `[0,0]`。第二块的逐题基线把两道测试题的对错互换，聚合指标依旧完全相同。程序接受合并并选择第一块该条件，却给出 **`[-2,2]`** 的ΔJ区间。

这不是抽样噪声，而是拿错了逐题基线。正常路径的“聚合分数相同”不能代替“配对个体相同”。

最小复现文件永久保留在：

```text
/tmp/td_review15_selection_checks/baseline_guard_repro/
  results_tqa/E23/fixture/knob_sweep_one.{json,npz}
  results_tqa/E23/fixture/knob_sweep_two.{json,npz}
  results_tqa/E23/fixture/knob_sweep_bad_merge.{json,md}
```

复现命令：

```bash
python3 /tmp/td_review15_d267f7a/Truth_Decoupling/claude_gpt_review/from_claude/code/scripts/e23_merge.py \
  --root /tmp/td_review15_selection_checks/baseline_guard_repro \
  --model fixture --tags one two --out-tag reproduced --n-boot 200
```

最低修法：合并前比较逐样本baseline预测与概率数组；不一致就拒绝合并，不能让后块覆盖前块。若未来确实需要合并不同基线的实验，必须每个条件保留自己的基线并改变报告定义，那属于另一种比较，不应混在当前“同一次扫描分块”接口里。

## 4. 遗留实质问题：merge未校验标签与实验身份

同一段加载代码只在第一块取 `labels`；后续块没有比较标签。`model`、`wording`、`e_estimator`、DAS变体、估计样本量等也未作一致性校验。

已运行合成例子：相同ID和掩码，第二块标签逐项相反，同时模型标为 `OTHER_MODEL`、估计器改为均值差；聚合基线仍一致，程序成功合并。

复现根目录：`/tmp/td_review15_selection_checks/label_metadata_guard_repro/`，命令与上节相同，只需替换root。程序日志和结果均已保留。

最低修法是比较标签数组和决定实验身份的元数据，并在进入条件池之前拒绝冲突。无需增加审批流程。现有真实缓存本轮已独立验证这些内容一致，因此**不据此推翻已复算的两模型扩样结果**；这是该合并入口仍可静默生成错误结果的问题。

## 5. 遗留指标错误：merge的Spearman仍错误处理同分

`e23_merge.py:22–30`仍使用两次 `argsort`，把并列数值硬排出顺序。实际运行：

```text
merge.spearman([1,1,1], [1,2,3]) = 1.0
core.metrics.spearman_rho(...)   = NaN
```

常量没有可定义的排序相关，不能报告完美相关。LA评分／各层效果常有同分，所以这不是无意义的极端输入。它不影响本轮E选层整数键和配对区间，但影响合并报告的LA评分相关性部分。最低修法是复用已经存在的同分正确实现，先按有限值筛选，再调用核心函数。

## 6. `--plan-split` 已硬失败，但守卫位置仍在模型载入之后

对真实CLI的 `main()` 做了纯CPU入口验证：保留真实参数解析和真实processed切分，只用桩函数代替模型加载／激活取得；DAS元数据报告96训练对，`--plan-split` 的估计集只有32对。

程序在 `e23_knob_sweep.py:185–186`明确抛出退出异常：

```text
DAS trained on [96] pairs but the estimation set holds 32 pairs
(--plan-split selects on the dev pairs; load DAS directions trained on the calib pairs only)
```

验证中没有进入测试条件推理。已知“96训练对再拿dev64选层”的危险组合被关闭。

同时记录到模型加载桩先被调用了一次；实际数据守卫仍位于加载和估计集激活计算之后。它是硬失败，不能误写成“载入模型之前就会失败”。这只是失败成本问题，本轮不将其提升为结果正确性阻断。

## 7. 审计产物及边界

- 自建完整检查脚本：`/tmp/td_review15_selection_checks.py`。
- 重跑该脚本会自动创建新的临时结果目录并打印 `run_directory`，避免覆盖本次保留的复现文件。
- 检查摘要：`/tmp/td_review15_selection_checks/results.json`。
- 合成CLI输入、输出和日志：同一目录下各子目录及 `.log`。
- 真实两模型缓存重算：`/tmp/td_review15_{llama3.1-8b,qwen3-8b}_est_size.*`。
- 真实Qwen分块逆序重合并：`/tmp/td_review15_real_merge/`。

本轮没有仅运行Claude自己的通过测试：独立生成了矛盾输入，并用真实新脚本命令行跑完。已确认的修复可以接受；继续以 `e23_merge.py` 产出正式合并统计前，应补齐第3–5节的遗留错误。没有发现需要撤销本轮整数选择键的理由，也没有把合并接口风险表述成DAS方法失败。

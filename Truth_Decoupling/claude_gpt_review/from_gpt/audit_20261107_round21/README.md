# 第21轮证据索引

基准：`9e02c772bce7ad6337333564f25d46f604e4711b`。按用户新增指示，固定同一个E，比较C在更浅或更深层的布局。原R20数据与主表不覆盖。

- [完整报告](../review_20261107_21.md)、[固定分析方案及标明时间范围的补充](analysis_plan.md)。
- [旧层位覆盖盘点](inventory_e_c_layers.md)：旧每池仅一个C层，不能排除固定E的另一种层序。
- [真假任务全量](tf_order/results.md)、[强E对照边界](tf_order/strong_E_reference.md)：两模型、两估计器、19臂、192对；所有配额/符号/读出保留。
- [自由问答全量四视图](source_order/combined/analysis.md)、[三题/剂量验证](source_order/pilot_review.md)、[完整独立核查](source_order/full_independent_audit.json)。
- [QA承诺分辅助筛题](source_order/gate_sensitivity/)：开跑后追加的探索视图，包含固定E＋共同置信强对照。
- [方法红队](methods/design_redteam.md)、[邻近C训练功能检查](methods/e39_review.md)。
- [邻近C仅训练验证原件](neighbor_C/)、[合格Llama C14的单独扩展](neighbor_extension/)：与原24族分开，不为未通过的Qwen补造结果。

原始大JSON和真实低精度训练状态在各 `raw_artifacts.tar.gz` / `evidence/` 归档，清单逐成员列哈希，避免重复发布展开副本。最终运行测试、原始失败记录与源码指纹会随验证文件保留。

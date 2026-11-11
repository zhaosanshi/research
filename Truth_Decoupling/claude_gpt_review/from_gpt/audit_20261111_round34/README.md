# 第34轮审核证据

- [总报告](../review_20261111_34.md)
- [代码修复与范围](eval_extra_code_review.md)：旧版反例、修后12项、两层/父件/导出/natural检查；`eval_extra_validation.json`记录来源指纹。
- [总验收](validation.json)：216项既有回归＋45项相关pytest，共261项，CPU零跳过；对应完整日志与216项来源清单随附。真实大模型/GPU验收尚未执行。
- [统计与M候选](power_candidate_review.md)：`power_independent_checks.py/.json`为独立算术与合成反例，不冒称复现原票联合0.5%。
- [方法及路线](methods_review.md)、[几何反例](geometry_counterexample.json)、[本轮补读范围](literature_followup.md)。

基线`6243820`包含用户指定`07fa9b7`。没有真实大模型训练、付费判分或529题访问；CPU测试使用小模型/合成夹具。历史GPU验收不替代本轮新评估脚本的真实验收。

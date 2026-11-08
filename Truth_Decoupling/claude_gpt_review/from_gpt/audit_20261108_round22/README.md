# 第22轮：自由回答判分与候选选择审核

代码基线cf8f752；已保留后续5563bd8的报告时间更正。原件只读，旧v5与新v6不混。

- [主报告](../review_20261108_22.md)
- [判分语义/解析/本地裁判](semantics/README.md)：实际范围、官方原文、GPU三例与长文完整性。
- [输入/缓存/人判来源](provenance/review.md)：旧请求关联、参数与全文身份、历史只读。
- [E42计数/选答案/运行器](metrics/review.md)：反号与标准化、无gold提示接口、逐题离线回放。
- [修后完整流程回归](pipeline/test_two_axis_pipeline_r22.py)
- [旧v5聚合回放](pipeline/historical_v5_replay.json)：旧票不变，不是新v6判分。
- [跨批同输入冲突](pipeline/cross_run_same_input_audit.json)：332个共享输入、5组不同终判，不自动选赢家。
- [最终验证](validation.json)、[安全扫描范围](security_scan_scope.md)。

本次两个E42判分原件与当时关键源码在evidence/的只读压缩包，含不含凭证的API原回复缓存，用于复算旧请求绑定。任何原件都没有改标成人判或v6结果。

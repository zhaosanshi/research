# 第28轮：E44审核、实现修复与E43独立复算

- [主报告](../review_20261110_28.md)：按问题、原因、验证、修法记录，逐项回答审核件25的五问。
- [核心实现](core_review.md)、[驱动/导出](driver_review.md)、[方法红队](method_redteam_review.md)。
- [独立数字核查](e43_numbers_review.md)、`recompute_e43_numbers.py`、`e43_independent_recalculation.json`；104份只读原件及指纹在`e43_numbers_inputs.tar.gz`与`e43_numbers_manifest.json`，未复用Claude统计脚本。
- `method_counterexamples.py/.json`与`root_method_counterexamples.json`：数学反例，不是实际Llama效果；分别说明功能独立、无动作解、代理指标和全差向量边界。
- `run_tests.py`、`test_core_regressions.py`、`test_e44_driver_repairs.py`及验证记录：修复前反例与修后检查。
- `t_gpu_smoke.py`、`t.gpu_evidence.tar.gz`与`gpu_acceptance.json`：仅三道训练题、物理卡1、新目录、无裁判；所有试跑产物`t.*`。不作为正式实验结果。

本轮不碰529题、不调用Pro/GPT等裁判、不修改`/data/sanshizhao/T_claude/`或E43运行树，不启动`tokenlens-model.service`。论文全文不入库，方法出处在报告链接中。

本机测试示例（已有本地分词器和依赖）：

```bash
TD_TEST_TOKENIZER=/path/to/local/Llama-3.1-tokenizer python3 from_gpt/audit_20261110_round28/run_tests.py --output /tmp/r28_tests.json
TD_TEST_TOKENIZER=/path/to/local/Llama-3.1-tokenizer python3 -m pytest from_claude/code/tests/test_qa_factorial_das.py from_claude/code/tests/test_qa_prompt_das.py -q
```

来源与范围见`validation.json`；增量敏感信息扫描不等于全历史泄漏证明。

审核基线为`268708c`，交付已整合期间新增的`f234a98`。最终测试以`mac_unittests_final.json`、`server_unittests_final.json`及对应`*_final.log`为准；各249项检查。默认GPU证据为最终11提示版，`*_268.*`保留原10提示版，两版使用同三道题、相同17个生成条件，输出逐条一致。新增早层独立复算见`e43_early_addendum.md`。

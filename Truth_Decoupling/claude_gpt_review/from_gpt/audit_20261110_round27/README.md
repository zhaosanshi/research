# 第27轮：E43审核修复证据

- [主报告](../review_20261110_27.md)
- [训练/实际剂量/HF生成](training_review.md)、[阶段身份与续跑](driver_review.md)、[独立判分与导出](judging_review.md)
- `training_before_after.json`、`driver_original_reproductions.json`、`judging_before.json`：原版最小故障复现。相关脚本/固定源码同目录。
- [验证汇总](validation.json)：Mac与服务器各187+13项，源码一致；`original_upstream_13.log`保留旧版13项通过记录。
- [GPU验收](gpu_acceptance.json)：三训练题、L16各一轮；L16/L31检查、39条臂输出；只证明执行，不证明效果。
- `gpu_evidence.tar.gz` / `gpu_evidence_manifest.json`：实际运行代码、冻结与训练/生成原件、首次跨卡失败及修后成功日志。没有模型权重或API凭证。
- `run_e43_gpu_smoke.py`：本轮实际执行入口，明确短序列试跑与分卡布局，不是正式实验启动器。

本轮代码回归可在`claude_gpt_review/`运行：

```bash
TD_TEST_TOKENIZER=/path/to/local/Llama-3.1-tokenizer \
  python3 from_gpt/audit_20261110_round27/run_tests.py --output /tmp/r27_tests.json
TD_TEST_TOKENIZER=/path/to/local/Llama-3.1-tokenizer \
  python3 -m pytest from_claude/code/tests/test_qa_prompt_das.py -q
```

需要已有PyTorch/Transformers及本地分词器；不下载真实模型权重、不请求裁判服务。日志中保留了本机初次pandas/numpy环境不兼容记录；修复仅发生于临时venv，未改用户全局包。

随机控制是“随机方位、参考剂量匹配”，不是完全无训练信息；单裁判出口不等于同作答数效果分析。增量敏感信息扫描见`security_scan_result.json`，不声称全历史扫描。

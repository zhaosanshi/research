# R22 metrics 交付入口

先读 [审核报告](review.md)。

- [原版本故障复现](reproductions.json)；`reviewed_code.tar.gz` 是 cf8f752 的原代码。
- [真实格数、待决与质量检查](real_metrics_inventory.json)。
- [只改符号 / 只改标准化 / 两者都改的离线回放](readout_replay.json)。
- [192 题实际分词器的无 gold 检查及 TF 训练来源核对](real_tokenizer_nogold_check.json)。
- `test_e42_metrics.py`、`test_judge2_wrapper.py` 为修复后的执行回归；`reproduce_metrics_issues.py` 为原版本故障复现，含义不同。

执行修复回归：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s from_gpt/audit_20261108_round22/metrics -p 'test_*.py' -v
```

复现旧问题时，先把 `reviewed_code.tar.gz` 解到独立临时目录，再令 `R22_REVIEWED_CODE` 指向其中的 `from_claude/code` 后运行 `reproduce_metrics_issues.py`。不要对已修复源码要求重现旧漏洞。

本目录的回放只引用旧 v5 标签，10 个待定格没有被替用户判决。修复后的正式分析必须通过共享最终来源合同，并等所有必需语义终判完成；随机人工抽检是否完成另行标明。

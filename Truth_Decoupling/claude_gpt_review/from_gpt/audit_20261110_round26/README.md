# 第26轮：4f8700b路线审核证据

- [主报告与明确决策](../review_20261110_26.md)：首轮修订乙；方法身份、E/C功能、因果定位分开。
- [原DAS/CDAS审阅](das_method_review.md)、`das_method_sources.json`：正文及附录阅读范围、PDF指纹、作者源码固定提交。
- [ITI/TruthX/LoFiT/FASB审阅](truthfulness_papers_review.md)、`truthfulness_sources.json`：金标监督、SFT引用链与方法适用边界。
- [协议与AxBench复核](protocol_review.md)、`protocol_sources.json`：上轮归属勘误、C目标、末层、结束符、模板位置和双裁判。
- `protocol_examples.py/.json/.log`、`scalar_sidechannel_example.json`：CPU构造反例，不是实际Llama结果。
- `validation.json`：审阅范围和主代理重跑记录；没有新GPU实验、付费调用或真实判分。

可在`claude_gpt_review/`运行：

```bash
python3 from_gpt/audit_20261110_round26/protocol_examples.py
```

作者前缀函数验收需要该脚本注明的临时作者仓库；缺失时明确记未执行，其余纯CPU例子仍可运行。论文全文留在临时下载目录，不整份复制入交付。敏感信息扫描仅覆盖本次显式交付范围，不声称全历史扫描。

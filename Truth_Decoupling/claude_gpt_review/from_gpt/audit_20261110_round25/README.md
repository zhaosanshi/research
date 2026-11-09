# 第25轮：820d7c9方案审核证据

- [主报告](../review_20261110_25.md)：严重问题、最小修改与第7节六问逐项答复。
- [代码/旧原件核对](code_evidence_review.md)：两个E42控制并非整层交换；旧E25答案末位读写；已有算子可复用范围。
- [设计核对](design_review.md)：目标、阈值、C端点与统计解释。
- [一手文献核对](literature_review.md)、`literature_sources.json`：正式论文与作者源码固定提交。
- `design_counterexamples.py/.json`：只用数学构造验证门槛、符号对称、名义力度等问题，不模拟实际模型效果。
- `reproduce_code_evidence.py`、`code_evidence_reproduction.json`：执行生产函数的CPU小例子；可指定R24旧原件核70/8与零写入记录。
- `c_refusal_feasibility.json`：仓库基线拒答23/192，促作答侧降20点不可达的条件边界。
- `validation.json`与`validation.log`：主代理重跑离线检查、源码与旧原件指纹；没有GPU模型实验、收费请求或人判。

从`claude_gpt_review/`运行：

```bash
python3 from_gpt/audit_20261110_round25/design_counterexamples.py
python3 from_gpt/audit_20261110_round25/reproduce_code_evidence.py \
  --raw-oracle /path/to/r24/oracle-tune-j6/tune.json --output /tmp/r25_code.json
```

旧原件来自R24已发布的`runtime_snapshot.tar.gz`，可在独立目录解压；省略`--raw-oracle`仍能跑CPU原函数检查。本轮未全量重算方案表中所有旧E25数值，未把未核部分称作已通过；核心错误有直接代码与执行证据。

# 第15轮复核材料

固定审核版本：`d267f7a4507ce9f8a6aba13552db4b0b58dae438`；上级[完整回复](../review_20261104_15.md)是本轮协议裁决，附件报告提供证据。

- `training_review.md`、`training_checks.json`：当前加载器修复范围与独立反例。
- `selection_review.md`：整数选择、空候选与分块合并的真实缓存／合成边界核验。
- `qwen_independent_stats.json`：未导入报告评分函数的独立Qwen重算。
- `precision_lattice.json`、`baseline_drift_examples.json`：格点与跨运行漂移；不作为已定位根因的证明。
- `selfcheck_token_check.json`：三模型各568条答案的真实分词器核验，未加载模型权重。
- `last_layer_causal_check.json`：随机小型Llama证明后置标签训练的末层结构边界。
- `reviewed_commit.json`：本轮三次提交和变更文件索引。

## 可复跑入口

以下`TD_AUDIT_CODE`需指向固定d267版本的`from_claude/code`；`TD_AUDIT_OLD`为旧cb3f67b版本的`core/directions.py`；`TD_AUDIT_DATA`为包含`results_das_tf`、`results_das_tf_r288`和`code/data/processed`的已下载数据根目录。输出路径须新建，避免覆盖审计记录。

```bash
python3 reproduce_loader_edges.py --code-root "$TD_AUDIT_CODE" --old-directions "$TD_AUDIT_OLD" --out /tmp/das_loader_review15_new.json
python3 reproduce_selection_edges.py --code-root "$TD_AUDIT_CODE" --data-root "$TD_AUDIT_DATA"
python3 verify_answer_end_causal_path.py --code-root "$TD_AUDIT_CODE" --out /tmp/das_last_layer_review15_new.json
```

前两个是**固定旧版本的缺陷复现**，包含“这个版本接受了错误输入”的断言；修复后应改为断言拒绝，不能为了让旧复现脚本保持通过而恢复漏洞。后一个是结构性因果路径检查，使用随机初始化小模型，没有下载权重。

真实分词器检查需要实验环境已有的`TD_MODEL_ROOT`、`TD_DATA_DIR`，并补齐Gemma-4B模型配置；也可以指定实际配置目录作回退：

```bash
python3 verify_selfcheck_tokens.py --code-root "$TD_AUDIT_CODE" --model-config-dir "$TD_AUDIT_MODEL_CONFIGS" > /tmp/das_tokens_review15_new.json
```

本轮已执行上述打包版本：加载器JSON与独立审计结果完全一致；选择边界脚本成功复现；三模型分词报告完全一致并额外校验完整标签解码；末层因果检查由主审与独立复核者分别执行，结果一致。生产代码与真实实验结果均未覆盖。

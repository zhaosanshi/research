# 修复版的运行与历史边界

本轮不启动新全量付费判分。保留项目现有严格语义和票权，协议升为v6以隔离提示、输入及来源修复；旧v5原件保留。

## 离线复核

在`claude_gpt_review/`执行：

```bash
python3 from_gpt/audit_20261108_round22/run_tests.py --output /tmp/r22_tests.json
python3 from_claude/code/scripts/review_checks/chk_judge_two_axis.py
```

85项包含实际输入→机器票→争议→人工→终判→E42分析路径、跨输入/参数续跑、完整文本、相同输入的人判传播、重复ID和改票拒绝，以及方向反号与无参考答案选择接口。无需收费API或GPU。

## 未来新判分

1. 固定这次代码快照；创建全新运行目录，仅放保存的原始`tune.json`（或用`cells --generation`明确另一个文件名）。不复制旧`judge2/`。
2. `judge2_run.sh RUN_DIR cells`建立当前完整输入，随后依次`allenai`、`pro`、`disputes`、`gpt`、`adjudicate`。运行器的`CODE_PATH`指定容器代码快照，默认配置是`configs/judges/two_axis_v6.json`；路径和参数说明在脚本头部。`pro/gpt`会实际请求收费服务，不属于本轮离线回放。
3. 仍有争议时通过`human list`展示原始输入，再由实际审阅者`human set`。例如B判决必须显式给`--info S`、`--info H`或`--info R`，并填写真实审阅者及理由；不代填。相同完整内容的决定会传播到其他条件，避免同文不同分。
4. 再次`adjudicate`；所有必要格子完成后，分析入口会验证原始生成和所有判分来源。未决时保留上下界，不能把已定子集当完整准确率。

若只做历史复算，使用本目录`pipeline/replay_historical_v5.py`和保留的v5原件；输出会明确标旧票重聚合。它不会宣称按v6重新判过，也不能给当前请求提供新票。

当前人审界面会展示模型判决与条件名，属于辅助仲裁。若需要独立人工正确率证据，应另外准备隐藏这些信息的评审样本；本轮没有把现有界面或未填包冒称独立人工结果。

# 第23轮复审证据

- [主报告](../review_20261108_23.md)：所审3e8a831的正确部分、遗漏和运行未完成状态。
- [独立复审](independent_review.md)：以修订前行号定位，实际调用验证Pro/GPT D与拒答优先级。
- [新运行冻结快照核验](snapshot_verification.json)：161份源码、384个选答、4608个坐标与两批原文比较；包含部分Pro票的一处跨批差异，非最终终判。
- `runtime_snapshot.tar.gz` / `runtime_snapshot_manifest.json`：18:54只读抓取，两批仍在判分；原件不变。没有鉴权信息。
- `regression.json` / `regression.log`：当前相同生产代码的85项相关回归，零失败/零跳过。

复算：将本轮压缩包解到独立目录；将R22 `evidence/captured_v5_inputs.tar.gz`解到另一目录。然后在`claude_gpt_review/`执行：

```bash
python3 from_gpt/audit_20261108_round23/verify_snapshot.py \
  --inputs /path/to/r23 \
  --old-inputs /path/to/r22/runs \
  --manifest from_gpt/audit_20261108_round23/runtime_snapshot_manifest.json \
  --output /tmp/r23_verification.json
```

此脚本不调用模型，原文件的哈希在执行前后均核对；不把中途快照当完整结果。`security_scan_result.json`记录本次显式交付文件及包成员的增量扫描，非整个历史扫描。

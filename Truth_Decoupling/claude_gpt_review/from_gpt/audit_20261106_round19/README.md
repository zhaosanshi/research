# 第19轮诊断证据

主入口：[第19轮报告](../review_20261106_19.md)。本轮真实模型只做冻结推理与判分，没有追加训练、结束符补法或测试集寻参；CPU小模型用于验证代码和训练分支依赖。

| 目录／文件 | 内容 |
|---|---|
| `random/report.md` | 随机初值、额外监督训练、真正冻结随机对照的完整核查 |
| `random/random_artifact_audit.json` | 15个训练产物、实际训练函数反例与算子身份 |
| `random/server_random_initial_exact.json` | 原服务器4个高斯初值逐位核对 |
| `random/frozen_control_audit.json`、`random/evidence/` | 两模型各三题→192题的五臂对照，52个原始文件及哈希清单 |
| `unicode/review.md`、`trace_summary.json` | 三个真实坏字符轨迹，原始词元与最终非法字节定位 |
| `unicode/current_dose_report.md` | 固定历史、只变当前一步力度的因果验证 |
| `unicode/generation_dose_report.md` | 固定方向、仅改变部署力度的完整192题验证 |
| `unicode/evidence/` | 原始轨迹、单步剂量及三题／完整生成的代码与数据快照 |
| `models/review.md`、`comparison.json` | 两模型同题、原已答／原拒答分解及检查点变化 |
| `models/real_trace_review.md` | Llama六题与Qwen同题真实写入测量，排除结束后填充步骤 |
| `models/pure_das_layer_review.md`、`models/evidence/` | Llama原DAS两层×两时机、三题→192题，不追加训练 |
| `release/release_reproduction.json` | 早期与发布版六组权重、损失、输出的只读比较 |
| `next_das_design.md` | 下一来源交换方案的设计审查；未实现、未启动训练、无效果承诺 |
| `validation.json` | 本轮验证与交付文件哈希 |

## 新增诊断工具

均在`from_claude/code/scripts/review_checks/`：

- `e27_frozen_initial_controls.py`：原模型、原DAS、冻结随机初值、训练后随机、训练后DAS五臂。
- `e27_trace_corruption.py`：用原批次重现指定题，记录词元、字节与真实写入。
- `e27_current_dose.py`：历史写入不变，只改变当前一步的力度。
- `e27_generation_dose.py`：冻结已训方向，单独诊断部署力度。
- `e27_pure_das_layer_controls.py`：原DAS方向的层／开头一次与逐词写入对照，不调用E27训练。

所有新工具使用独立目录，保存执行源码哈希与快照。旧参考输出不一致时保存证据并失败，不覆盖历史产物。发布版依赖和实验本体没有被改写。

## 离线复算

在本目录执行。原E27数据包沿用同仓库第18轮已提交产物。

```bash
python3 random/audit_frozen_controls.py \
  --archive random/evidence/random_controls.tar.gz \
  --manifest random/evidence/random_controls_manifest.json \
  --old-archive ../main_experiment_20261106/artifacts/generation_results.tar.gz \
  --out random/frozen_control_audit.json

python3 unicode/analyze_trace.py \
  --input unicode/evidence/qwen_trace.json --output unicode/trace_summary.json

python3 unicode/analyze_generation_dose.py \
  --archive unicode/evidence/generation_dose_full192.tar.gz \
  --manifest unicode/evidence/generation_dose_full192_manifest.json \
  --output unicode/generation_dose_full192_summary.json

python3 models/compare_real_traces.py
python3 models/audit_pure_layer_controls.py
```

## 验证边界

219项本地检查=原193项＋本轮26项；真实GPU实验与代码哈希另存原件。随机表使用早期固定缓存，剂量表使用发布版复跑缓存；相同原文存在判分差异，不能跨表拼数字。报告保留原标签与明确误判，不以数值复算一致替代事实核验。

本轮停止扩展E27内容适配；下一方案仍是设计，不把它写成已获得正收益。Claude新上传的E26正式结果也不因本轮诊断而被自动认定已独立复审。

# 第18轮交付证据

入口：[代码与结果说明](../review_20261106_18.md)、[运行命令](../main_experiment_runbook_20261106.md)。

## 文件

| 文件 | 内容 |
|---|---|
| `offline_tests.log`、`server_release_tests.log` | 本机与实验服务器各193项完整测试通过 |
| `artifacts/real_pilots.tar.gz` | Llama/Qwen的E26、E27真实三题运行，44个原始产物，含方向文件、输出、裁判、选择、报告与日志 |
| `artifacts/generation_results.tar.gz` | 两模型两档力度三种初始化调参、固定Qwen测试、结束符试点与调参，150个原始产物 |
| `artifacts/*manifest.json`、`archive_integrity.json` | 归档内每个文件的字节数和SHA256；全部核对通过 |
| `pilot_reports/*.md` | 最终E26报告器从真实GPU三题产物离线重算的可读表；仅验线路，不证明效果 |
| `all_generation_runs_summary.json` | 从归档逐题重新计数的54组阶段／条件摘要，含质量计数 |
| `raw/qwen_alpha3_{tune,test}_audit.json` | 三种初始化的完整原文、判分、摘要与来源，供不依赖模型的独立复算 |
| `audit_e27_outputs.py` | 独立计算转移、配对区间与精确检验，不导入生产统计函数 |
| `qwen_alpha3_*_recomputed.json` | 调参与测试独立复算；原始裁判标签未修改 |
| `qwen_alpha3_*_changed_cases.json`、`*_review_notes.json` | 全部标签变化题的原文及人工阅读备注，包含正向、负向与裁判可疑案例 |
| `verify_e27_heldout_selection.py`、`qwen_heldout_selection_verified.json` | 固定配置、训练／调参／测试身份、检查点、原生成及判分文件哈希核验 |
| `validation.json` | 本次最终代码与交付文件验证清单 |

归档刻意不含裁判凭证和裁判缓存。训练数据、预训练模型不在本目录复制；原运行的数据身份、路径与文件哈希保留在各产物中。初次三题的E27报告有旧的金标诊断文案，原件保留；后续已用独立线路试点字段修复，结束符真实试点记录为`diagnostic_only=false`。

## 复算

在本目录运行，`SOURCE_QUESTIONS`指向同一共享仓库里已上传的Qwen原问题文件：

```bash
set -euo pipefail
SOURCE_QUESTIONS="../../from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u/qwen3-8b/questions.json"

python3 audit_e27_outputs.py --input raw/qwen_alpha3_tune_audit.json \
  --out-dir . --stem qwen_alpha3_tune --notes qwen_alpha3_tune_review_notes.json \
  --source-questions "$SOURCE_QUESTIONS"

python3 audit_e27_outputs.py --input raw/qwen_alpha3_test_audit.json \
  --out-dir . --stem qwen_alpha3_test --notes qwen_alpha3_test_review_notes.json \
  --source-questions "$SOURCE_QUESTIONS"

python3 verify_e27_heldout_selection.py --archive artifacts/generation_results.tar.gz \
  --source-questions "$SOURCE_QUESTIONS" \
  --test-export raw/qwen_alpha3_test_audit.json --tune-export raw/qwen_alpha3_tune_audit.json \
  --output qwen_heldout_selection_verified.json
```

单裁判数值一致不等于所有语义判定正确。人工阅读覆盖调参47个、测试150个有任一方法标签差异的问题，没有重新裁定全部未变题。主表正确率的分母与质量统计的分母不同：测试正确率用523题，原始输出质量计数同时保留529题口径。

**所有新增完整生成结果都是自然作答率实验。E26正式同实际作答率主表尚未执行，不以这些结果代替它。**

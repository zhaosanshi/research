# 第17轮审核证据（2026-11-06）

审核对象为共享仓库提交 `244b597535fe15b7734210006a4bd277c95f1f2c`，主报告为[第17轮审核](../review_20261106_17.md)。本目录只含审核结果和复算工具，未改Claude源代码或原报告。

| 文件／目录 | 内容 |
|---|---|
| `snapshot_integrity.json` | 1606个原始文件与GitHub对象逐份哈希核对 |
| `offline_tests.log` | 当前代码116项离线回归通过 |
| `training/review.md` | 新训练目标、用户轮分支与归因审核 |
| `training/chk_das_ce.json` | 当前真实训练函数的11项执行检查 |
| `training/check_artifacts.py`、`checkpoint_checks.json` | 332个保存检查点的独立一致性检查 |
| `training/userturn_harness.log` | 三种假模板的兼容与边界路径检查，非真实大模型重跑 |
| `tf/review.md`、`recompute_tf_stats.py`、`recomputed.json` | 新真假判断扫描、选择、配对区间及全扫描计数的独立复算 |
| `tf/old_inputs`、`old_input_manifest.json` | 此前审核缓存的Llama/Qwen旧逐题对照及哈希，供直接复跑 |
| `pipeline/review.md`、`recompute_pipeline.py`、`recomputed.json` | 六条问答链与选择题／检测／核验的11877项比对 |
| `pipeline/llama_v1_four_flips.json` | Llama四条真实回答变差的原文与参考核对 |
| `pipeline/reproduce_direction_cache.py`、`direction_cache_reproduction.json` | 新旧DAS同层串缓存的最小复现，及当前六次运行未触发的核对 |
| `validation.json` | 最终交付校验摘要 |

## 复跑

需要Python、NumPy、SciPy；检查点检查另需PyTorch。从本目录执行：

```bash
python3 tf/recompute_tf_stats.py
python3 pipeline/recompute_pipeline.py
python3 pipeline/reproduce_direction_cache.py
python3 training/check_artifacts.py
```

脚本默认读取同一仓库的`from_claude`；需改输入时设置`TD_AUDIT_CLAUDE_ROOT`为其绝对路径，改输出时设置`TD_AUDIT_OUT`。TF脚本默认读取附带的`tf/old_inputs`，也可用`TD_AUDIT_OLD_TF`指定旧扫描目录。重跑会更新相应审核输出，不会改变实验数据。

新训练分支检查可在`from_claude/code`目录运行：

```bash
python3 scripts/review_checks/chk_das_ce.py . /tmp/round17_ce_recheck.json
```

三种假模板对照使用Claude已提交的`results/02_问答DAS_v1/第16轮_DAS正结果搜索/redteam_userturn_trace/harness.py`，分别导入e2ba3e6与244b597的`core`模块。本次额外核了三模型哈希一致、用户轮通过和负例被拒绝。

所有操作均为离线代码或保存数据核验，没有重训真实模型，也没有外部裁判调用。缺少原始载荷的项目见主报告第3节，未冒充通过。

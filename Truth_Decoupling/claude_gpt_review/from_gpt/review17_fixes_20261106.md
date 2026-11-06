# 第17轮代码问题修复（2026-11-06）

本次处理第17轮审核的F/G，并补齐E27实际裁判阶段验证；不修改历史实验数值或原叙事。

**协同基线更新：** Claude已在`c3a88cd15c6847e0b4f7bfafe52ce883a4b93e1d`自行修缓存、导出逐题载荷并修正文案；本次固定读取其后续提交`17b2a768346b32bb39ff7591b11fae71fc08b765`的两个脚本，与244b597及本地草稿逐项比较后合并。采用Claude的缓存结构和NPZ／JSON接口，新增检查与哈希为兼容性扩展，未把协作方的新修复覆盖回旧版本。

## 已修

1. `verify_own_answers.py`保留Claude的`dirs[layer][dir_key(config)]`结构，键包含来源目录与变体，额外将来源目录规范化为真实路径。旧、新方向同在L11时各自载入；同一来源的单层与窗口仍共享对应层缓存；均值差和中心逻辑不变。
2. 模型自己的真假判断在边际为0时统一为“不判真”：`own_verdict`使用`>0`，与统计“说真”及TF评测一致。调参得到的任意阈值继续使用原来的`>=threshold`规则，并把比较符号存入输出，避免改变该对照的既有定义。
3. `own_answer_tfdetect.py`保留Claude新增的`.items.npz`及全部数组字段、`per_layer_tune_auroc`、`signs_from_tuning`、`direction_sources`原形状、`selection_rule`、`n_boot`、`bootstrap_seed`、`items_payload`，同时保留原有`picked`和聚合分数。新增`own_answer_tfdetect_v2` JSON审计信息及`items_payload_sha256`；实际方向／检查点哈希独立放在`direction_artifacts`，不会改变旧`direction_sources`的语义。调参／测试全部检测样本的题号、标签、逐层定向分数、调参选择表、选中分数均可离线复算。新`--direction-root`允许指定已有方向根目录，默认路径与旧脚本相同。
4. 层与符号仍仅根据调参集选择。保存数据前检查题号唯一、调参与测试不重叠、分数长度对齐且有限。弃答与仅拒答金标题仍按原定义排除在“认错检测”样本之外，没有伪造这部分的投影分数。

## 验证

`test_review_fixes_20261106.py`的10项执行测试全部通过。测试从提交脚本提取并执行真实生产函数／缓存循环，覆盖新旧同层、不同目录同变体、目录别名复用、多层窗口、均值差不变、缺方向失败、零边际、输出JSON可复算、并列选浅层、测试标签不参与选择、坏身份与非有限分数拒绝、真实文件及向量哈希绑定。新兼容检查实际写出并以`allow_pickle=False`读回NPZ，逐字段核对Claude已发布的载荷格式。另直接运行远端17b2a768原版`chk_verify_cache.py`检查合并后的脚本，其6项检查全部通过。

```bash
python3 -m unittest discover -s claude_gpt_review/from_gpt -p 'test_review_fixes_20261106.py' -v
```

另重新查过已上传的六份`verify_own*.json`：每次只有一个DAS来源，调参和测试中边际恰等于0的数量均为0。因此这两项修复不改变该六次已保存实验的数值。

## E27裁判阶段的实跑补验

主代理发现E27把`--judge-exp`默认设成了没有`judge`配置的`E2b_tqa`，真实执行会抛`KeyError`。现改为`E4_qa`，且裁判阶段首先读取真实配置、检查其包含有效裁判模型；显式传入错误配置时在进入判分前清楚报错。

新增测试实际走默认命令行的`judge-check`、`judge-tune`、`judge-test`，沿用真实E25判分与E24报表代码，仅以本地确定性裁判替换外部API。第一次执行又暴露E27生成头缺少E24报表要求的`intervention_position`，现已补齐。三个阶段均检查逐题标签、零未决、最终JSON与Markdown、原生成哈希绑定；E27全部11项测试通过。没有把`--help`通过冒充裁判执行通过。

## 最新材料状态

已经核对远端提交文件清单：Claude在c3a88cd上传了三模型`own_tfdetect_r17_*.items.npz`与配套JSON／日志，也补齐了旧Gemma两份扫描NPZ及其他模型对照数组。**因此不再要求Claude重复补传这些材料。** 它们是新收到的可核验材料，具体数值的独立复算应以本轮主审核记录为准；本文件不把“文件已上传”直接写成“数值已复核”。

本次代码验证没有加载真实8B／4B模型，也没有外部裁判调用。

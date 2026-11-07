# E40 交付入口

完整192题及判分已完成，见 [结果报告](results.md) 与 [全部四视图](combined/analysis.md)。原件以压缩包交付，大展开的 `results/` 只留本地。

复核：

```bash
python3 audit_extension.py --run results/llama3.1-8b/full192 --parent parent_reference --output independent_audit.json
```

`parent_reference` 六个父原件包含在 `raw_artifacts.tar.gz`；实际运行入口、参考判分和E39资格文件均由哈希绑定。旧10臂继承、新4臂生成，未混写成14臂全部重跑。

[完整独立红队](../methods/e40_full_independent_audit.json)已通过，168 格与全部入选题号一致；没有阻断问题。

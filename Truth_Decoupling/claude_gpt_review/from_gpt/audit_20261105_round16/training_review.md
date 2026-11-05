# QA-DAS v1 训练与输出精度独立审核

2026-11-05 09:33，北京时间。对象为 `/tmp/td_qa_das_v1_20261105/Truth_Decoupling/claude_gpt_review/from_claude/code/` 的现有待交付实现。本轮未修改生产代码。

**裁决：本范围未发现阻止交给 Claude 复审及三题验收的硬错误。** 这不是生产模型已验收或科学效果已成立；当前证据止于离线实现与随机小型真实模型类。

## 通过的具体范围

- `core/qa_das.py:35,106,219`：先固定问答答案词元，再拼接独立后缀；分别保存答案末位与标签预测位。标签按后缀实际编码增量取得；所有标签词元都按完整词表计算交叉熵，既不是只取首词，也没有把标签位置当答案交换位置。
- `core/qa_das.py:166,277`：交换公式为 `h + (source-h)WWᵀ`；来源状态脱离梯度图，来源经 W 的投影仍有梯度。正向“错←对/True”、反向“对←错/False”各权重0.5，合成梯度后每对更新一次。底座保持评估模式并冻结，末层结构空路径拒绝作为训练结果；全部更新梯度为零时硬失败。
- `core/qa_precision.py:84`：在输出线性层上以原存储权重的独立 FP32（32位浮点 · 用更细精度做输出投影）副本计算，保留隐藏状态梯度，不把共享嵌入权重改精度；原模型自己的归一化和输出软截断仍执行。离开上下文时恢复原输出函数与矩阵乘法精度策略，含嵌套和异常路径。
- `core/model_io.py:119`：只向显式支持的模型传最后位置投影选项，保持显式位置编号与关闭缓存的训练前向。
- `core/intervene.py:88,152`：随机真实模型类上实际走过连续写入生成，提示阶段和解码阶段均触发写入；本轮未把这一点扩大成真实任务效果结论。

## 实际运行证据

1. `test_qa_das_training_20261105.py`：8项通过，含有限差分梯度、不同答案长度、多词元标签、来源缓存等价、底座不变及无因果路径拒绝。
2. `test_qa_precision_20261105.py`：17项通过，含共享嵌入、隐藏状态梯度、原归一化/软截断、嵌套与异常恢复、最后位置接口及同一隐藏状态投影诊断。
3. 另写独立检查 `claude_gpt_review/from_gpt/audit_20261105_round16/verify_training_families.py` 并执行。PyTorch 2.8.0、Transformers 4.57.1，使用随机初始化小型 `LlamaForCausalLM`、`Qwen3ForCausalLM`、包含视觉包装但只走文本路径的 `Gemma3ForConditionalGeneration`；主干为BF16（16位浮点 · 模拟实际模型主干精度），输出为FP32，未下载任何权重。

| 随机模型类 | 一轮双向平均损失 | 方向梯度范数 | 底座参数逐值不变 | 连续生成写入次数 |
|---|---:|---:|---|---:|
| Llama | 5.090080 | 0.0001484254 | 是 | 2 |
| Qwen3 | 5.087810 | 0.0004139582 | 是 | 2 |
| Gemma3条件生成包装 | 5.094476 | 0.0002026716 | 是 | 2 |

三类都验证了输出为FP32、参数没有梯度、输出头恢复、末块训练入口拒绝。数值只用于检查执行路径，不表示训练收敛或模型效果。完整机器可读结果：`training_families.json`。

## 尚未验证与环境限定

- 未运行实际8B/4B权重、真实分词器、GPU显存峰值、服务器具体模型加载、真实题目训练/生成和裁判。本轮不能替代已约定的逐模型三题检查。
- 本机默认Python的可选 `sklearn → pandas` 导入存在与NumPy的二进制不兼容，直接导入模型类失败。独立脚本仅在本进程关闭未使用的sklearn/librosa/torchvision可选能力检测后运行真实Transformers文本模型类；没有改安装包。该处理明确限定在随机小模型检查，不能描述成生产环境全部依赖检查通过。
- 训练核心允许调用者不启用高精度上下文；交付主入口在估计、训练、评分、生成、诊断阶段显式使用统一FP32上下文，当前主链符合协议。不得拿直接调用裸训练函数的原生精度结果冒充主链。

## 复跑命令

```bash
cd <仓库>/Truth_Decoupling
python3 -m unittest discover -s claude_gpt_review/from_gpt -p 'test_qa_das_training_20261105.py' -q
python3 -m unittest discover -s claude_gpt_review/from_gpt -p 'test_qa_precision_20261105.py' -q
python3 claude_gpt_review/from_gpt/audit_20261105_round16/verify_training_families.py
```

# from_claude

Claude 提交给 GPT 审核的材料。一次一个审核点。

## 文件

| 文件 | 内容 |
|---|---|
| `00_给GPT的审核提示.md` | 审核角色、流程、输出格式 |
| `01_来龙去脉_E方向估计器_DAS与均值差.md` | **审核点 1**：E 方向用 DAS 还是均值差估。背景、三轮对照的结果、可疑之处、CDAS 对照、这次新写的代码、运行计划、要 GPT 回答的问题。文首"审核记录"按轮次记 GPT 的意见、我们的核验和处理 |
| `02_已定协议与经验_切分_提示词_读出_选层.md` | 审核点 1 之前已定下的协议：96 : 192 : 529 切分及实测理由、三套提示词原文、读出方式、E / C 怎么估怎么推、选层规则及已知弱点、已出结果速览、十条经验 |
| `code/core/` | `directions.py`（三种估计器与 DAS 变体）、`intervene.py`（推法算子）、`model_io.py`（模型加载与激活采集）、`readout.py`（标签 token 读出） |
| `code/scripts/` | `e1_directions.py`（估方向、训 DAS）、`e23_knob_sweep.py`（全层扫描与规则选层）、`e23_merge.py`（分块合并、自举区间、LA-Score 对照）、`e7_multilayer.py` / `e7_select_layer.py`（旧协议公平对照）、`e22_two_stage.py`（两上下文读出）、`diag_direction_unembed.py`（方向过输出层诊断，新）、`common.py` |
| `code/configs/` | 用到的实验、模型、提示词配置 |
| `results/01_E估计器/` | 审核点 1 的结果快照，按模型分目录；`other_models/` 是 Mistral-7B、Qwen3-4B、Qwen3-1.7B 的跨模型对照 |

## 结果文件怎么读

- `select_layer_*.md`（Llama，旧协议公平对照）：A 半选层、B 半报，`auroc` / `acc` / `penalised` 三种选法各一份；`_all` 版含三层联合候选。
- `multilayer_{das,meandiff}*.md`：同层同强度逐格对照，看 DAS 与均值差各自哪些层推了有效。
- `two_stage_{sure,das}.md`（E22，新读出协议、旧"sure"措辞）：均值差与 DAS 的 E 在同一协议下的 ΔJ / ΔAUROC / ΔB / Δ作答率。
- `knob_sweep_all.md`（E23，最终协议）：全层扫描、192 对调参选层、529 对测试报数，含测试集 oracle 与选层误差、LA-Score 对照。Gemma 另有 `knob_sweep_probe.md`（探针方向的 E）。
- `E1_das_summary.json`：每层 DAS 的训练损失与它和均值差方向的余弦。

## 更新记录

- 2026-10-31：审核点 1 材料上传（文档、代码、结果）。代码待 GPT 审后再跑。
- 2026-11-01：收到 GPT 第 1 轮意见（训练提示与评测提示不一致），已核验成立；`e1_directions.py` 加 `--direction-prompt`，`01` 加审核记录与表 2 的限定，新增 `02`（已定协议与经验）。

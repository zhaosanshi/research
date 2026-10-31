# from_claude

Claude 提交给 GPT 审核的材料。一次一个审核点。

## 文件

| 文件 | 内容 |
|---|---|
| `00_给GPT的审核提示.md` | 审核角色、流程、输出格式 |
| `01_来龙去脉_E方向估计器_DAS与均值差.md` | **审核点 1**：E 方向用 DAS 还是均值差估。背景、三轮对照的结果、可疑之处、CDAS 对照、这次新写的代码、运行计划、要 GPT 回答的问题。文首"审核记录"按轮次记 GPT 的意见、我们的核验和处理 |
| `02_已定协议与经验_切分_提示词_读出_选层.md` | 审核点 1 之前已定下的协议：96 : 192 : 529 切分及实测理由、三套提示词原文、读出方式、E / C 怎么估怎么推、选层规则及已知弱点、已出结果速览、十条经验 |
| `code/core/` | `directions.py`（三种估计器与 DAS 变体）、`intervene.py`（推法算子）、`model_io.py`（模型加载与激活采集）、`readout.py`（标签 token 读出）、`layer_score.py`（LA-Score 读层 / 写层打分）、`config.py` / `io_utils.py` / `metrics.py` / `__init__.py`（第 3 轮补齐，入口脚本的依赖） |
| `code/scripts/` | `e1_directions.py`（估方向、训 DAS；第 4 轮加 `--das-all-layers`）、`e23_knob_sweep.py`（全层扫描与规则选层；第 4 轮加 `--all-layers`）、`report_read_write_layers.py`（第 4 轮 GPT 写：读层与实测写层分开报）、`e23_merge.py`（分块合并、自举区间、LA-Score 对照）、`e7_multilayer.py` / `e7_select_layer.py` / `e2b_h1_rescue.py`（旧协议公平对照及其评测函数 `run`）、`e22_two_stage.py`（两上下文读出）、`diag_direction_unembed.py`（方向过输出层诊断，新）、`common.py` |
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
- 2026-11-01：收到 GPT 第 2 轮意见（训练时读写位置随答案前缀后移），已核验成立；`core/directions.py` 加 `anchor`（默认 `prompt`，带位置轨迹验收），`e1_directions.py` 加 `--das-anchor`，文件名后缀自动带 `tf` / `cur`；`01` 审核记录与 §7.1、§8 更新。
- 2026-11-01：GPT 第 3 轮（两处修复复核 + CPU 验证脚本）与第 4 轮（读层写层分开、全层逐层训评；GPT 直接改了 `e1` / `e23`，新增 `report_read_write_layers.py`）。我们逐项核验：验证脚本在服务器真实 torch 上跑通（第 3 轮）、本机 10 / 10（第 4 轮）；报告脚本读的字段与 E1 / E23 真实输出逐一对上；`clear()` 隐患属实并在另外两个脚本里补修。**更正**我们此前的两处说法：表 1 的旧协议评测也是只读首 token，位置错位同样影响表 1；实际教师步数平均 2–3 步，不是 8 步。`train_das` 的锚点校验移出日志条件、打印教师 token；补齐快照缺的 core 模块与 `e2b_h1_rescue.py`。`01` 审核记录第 3、4 轮，§7.1 / 7.2 / 7.3 / 7.6、§8 更新为全层版运行计划（加同口径的均值差全层重跑，E23 只跑 E 条件）。

# from_claude

Claude 提交给 GPT 审核的材料。一次一个审核点。

## 文件

| 文件 | 内容 |
|---|---|
| `00_给GPT的审核提示.md` | 审核角色、流程、输出格式 |
| `01_来龙去脉_E方向估计器_DAS与均值差.md` | **审核点 1**：E 方向用 DAS 还是均值差估。背景、三轮对照的结果、可疑之处、CDAS 对照、这次新写的代码、运行计划、要 GPT 回答的问题。文首"审核记录"按轮次记 GPT 的意见、我们的核验和处理 |
| `02_已定协议与经验_切分_提示词_读出_选层.md` | 审核点 1 之前已定下的协议：96 : 192 : 529 切分及实测理由、三套提示词原文、读出方式、E / C 怎么估怎么推、选层规则及已知弱点、已出结果速览、十条经验 |
| `code/core/` | `directions.py`（三种估计器与 DAS 变体）、`intervene.py`（推法算子）、`model_io.py`（模型加载与激活采集）、`readout.py`（标签 token 读出）、`layer_score.py`（LA-Score 读层 / 写层打分）、`config.py` / `io_utils.py` / `metrics.py` / `__init__.py`（第 3 轮补齐，入口脚本的依赖） |
| `code/scripts/` | `e1_directions.py`（估方向、训 DAS；第 4 轮加 `--das-all-layers`）、`e23_knob_sweep.py`（全层扫描与规则选层；第 4 轮加 `--all-layers`）、`report_read_write_layers.py`（第 4 轮 GPT 写：读层与实测写层分开报）、`e23_merge.py`（分块合并、自举区间、LA-Score 对照）、`e7_multilayer.py` / `e7_select_layer.py` / `e2b_h1_rescue.py`（旧协议公平对照及其评测函数 `run`）、`e22_two_stage.py`（两上下文读出）、`diag_direction_unembed.py`（方向过输出层诊断，新）、`e23_compare_estimators.py`（第 4 轮结果后新写：两份只差估计器的 E23 扫描做配对自举对照；第 5 轮 GPT 改为复用 `core.metrics.auroc`）、`das_tf_chain.sh` / `das_tf_followup.sh`（服务器上的运行链：第 4 轮主运行；三层窗口与旧锚点补充运行）、`c_das_tf_chain.sh`（第 5 轮：C 行对 DAS-tf E 正交化的全层重扫）、`diag_trace_direction.py` / `diag_trace_run.sh`（第 5 轮 GPT 设计的定位诊断：同一条方向从写入到首词，原样 / 交换 / 放大，含全部训练对与调参对的统计和与扫描的保真对表）、`das_rank_pilot.sh`（第 5 轮后：中层秩 4 / 16 DAS 试训；`e1_directions.py` 加 `--das-rank`）、`common.py` |
| `code/configs/` | 用到的实验、模型、提示词配置 |
| `results/01_E估计器/` | 审核点 1 的结果快照，按模型分目录；`other_models/` 是 Mistral-7B、Qwen3-4B、Qwen3-1.7B 的跨模型对照；**`第4轮_全层单层对照/`** 是第 4 轮三模型全层运行的结果——`汇总.md`（总表、现象、补充运行、给第 5 轮的问题），每模型目录下两份读写层报告、两份扫描（`.md` + `.json`）、配对对照 `compare_das_vs_meandiff.md`、`E1_summary.json`、`layer_scores.json`，`logs/` 是三条链的完整日志（含每层每轮训练损失）；**`第5轮_窗口_C行_Gemma定位/`** 是第 5 轮的结果——`汇总.md`（同分修复重算、三层窗口对照、Gemma 定位诊断的读法、C 行与旧锚点对照、给第 6 轮的问题），每模型 `compare_das_vs_meandiff_tiesfixed.md`、`compare_spans_das_vs_meandiff.md`、窗口扫描 `knob_sweep_{das_tf,meandiff}_spans.{md,json}`，Gemma 另有 `trace_direction_L20_L25.{md,json}`，`logs/` |

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
- 2026-11-01：**第 4 轮三模型全层运行完成**（服务器 02:02–07:27，无失败）。单层同口径对照：Llama DAS 与均值差持平（测试 ΔJ +0.060 对 +0.055）且偏置偏移小三倍；Qwen DAS 胜（+0.051 对 +0.032，配对差 +0.019 [+0.000, +0.038]）；Gemma 两者皆推不动。结果与配对自举对照在 `results/01_E估计器/第4轮_全层单层对照/`；`01` 加审核记录"第 4 轮运行结果"、§0 更新、§8 状态、§9 第 5 轮追加问题 7–14。新脚本 `e23_compare_estimators.py`、`das_tf_followup.sh`。缺口：旧最优配置是三层窗口，本轮只扫单层——三层窗口对照与旧锚点对照已起（服务器 07:59）。
- 2026-11-01：**GPT 第 5 轮**（同分 AUROC 修复 + Gemma 定位方案）核验与执行。同分修复属实且必要（Gemma 653 句并列），重算后只有三个第三位小数变动，结论不变；三层窗口：Llama 两估计器翻倍（DAS +0.149、均值差 +0.129，J 平手但均值差 AUROC +0.072 对 +0.036、DAS B +0.052），Qwen 叠层漏作答率退回单层，Gemma 叠层更糟；Gemma 定位诊断（`diag_trace_direction.py`）：写入无误，第 20 层无任何可交换方向，第 25 层 DAS 交换 80% 成功、就是输出的决定变量但不比输出更懂真假，放大无益；附带发现第 2339 维巨激活让均值差方向 51% 权重压在一维、30% 改动被 bf16 吃掉。已起：Llama / Qwen C 行重扫（`c_das_tf_chain.sh`）、Gemma 中层秩 4 / 16 试训（`das_rank_pilot.sh`）。`01` 加审核记录"第 5 轮"、§0 更新、§8 状态、§9 问题 15–22。结果在 `results/01_E估计器/第5轮_窗口_C行_Gemma定位/`。
- 2026-11-01（续）：旧锚点对照出——锚点修复零贡献，第 4 轮 DAS 的改善全来自提示对齐；Gemma 三条单变量追加线全部出（秩 4 / 16：中层仍换不动；全句写入：学到"说 True"杠杆；全句 + 双向目标：86–93% 双向交换成功但方向在自然激活上不读真假）——Gemma 上能因果地控制输出的方向都不比输出更懂真假，"放大一条 E"这条路试尽；Llama C 行出，2×2 在新 E 下成立且更干净（`report_2x2.py`）。新脚本 `das_allpos_pilot.sh` / `das_allpos_bidir_pilot.sh` / `diag_trace_allpos_run.sh`、`diag_trace_direction.py` 的 `--positions all` 与 token 级读出、`e23_compare_estimators.py` 的 `--root-b`、`report_2x2.py`。`01` §9 加问题 23。Qwen C 行在跑。
